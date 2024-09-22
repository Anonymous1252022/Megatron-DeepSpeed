# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import math
from typing import Callable, Iterable, Tuple, Tuple, List
from enum import Enum, auto
import torch
import torch.distributed
from torch.optim import Optimizer
from habana_frameworks.torch import core as htcore
from habana_frameworks.torch.utils.internal import is_lazy
from megatron.core.parallel_state import get_data_parallel_world_size


FP16_MAX = 65504.0
FP8_E4M3_MAX = 240.0
FP8_E5M2_MAX = 57344.0

class ROUND_MODE(Enum):
    RNE = auto()
    SR = auto()

def to_bf16(t:torch.Tensor, round_mode:ROUND_MODE=ROUND_MODE.RNE) -> torch.Tensor:
    result = None

    if round_mode == ROUND_MODE.RNE:
        result =  t.to(torch.bfloat16)
    
    if round_mode == ROUND_MODE.SR:
        # Use the following link to check representations:
        # https://www.h-schmidt.net/FloatConverter/IEEE754.html
        device = t.device
        x = t.abs().cpu()

        # '0 01111111 0000000 0000000000000000'
        q_1 = torch.tensor(1.0, dtype=torch.float32) # 0x3F800000
        # '0 01111111 0000001 0000000000000000'
        q_2 = torch.tensor(1.0078125, dtype=torch.float32) # 0x3F810000

        # Example:
        # Before: '0 01111110 1111010 1110000101011000'
        # After:  '0 01111111 0000000 1110000101011000'
        x_ = ((x.view(torch.int32) & 0x0000FFFF) | 0x3F800000).view(torch.float32).to(device)
        p = (x_ - q_1) / (q_2 - q_1)
        c = torch.bernoulli(p)
        
        # '0 01111110 1111010 1110000101011000'
        # '0 01111110 1111010 0000000000000000'
        # '0 01111110 1111010'
        down = (x.view(torch.int32) & 0x7FFF0000).view(torch.float32).to(device).to(torch.bfloat16)

        # '0 01111110 1111010 1110000101011000'
        # '0 01111110 1111010 1111111111111111'
        # '0 01111110 1111011'
        up = ((x.view(torch.int32) & 0x7FFF0000) | 0x0000FFFF).view(torch.float32).to(device).to(torch.bfloat16)

        result = c * up + (1.0 - c) * down
        sign = - 1.0 * (t < 0) + 1.0 * (t >= 0)
        result = sign * result

    return result

def synthetic_sftz(t:torch.Tensor, scale:torch.Tensor, format:str='e5m2') -> torch.Tensor:
    # t is before scaling
    if format == 'e5m2':
        msn = 2**-16 # min subnormal
    else: # format == 'e4m3'
        msn = 2**-9 # min subnormal
    s_msn = msn / scale
    mask = (t.abs() <= s_msn)
    t_p = (t > 0) * mask * t
    t_n = - ((t < 0) * mask * t)
    x = s_msn * torch.rand(t.shape, device=t.device)
    t_p = s_msn * (x < t_p)
    t_n = s_msn * (x < t_n)
    t = t * (1 - mask) + t_p - t_n
    return t


class QuantDequant:
    def __init__(self, dtype:str) -> None:
        '''
            dtype: ["fp32", "bf16", "bf16_sr", "fp16", "fp16_pts", "fp8_e4m3", "fp8_e5m2", "fp8_e4m3_sr", "fp8_e5m2_sr"]
        '''
        self.dtype = dtype.split('_')[0] # fp32, bf16, fp16, fp8
        self.format = None
        if self.dtype == 'fp8':
            self.format = dtype.split('_')[1] # e4m3, e5m2
        self.pts = ('pts' in dtype) or ('fp8' in dtype)
        self.sr = 'sr' in dtype


    def __call__(self, t:torch.Tensor) -> torch.Tensor:
        assert t.dtype == torch.float32

        if self.dtype == 'fp32':
            return t

        if self.dtype == 'bf16':
            down_cast = to_bf16(t, round_mode=ROUND_MODE.SR if self.sr else ROUND_MODE.RNE)
            htcore.mark_step()
            up_cast = down_cast.to(torch.float32)
            return up_cast
            
        if self.dtype == 'fp16':
            if self.pts:
                amax = t.abs().max().item()
                if amax > 0.0:
                    scale = FP16_MAX / amax
                    t = scale * t
                    t = t.clip(-FP16_MAX, FP16_MAX)
                    down_cast = t.to(torch.float16)
                    htcore.mark_step()
                    up_cast = down_cast.to(torch.float32)
                    up_cast = up_cast / scale
                    return up_cast
                else:
                    return t
            else:
                down_cast = t.to(torch.float16)
                htcore.mark_step()
                up_cast = down_cast.to(torch.float32)
                return up_cast

        if self.dtype == 'fp8':
            dtype = torch.float8_e5m2 if self.format == 'e5m2' else torch.float8_e4m3fn
            max_ = FP8_E5M2_MAX if self.format == 'e5m2' else FP8_E4M3_MAX

            amax = t.abs().max().item()
            if amax > 0.0:
                scale = max_ / amax
                if self.sr and (os.getenv('SFTZ', 'false').lower() == 'true'):
                    t = synthetic_sftz(t, scale, self.format)

                down_cast = torch.ops.hpu.cast_to_fp8_v2(t, scale, self.sr, False, dtype)[0]
                htcore.mark_step()
                up_cast = down_cast.to(torch.float32)
                up_cast = up_cast / scale
                return up_cast
            else:
                return t



def divide_list(list_:List[int], value:int) -> Tuple[List[int], List[int]]:
    assert (value <= sum(list_)) & (value >= 0)
    s = -1
    for i in range(len(list_)):
        if (sum(list_[:i+1]) >= value) and (s == -1):
            s = i
    
    list_1 = list_[:s]
    residual = list_[s]
    list_2 = list_[s+1:]

    if sum(list_1) + residual == value:
        list_1.append(residual)
    else: # (sum(list_1) + residual > value) & (sum(list_1) < value)
        left_chunk = value - sum(list_1)
        right_chunk = residual - (value - sum(list_1))
        if left_chunk > 0:
            list_1.append(left_chunk)
        list_2.insert(0, right_chunk)
    
    return list_1, list_2


def derive_sizes(sizes:List[int], rank:int, num_ranks:int) -> List[int]:
    chunk_size = sum(sizes) // num_ranks
    tmp = divide_list(sizes, rank * chunk_size)[1]
    return divide_list(tmp, chunk_size)[0]



class FP8FusedAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

        self.neg_step_list = []
        self.is_lazy = is_lazy()
        self.modified_wd_list = []
        
        # TODO: Check with DP > 1.
        self.sizes = []
        for group in self.param_groups:
            self.sizes.append([])
            for p in group['params']:
                self.sizes[-1].append(p.numel())


    def step_wrap(step_func):
        def wrap_(*args, **kwargs):
            result = step_func(*args, **kwargs)
            htcore.step_closure._mark_step_if_lazy()
            return result

        return wrap_

    @step_wrap
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.neg_step_list.clear()
        self.modified_wd_list.clear()

        for i, group in enumerate(self.param_groups):
            htcore.step_closure._mark_step_if_lazy()
            grad_list, wt_list, exp_avg_list, exp_avg_sq_list = [], [], [], []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                weight = p.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(p.data.shape).to(p.dtype).to(p.device)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(p.data.shape).to(p.dtype).to(p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                grad_list.append(grad)
                wt_list.append(weight)
                exp_avg_list.append(exp_avg)
                exp_avg_sq_list.append(exp_avg_sq)

            if len(wt_list) > 0:
                beta1, beta2 = group["betas"]
                if "step" in group:
                    group["step"] += 1
                else:
                    group["step"] = 1

                bias_correction_key = None
                if "bias_correction" in group.keys():
                    bias_correction_key = "bias_correction"
                else:
                    print("FusedAdamW: key 'bias_correction' not found. using 'correct_bias' instead")
                    print("This might occur when loading old checkpoints.")
                    bias_correction_key = "correct_bias"

                bias_correction = 1 if group[bias_correction_key] else 0

                step_size = group["lr"]
                if bias_correction:
                    bias_correction1 = 1.0 - pow(beta1, group["step"])
                    bias_correction2 = 1.0 - pow(beta2, group["step"])
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                neg_step = -step_size
                neg_step_t = (
                    torch.tensor([neg_step], dtype=torch.float, requires_grad=False)
                    .to(wt_list[0].dtype)
                    .to(wt_list[0].device, non_blocking=True)
                )
                self.neg_step_list.append(neg_step_t)

                # since lr is fed into the kernel as tensor, perform the scalar multiplication of wd here
                # NOTE: TODO if lr is updated every step, then we need to convert it as tensor and
                # perform weight decay unconditonally.
                modified_wd = 1.0 - group["weight_decay"] * group["lr"]

                if self.is_lazy:
                    torch.ops.hpu.optimizer_adamw(
                        grad_list,
                        wt_list,
                        exp_avg_list,
                        exp_avg_sq_list,
                        neg_step_t,
                        beta1,
                        beta2,
                        group["eps"],
                        modified_wd,
                    )
                else:
                    modified_wd_t = (
                        torch.tensor(
                            [modified_wd], dtype=torch.float, requires_grad=False
                        )
                        .to(wt_list[0].dtype)
                        .to(wt_list[0].device, non_blocking=True)
                    )
                    self.modified_wd_list.append(modified_wd_t)

                    torch.ops.hpu.optimizer_adamw(
                        grad_list,
                        wt_list,
                        exp_avg_list,
                        exp_avg_sq_list,
                        neg_step_t,
                        beta1,
                        beta2,
                        group["eps"],
                        modified_wd_t,
                        modified_wd != 1.0,
                    )
                

                if i == 0: # NOTE: Only self.param_groups[0] is quantized.
                    persistent_tensors = [wt_list, exp_avg_list, exp_avg_sq_list]
                    quantdequants = [QuantDequant(os.getenv('MW_DTYPE')), 
                                    QuantDequant(os.getenv('M1_DTYPE')), 
                                    QuantDequant(os.getenv('M2_DTYPE'))]

                    for persistent_tensor, quantdequant in zip(persistent_tensors, quantdequants):
                        tmp = persistent_tensor[0].clone()

                        if get_data_parallel_world_size() > 1:
                            sizes = derive_sizes(self.sizes[0], torch.distributed.get_rank(), get_data_parallel_world_size())
                        else:
                            sizes = self.sizes[0]

                        start = 0
                        for size in sizes:
                            t = tmp[start:start + size]
                            t = quantdequant(t)
                            tmp[start:start + size] = t
                            start += size

                        persistent_tensor[0].zero_()
                        persistent_tensor[0].add_(tmp)


        return loss
