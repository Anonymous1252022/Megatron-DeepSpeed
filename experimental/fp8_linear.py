# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
import torch
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Function
from megatron.core.parallel_state import get_data_parallel_world_size, get_data_parallel_group
from deepspeed.comm import all_gather


def update_amax_history(amax_history:torch.Tensor, amax:torch.Tensor) -> None:
    amax_history[1:] = amax_history[:-1].clone()
    amax_history[0] = amax



class LinearF(Function):
    @staticmethod
    def forward(
        ctx,
        input:torch.Tensor,
        weight:torch.Tensor,
        input_amax_history:torch.Tensor,
        weight_amax_history:torch.Tensor,
        grad_amax_history:torch.Tensor,
        input_scale_e4m3:torch.Tensor,
        input_scale_e5m2:torch.Tensor,
        weight_scale_e4m3:torch.Tensor,
        weight_scale_e5m2:torch.Tensor,
        grad_scale_e5m2:torch.Tensor,
    ) -> torch.Tensor:
        assert input.dtype == weight.dtype
        out_dtype = input.dtype

        if os.getenv('DISABLE_FP8', 'false').lower() == 'true':

            amax = input.abs().max().to(torch.float32)
            update_amax_history(input_amax_history, amax)
            amax = weight.abs().max().to(torch.float32)
            update_amax_history(weight_amax_history, amax)

            out = input @ weight.T

            ctx.save_for_backward(
                input,
                weight,
                grad_amax_history
            )

            return out
        else:
            input_e5m2, input_e4m3, amax = torch.ops.hpu.cast_to_fp8_hybrid(input, input_scale_e5m2, input_scale_e4m3, False, True)
            update_amax_history(input_amax_history, amax)
            weight_e5m2, weight_e4m3, amax = torch.ops.hpu.cast_to_fp8_hybrid(weight, weight_scale_e5m2, weight_scale_e4m3, False, True)
            update_amax_history(weight_amax_history, amax)

            out = torch.ops.hpu.fp8_gemm_v2(
                input_e4m3,
                False,
                weight_e4m3.T,
                False,
                None,
                out_dtype,
                1.0 / input_scale_e4m3,
                1.0 / weight_scale_e4m3,
                None,
                False
            )

            ctx.save_for_backward(
                input_e5m2,
                input_scale_e5m2,
                weight_e5m2,
                weight_scale_e5m2,
                grad_amax_history,
                grad_scale_e5m2
            )

            return out


    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None]:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        
        if os.getenv('DISABLE_FP8', 'false').lower() == 'true':
            (
                input,
                weight,
                grad_amax_history
            ) = ctx.saved_tensors

            out_dtype = grad_output.dtype

            amax = grad_output.abs().max().to(torch.float32)
            update_amax_history(grad_amax_history, amax)
            
            grad_input = grad_output @ weight

            grad_weight = grad_output.T @ input
        else:
            (
                input_e5m2,
                input_scale_e5m2,
                weight_e5m2,
                weight_scale_e5m2,
                grad_amax_history,
                grad_scale_e5m2
            ) = ctx.saved_tensors

            out_dtype = grad_output.dtype

            grad_e5m2sr, amax = torch.ops.hpu.cast_to_fp8_v2(grad_output, grad_scale_e5m2, True, True, dtype=torch.float8_e5m2)
            update_amax_history(grad_amax_history, amax)

            grad_input = torch.ops.hpu.fp8_gemm_v2(
                grad_e5m2sr,
                False, 
                weight_e5m2,
                False, 
                None, 
                out_dtype, 
                1.0 / grad_scale_e5m2, 
                1.0 / weight_scale_e5m2, 
                None, 
                False
            )

            grad_weight = torch.ops.hpu.fp8_gemm_v2(
                grad_e5m2sr.T,
                False, 
                input_e5m2,
                False, 
                None, 
                out_dtype, 
                1.0 / grad_scale_e5m2, 
                1.0 / input_scale_e5m2,
                None, 
                False
            )

        return grad_input, grad_weight, None, None, None, None, None, None, None, None



class FP8Linear(Module):
    def __init__(self) -> None:
        super().__init__()
        self.amax_history_len = int(1024 / get_data_parallel_world_size()) # TODO: Make generic.
        self.call_counter = 0
        self.update_scales = False
        self.device = torch.device('cpu')

        self.input_amax_history = torch.zeros(self.amax_history_len, dtype=torch.float32, device=self.device)
        self.weight_amax_history = torch.zeros(self.amax_history_len, dtype=torch.float32, device=self.device)
        self.grad_amax_history = torch.zeros(self.amax_history_len, dtype=torch.float32, device=self.device)

        self.input_scale_e4m3 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.input_scale_e5m2 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.weight_scale_e4m3 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.weight_scale_e5m2 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.grad_scale_e5m2 = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        fp8_linear_list.append(self)



    def set_device(self, device:torch.device) -> None:
        if self.device != device:
            self.device = device

            self.input_amax_history = self.input_amax_history.to(self.device)
            self.weight_amax_history = self.weight_amax_history.to(self.device)
            self.grad_amax_history = self.grad_amax_history.to(self.device)

            self.input_scale_e4m3 = self.input_scale_e4m3.to(self.device)
            self.input_scale_e5m2 = self.input_scale_e5m2.to(self.device)
            self.weight_scale_e4m3 = self.weight_scale_e4m3.to(self.device)
            self.weight_scale_e5m2 = self.weight_scale_e5m2.to(self.device)
            self.grad_scale_e5m2 = self.grad_scale_e5m2.to(self.device)


    def forward(self, 
                input:torch.Tensor,
                weight:torch.Tensor,
                bias:Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert bias is None # NOTE: In llama2-7b bias is None.
        device = input.device
        fp8_linear_list.set_device(device)
        fp8_linear_list.update_scales()

        A, B, C = input.shape

        out = LinearF.apply(input.reshape(A * B, C),
                            weight,
                            self.input_amax_history,
                            self.weight_amax_history,
                            self.grad_amax_history,
                            self.input_scale_e4m3,
                            self.input_scale_e5m2,
                            self.weight_scale_e4m3,
                            self.weight_scale_e5m2,
                            self.grad_scale_e5m2,
                            )
        
        self.call_counter += 1
        if self.call_counter == self.amax_history_len:
            self.update_scales = True
            self.call_counter = 0
        
        return out.reshape(A, B, -1)
    

class ScaledSwiglu:
    def __init__(self,
                 delayed:bool=False) -> None:
        self.delayed = delayed
        self.initialized = False
        self.scale = None
    
    def __call__(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.chunk(x, 2, dim=-1)

        if self.delayed:
            tmp = x[1].detach().abs().max(dim=-1, keepdim=True)[0]
            if self.initialized:
                s = self.scale.clone()
                self.scale.zero_()
            else:
                s = tmp
                self.scale = torch.zeros_like(tmp)
                self.initialized = True
            self.scale.add_(tmp)
        else:
            if os.getenv('DETACH_SCALED_SWIGLU', 'false').lower() == 'true':
                s = x[1].detach().abs().max(dim=-1, keepdim=True)[0]
            else:
                s = x[1].abs().max(dim=-1, keepdim=True)[0]
        
        tmp = x[1] / s
        return F.silu(x[0]) * tmp, s



class FP8LinearList:
    def __init__(self) -> None:
        self.fp8_linear_list = []

    def append(self, fp8_linear:FP8Linear) -> None:
        self.fp8_linear_list.append(fp8_linear)

    def set_device(self, device:torch.device) -> None:
        for fp8_linear in self.fp8_linear_list:
            fp8_linear.set_device(device)
            
    def update_scales(self) -> None:
        update_scales_list = [fp8_linear.update_scales for fp8_linear in self.fp8_linear_list]
        assert len(update_scales_list) > 0 # NOTE: If len(update_scales_list) == 0, there is some bug.

        if all(update_scales_list):
            # NOTE: This is the point where the previous gbs fully finished, and the new gbs still did not start.
            for fp8_linear in self.fp8_linear_list:
                assert fp8_linear.call_counter == 0 # NOTE: If call_counter != 0, there is some bug.

            # Aggregate all amax history buffers on single device:
            input_amax_history_list = [fp8_linear.input_amax_history for fp8_linear in self.fp8_linear_list]
            weight_amax_history_list = [fp8_linear.weight_amax_history for fp8_linear in self.fp8_linear_list]
            grad_amax_history_list = [fp8_linear.grad_amax_history for fp8_linear in self.fp8_linear_list]
            amax_history_list = input_amax_history_list + weight_amax_history_list + grad_amax_history_list
            amax_history = torch.stack(amax_history_list) # (384, 128) or (384, 1024)

            # Aggregate amax history buffers across devices:
            if get_data_parallel_world_size() > 1:
                tensor_list = [amax_history.clone() for _ in range(get_data_parallel_world_size())]
                all_gather(tensor_list, amax_history, get_data_parallel_group())
                amax_history = torch.stack(tensor_list).transpose(0, 1).reshape(len(amax_history_list), -1) # (384, 1024)

            # Filter outliers:
            amax = amax_history.max(-1)[0] # (384,)

            # Compute scales:
            scale_e4m3 = torch.pow(2.0, torch.floor(torch.log2(240.0 / (240.0 * (amax == 0) + amax)))) * (amax != 0) + (amax == 0)
            scale_e5m2 = torch.pow(2.0, torch.floor(torch.log2(57344.0 / (57344.0 * (amax == 0) + amax)))) * (amax != 0) + (amax == 0)

            # Scatter scales:
            input_scale_list = [fp8_linear.input_scale_e4m3 for fp8_linear in self.fp8_linear_list]
            weight_scale_list = [fp8_linear.weight_scale_e4m3 for fp8_linear in self.fp8_linear_list]
            scale_list = input_scale_list + weight_scale_list
            for i in range(len(scale_list)):
                scale_list[i].copy_(scale_e4m3[i])
            
            input_scale_list = [fp8_linear.input_scale_e5m2 for fp8_linear in self.fp8_linear_list]
            weight_scale_list = [fp8_linear.weight_scale_e5m2 for fp8_linear in self.fp8_linear_list]
            grad_scale_list = [fp8_linear.grad_scale_e5m2 for fp8_linear in self.fp8_linear_list]
            scale_list = input_scale_list + weight_scale_list + grad_scale_list
            for i in range(len(scale_list)):
                scale_list[i].copy_(scale_e5m2[i])
            
            for fp8_linear in self.fp8_linear_list:
                fp8_linear.update_scales = False



fp8_linear_list = FP8LinearList()