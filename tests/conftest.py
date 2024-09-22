# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# import os
import pytest

from deepspeed.accelerator import get_accelerator

# from megatron import initialize_megatron
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.core.transformer.transformer_config import TransformerConfig

import torch

world_size = 1
rank = 0
torch.distributed.init_process_group(
    backend=get_accelerator().communication_backend_name(),
    world_size=world_size, rank=rank)

tp_world_size = world_size
pp_world_size = world_size
assert world_size == (tp_world_size * pp_world_size)

# initialize model parallel for tests
parallel_state.set_tensor_model_parallel_world_size(tp_world_size)
parallel_state.set_tensor_model_parallel_rank(rank)
# parallel_state._set_global_memory_buffer()
parallel_state.set_pipeline_model_parallel_world_size(pp_world_size)
parallel_state.set_pipeline_model_parallel_rank(rank)
parallel_state.initialize_model_parallel()

model_parallel_cuda_manual_seed(123)

num_layers = 2
hidden_size = 12
num_attention_heads = 4
use_cpu_initialization = True
# seq_len = 16
# tokenizer_type = 'GPT2BPETokenizer'
# data_dir = os.getenv("HL_DATA_DIR_ROOT", "")

# external_args = {}
# external_args.update({"micro_batch_size": 1})
# external_args.update({"num_layers": num_layers})
# external_args.update({"hidden_size": hidden_size})
# external_args.update({"num_attention_heads": num_attention_heads})
# external_args.update({"seq_length": seq_len})
# external_args.update({"max_position_embeddings": seq_len})
# external_args.update({'tokenizer_type': tokenizer_type})
# external_args.update({'vocab_file': os.path.join(data_dir, "vocab.json")})
# external_args.update({'merge_file': os.path.join(data_dir, "merges.txt")})

# initialize_megatron(ignore_unknown_args=True, external_args=external_args)


@pytest.fixture
def transformer_config():
    print(f"transformer_config")
    return TransformerConfig(num_layers=num_layers, hidden_size=hidden_size,
                             num_attention_heads=num_attention_heads,
                             use_cpu_initialization=use_cpu_initialization)
