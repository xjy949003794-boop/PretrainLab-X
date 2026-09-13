"""Explicit PyTorch FSDP setup for a single-node multi-GPU experiment."""

from __future__ import annotations

import os
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model.llama import TransformerBlock


def initialize_distributed() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP experiment requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, world_size


def wrap_fsdp(model: torch.nn.Module) -> FSDP:
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
