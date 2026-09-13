"""RMSNorm: the normalization used by the LLaMA family."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulating the square in fp32 avoids avoidable bf16 underflow.
        rms_inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms_inv.to(dtype=x.dtype)) * self.weight
