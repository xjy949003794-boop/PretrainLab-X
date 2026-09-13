"""SwiGLU feed-forward block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """LLaMA-style gated MLP with a configurable width multiple."""

    def __init__(self, dim: int, multiple_of: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else int(8 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
