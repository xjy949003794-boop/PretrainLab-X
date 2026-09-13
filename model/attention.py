"""Grouped-query causal attention using PyTorch's fused SDPA interface."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rope import apply_rope, rope_cache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [B, kv_heads, T, D] to [B, query_heads, T, D]."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class GQAAttention(nn.Module):
    """Grouped-query attention (GQA) with RoPE and causal SDPA.

    ``scaled_dot_product_attention`` lets the installed PyTorch select a
    fused Flash or memory-efficient kernel when the device and dtype support it;
    the experiment records this as ``sdpa_auto`` rather than claiming a kernel
    that was not directly profiled.
    """

    backend_name = "sdpa_auto"

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_base = rope_base
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.max_seq_len}")
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(t, self.head_dim, x.device, self.rope_base)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.dim)
        return self.o_proj(y)
