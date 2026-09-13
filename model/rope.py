"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch


def rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine/sine pairs shaped for [batch, heads, sequence, dim]."""
    if head_dim % 2:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs in x, where x is [batch, heads, sequence, head_dim]."""
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
