"""Standalone Day 2 RoPE shape and identity-at-position-zero check."""

import torch

from model.rope import apply_rope, rope_cache


def main() -> None:
    x = torch.randn(2, 4, 8, 16)
    cos, sin = rope_cache(8, 16, x.device)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :, :1], x[:, :, :1], atol=1e-5)
    print("test_rope: PASS", tuple(y.shape))


if __name__ == "__main__":
    main()
