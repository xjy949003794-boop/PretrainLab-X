"""Standalone Day 2 RMSNorm check."""

import torch

from model.rmsnorm import RMSNorm


def main() -> None:
    module = RMSNorm(32)
    x = torch.randn(2, 7, 32)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print("test_rmsnorm: PASS", tuple(y.shape))


if __name__ == "__main__":
    main()
