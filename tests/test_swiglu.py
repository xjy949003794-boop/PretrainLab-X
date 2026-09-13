"""Standalone Day 2 SwiGLU forward/backward check."""

import torch

from model.swiglu import SwiGLU


def main() -> None:
    module = SwiGLU(64, multiple_of=16)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = module(x)
    y.square().mean().backward()
    assert y.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("test_swiglu: PASS", tuple(y.shape), "hidden", module.hidden_dim)


if __name__ == "__main__":
    main()
