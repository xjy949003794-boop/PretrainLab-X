"""Standalone Day 2 GQA forward check."""

import torch

from model.attention import GQAAttention


def main() -> None:
    module = GQAAttention(128, n_heads=16, n_kv_heads=4, max_seq_len=32)
    x = torch.randn(2, 32, 128)
    y = module(x)
    assert y.shape == x.shape
    assert module.n_rep == 4
    assert torch.isfinite(y).all()
    print("test_attention: PASS", tuple(y.shape), "gqa_repeat", module.n_rep)


if __name__ == "__main__":
    main()
