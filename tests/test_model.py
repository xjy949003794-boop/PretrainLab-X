"""Standalone Day 2 complete-model forward check."""

import torch

from model import LlamaConfig, LlamaForCausalLM


def main() -> None:
    cfg = LlamaConfig(vocab_size=512, dim=128, n_layers=2, n_heads=8, n_kv_heads=2, max_seq_len=32, multiple_of=32)
    model = LlamaForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    print("test_model: PASS", tuple(logits.shape), "params", model.num_parameters())


if __name__ == "__main__":
    main()
