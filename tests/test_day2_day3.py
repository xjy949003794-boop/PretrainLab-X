"""Fast CPU checks for model components and observatory logic."""

from __future__ import annotations

import json

import torch

from model import LlamaConfig, LlamaForCausalLM
from monitor.observatory import diagnose


def main() -> None:
    cfg = LlamaConfig(vocab_size=128, dim=64, n_layers=2, n_heads=8, n_kv_heads=2, max_seq_len=32, multiple_of=16)
    model = LlamaForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(loss)
    assert model.layers[0].attention.n_rep == 4
    good = diagnose([{"step": 1, "loss": 3.0, "lr": 1e-4, "grad_norm": 1.0}, {"step": 2, "loss": 2.0, "lr": 9e-5, "grad_norm": 1.1}])
    assert good["passed"] and good["loss_decreased"]
    print(json.dumps({"passed": True, "logits_shape": list(logits.shape), "gqa_repeat": 4}))


if __name__ == "__main__":
    main()
