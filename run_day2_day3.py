"""Run the Day 2/3 modular model and training engine."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from engine.trainer import Trainer
from model import LlamaConfig, LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day2_day3.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    root = Path(cfg.get("output_dir", "."))
    data_dir = root / cfg.get("data_dir", "data/tokenized")
    train_data = np.load(data_dir / "train_split.npy", mmap_mode="r")
    val_data = np.load(data_dir / "val_split.npy", mmap_mode="r")
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    model_cfg = LlamaConfig.from_dict(int(data_meta["vocab_size"]), cfg)
    model = LlamaForCausalLM(model_cfg)
    print(
        json.dumps(
            {
                "model": "LlamaForCausalLM",
                "parameters": model.num_parameters(),
                "n_heads": model_cfg.n_heads,
                "n_kv_heads": model_cfg.n_kv_heads,
                "gqa_repeat": model_cfg.n_heads // model_cfg.n_kv_heads,
                "attention_backend": model.attention_backend,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    Trainer(model, train_data, val_data, cfg, root=root, data_meta=data_meta).run(args.resume)


if __name__ == "__main__":
    main()
