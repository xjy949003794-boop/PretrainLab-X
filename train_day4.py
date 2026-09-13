"""Run a Day 4 single-GPU scaling experiment from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from engine.trainer import Trainer
from model import LlamaConfig, LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="explicit bounded benchmark override")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    root = Path(cfg.get("output_dir", "."))
    data_dir = root / cfg.get("data_dir", "data/fineweb_10m")
    train_data = np.load(data_dir / "train_split.npy", mmap_mode="r")
    val_data = np.load(data_dir / "val_split.npy", mmap_mode="r")
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    tokens_per_step = int(cfg.get("batch_size", 1)) * int(cfg.get("max_seq_len", 2048)) * int(
        cfg.get("gradient_accumulation_steps", 1)
    )
    if args.max_steps is None and cfg.get("target_tokens") is not None and cfg.get("max_steps") is None:
        cfg["max_steps"] = math.ceil(int(cfg["target_tokens"]) / tokens_per_step)
    model_cfg = LlamaConfig.from_dict(int(data_meta["vocab_size"]), cfg)
    model = LlamaForCausalLM(model_cfg)
    print(
        json.dumps(
            {
                "model": "LlamaForCausalLM",
                "parameters": model.num_parameters(),
                "dim": model_cfg.dim,
                "layers": model_cfg.n_layers,
                "n_heads": model_cfg.n_heads,
                "n_kv_heads": model_cfg.n_kv_heads,
                "gqa_repeat": model_cfg.n_heads // model_cfg.n_kv_heads,
                "max_seq_len": model_cfg.max_seq_len,
                "target_tokens": cfg.get("target_tokens"),
                "tokens_per_optimizer_step": tokens_per_step,
                "attention_backend": model.attention_backend,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    Trainer(model, train_data, val_data, cfg, root=root, data_meta=data_meta).run(args.resume)


if __name__ == "__main__":
    main()
