"""Run a small, auditable FSDP training experiment with torchrun."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from distributed.fsdp import cleanup_distributed, initialize_distributed, wrap_fsdp
from engine.trainer import _batch
from model import LlamaConfig, LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    local_rank, world_size = initialize_distributed()
    rank = torch.distributed.get_rank()
    seed = int(cfg.get("seed", 1337)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(cfg.get("output_dir", "."))
    data_dir = root / cfg.get("data_dir", "data/fineweb_10m")
    train_data = np.load(data_dir / "train_split.npy", mmap_mode="r")
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    model_cfg = LlamaConfig.from_dict(int(data_meta["vocab_size"]), cfg)
    model = wrap_fsdp(LlamaForCausalLM(model_cfg))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    max_steps = int(cfg.get("max_steps", 8))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    batch_size = int(cfg.get("batch_size", 1))
    seq_len = int(cfg.get("max_seq_len", 512))
    records: list[dict] = []
    start_time = time.time()
    model.train()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(grad_accum):
            x, y = _batch(train_data, batch_size, seq_len, torch.device("cuda", local_rank), generator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            losses.append(float(loss.detach()))
            (loss / grad_accum).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0))))
        optimizer.step()
        local_loss = torch.tensor(sum(losses) / len(losses), device="cuda")
        torch.distributed.all_reduce(local_loss, op=torch.distributed.ReduceOp.AVG)
        if rank == 0:
            record = {
                "step": step,
                "loss": float(local_loss),
                "grad_norm": grad_norm,
                "world_size": world_size,
                "tokens_per_step": batch_size * seq_len * grad_accum * world_size,
                "elapsed_s": time.time() - start_time,
                "precision": "bf16",
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    if rank == 0:
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "fsdp_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        summary = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "world_size": world_size,
            "local_rank": local_rank,
            "steps": max_steps,
            "first_loss": records[0]["loss"] if records else None,
            "last_loss": records[-1]["loss"] if records else None,
            "elapsed_s": time.time() - start_time,
            "precision": "bf16",
            "sharding": "FullyShardedDataParallel",
        }
        (root / "fsdp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
