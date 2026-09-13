"""Measure training throughput and memory across batch sizes.

This is intentionally a bounded benchmark.  It reports OOM as data instead of
turning an out-of-memory trial into a failed Day 5 run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Keep both `python -m experiments.benchmark` and the more convenient
# `python experiments/benchmark.py` entry points working from the project
# root.  The latter otherwise puts only `experiments/` on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.trainer import _batch
from model import LlamaConfig, LlamaForCausalLM
from monitor.gpu import snapshot


def _bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def run_one(
    *,
    cfg: dict,
    train_data: np.ndarray,
    data_meta: dict,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    measure_steps: int,
    checkpointing: bool,
    device: torch.device,
) -> dict:
    trial = dict(cfg)
    trial["gradient_checkpointing"] = checkpointing
    trial["max_seq_len"] = seq_len
    model_cfg = LlamaConfig.from_dict(int(data_meta["vocab_size"]), trial)
    model = LlamaForCausalLM(model_cfg).to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(trial.get("seed", 1337)) + batch_size)
    result: dict = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "checkpointing": checkpointing,
        "parameters": model.num_parameters(),
        "device": str(device),
        "precision": "bf16" if device.type == "cuda" else "fp32",
        "status": "ok",
    }
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial.get("learning_rate", 3e-4)))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        for _ in range(max(0, warmup_steps)):
            optimizer.zero_grad(set_to_none=True)
            x, y = _batch(train_data, batch_size, seq_len, device, generator)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses: list[float] = []
        for _ in range(max(1, measure_steps)):
            optimizer.zero_grad(set_to_none=True)
            x, y = _batch(train_data, batch_size, seq_len, device, generator)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        processed = batch_size * seq_len * max(1, measure_steps)
        result.update(
            {
                "measure_steps": max(1, measure_steps),
                "elapsed_s": elapsed,
                "tokens": processed,
                "tokens_per_sec": processed / max(elapsed, 1e-9),
                "mean_loss": float(np.mean(losses)),
                "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else None,
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2)
                if device.type == "cuda"
                else None,
            }
        )
        result.update(snapshot(device))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.update({"status": "oom", "error": str(exc).splitlines()[0]})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--checkpointing", type=_bool, default=False)
    parser.add_argument("--out", default="report/throughput_benchmark.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg.get("output_dir", "."))
    data_dir = root / cfg.get("data_dir", "data/fineweb_10m")
    train_data = np.load(data_dir / "train_split.npy", mmap_mode="r")
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    seq_len = int(args.seq_len or cfg.get("benchmark_seq_len", cfg.get("max_seq_len", 512)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for raw in str(args.batch_sizes).split(","):
        batch_size = int(raw.strip())
        rows.append(
            run_one(
                cfg=cfg,
                train_data=train_data,
                data_meta=data_meta,
                batch_size=batch_size,
                seq_len=seq_len,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                checkpointing=args.checkpointing,
                device=device,
            )
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {
        "config": str(args.config),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "data": data_meta,
        "checkpointing": args.checkpointing,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
