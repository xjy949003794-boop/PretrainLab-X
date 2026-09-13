"""Day 6 fresh-data pretraining with a monotonic, no-replay token stream."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from engine.checkpoint import save_checkpoint
from engine.scheduler import WarmupCosineScheduler
from model import LlamaConfig, LlamaForCausalLM
from monitor.gpu import snapshot as gpu_snapshot
from monitor.gradient import clip as clip_gradients
from monitor.logger import MetricLogger


class DatasetExhaustedError(RuntimeError):
    """Raised instead of wrapping around or replaying the fresh corpus."""


class SequentialTokenStream:
    """Read contiguous windows once, with an explicit hard exhaustion check."""

    def __init__(self, data: np.ndarray, start: int, end: int, *, seq_len: int, batch_size: int) -> None:
        self.data = data
        self.start = int(start)
        self.end = int(end)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.cursor = int(start)
        self.tokens_consumed = 0
        self.windows_read = 0

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
        span = self.batch_size * self.seq_len
        # Labels need one token after the last input. Never modulo or wrap.
        required_end = self.cursor + span + 1
        if required_end > self.end:
            raise DatasetExhaustedError(
                f"fresh train corpus exhausted at cursor={self.cursor}, end={self.end}, "
                f"requested={required_end}; replay is disabled"
            )
        starts = [self.cursor + i * self.seq_len for i in range(self.batch_size)]
        x = torch.stack([torch.from_numpy(self.data[s : s + self.seq_len].astype(np.int64, copy=False)) for s in starts])
        y = torch.stack([torch.from_numpy(self.data[s + 1 : s + self.seq_len + 1].astype(np.int64, copy=False)) for s in starts])
        first = self.cursor
        self.cursor += span
        self.tokens_consumed += span
        self.windows_read += self.batch_size
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True), {
            "data_cursor_start": first,
            "data_cursor_end": self.cursor,
            "unique_corpus_tokens_consumed": self.tokens_consumed,
        }


def _validation_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    start: int,
    end: int,
    *,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, int]:
    """Evaluate fixed non-overlapping validation windows from the held-out region."""

    was_training = model.training
    model.eval()
    values: list[float] = []
    cursor = int(start)
    used = 0
    with torch.no_grad():
        for _ in range(max(1, batches)):
            span = batch_size * seq_len
            required_end = cursor + span + 1
            if required_end > end:
                break
            starts = [cursor + i * seq_len for i in range(batch_size)]
            x = torch.stack([torch.from_numpy(data[s : s + seq_len].astype(np.int64, copy=False)) for s in starts]).to(device)
            y = torch.stack([torch.from_numpy(data[s + 1 : s + seq_len + 1].astype(np.int64, copy=False)) for s in starts]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            values.append(float(loss.detach()))
            cursor += span
            used += span
    if was_training:
        model.train()
    if not values:
        raise DatasetExhaustedError("held-out validation region is too short for one evaluation batch")
    return float(np.mean(values)), used


def _health(record: dict) -> dict:
    """Read-only health fields; no adaptive intervention is allowed on Day 6."""

    loss = float(record["loss"])
    grad = float(record["grad_norm"])
    return {
        "health/loss_finite": bool(math.isfinite(loss)),
        "health/grad_finite": bool(math.isfinite(grad)),
        "health/intervention": "none",
        "health/adaptive_monitor": False,
    }


def run(cfg: dict, *, root: Path, max_steps_override: int | None = None, smoke: bool = False) -> dict:
    data_dir = Path(cfg["fresh_data_dir"])
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    total_tokens = int(metadata["total_tokens"])
    train_tokens = int(metadata["train_tokens"])
    val_tokens = int(metadata["val_tokens"])
    data_path = data_dir / metadata["token_file"]
    data = np.memmap(data_path, mode="r", dtype=np.int32, shape=(total_tokens,))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    batch_size = int(cfg.get("batch_size", 8))
    seq_len = int(cfg.get("max_seq_len", 512))
    grad_accum_steps = int(cfg.get("gradient_accumulation_steps", 8))
    tokens_per_step = batch_size * seq_len * grad_accum_steps
    max_steps = int(max_steps_override if max_steps_override is not None else cfg.get("max_steps", 15259))
    precision = str(cfg.get("precision", "bf16"))
    amp_enabled = device.type == "cuda" and precision == "bf16"
    model_cfg = LlamaConfig.from_dict(int(metadata["vocab_size"]), cfg)
    model = LlamaForCausalLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.1))
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_steps,
        int(cfg.get("warmup_steps", 1000)),
        float(cfg.get("learning_rate", 3e-4)),
        float(cfg.get("min_learning_rate", 3e-5)),
    )
    stream = SequentialTokenStream(data, 0, train_tokens, seq_len=seq_len, batch_size=batch_size)
    log_path = root / "logs" / str(cfg.get("log_name", "day6_336m_fresh_metrics.jsonl"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "/root/day6_checkpoint_store"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = str(cfg.get("checkpoint_prefix", "day6_336m_500m_fresh"))
    start_time = time.time()
    records: list[dict] = []
    best_val = float("inf")
    best_step: int | None = None
    eval_interval = int(cfg.get("eval_interval", 1000))
    eval_batches = int(cfg.get("eval_batches", 8))
    log_interval = int(cfg.get("log_interval", 100))
    save_interval = int(cfg.get("save_interval", max_steps))

    with MetricLogger(
        log_path,
        wandb_enabled=bool(cfg.get("wandb_enabled", True)),
        wandb_project=str(cfg.get("wandb_project", "PretrainLab-X")),
        wandb_run_name=cfg.get("wandb_run_name"),
        config={**cfg, "fresh_data": True, "replay_enabled": False, "adaptive_monitor": False, "smoke": smoke},
    ) as logger:
        for step in range(1, max_steps + 1):
            lr = scheduler.step(step)
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            cursor_start = stream.cursor
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(grad_accum_steps):
                x, y, _cursor = stream.next_batch(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite train loss at step {step}")
                losses.append(float(loss.detach()))
                (loss / grad_accum_steps).backward()
            grad_norm = clip_gradients(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            elapsed = time.time() - start_time
            record = {
                "step": step,
                "loss": float(np.mean(losses)),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens": step * tokens_per_step,
                "unique_corpus_tokens_consumed": stream.tokens_consumed,
                "data_cursor_start": cursor_start,
                "data_cursor_end": stream.cursor,
                "train_tokens_available": train_tokens,
                "val_tokens_available": val_tokens,
                "tokens_per_sec": (step * tokens_per_step) / max(elapsed, 1e-6),
                "elapsed_s": elapsed,
                "precision": precision if amp_enabled else "fp32",
                "grad_accum_steps": grad_accum_steps,
                "attention_backend": getattr(model, "attention_backend", "unknown"),
                "fresh_data": True,
                "replay_enabled": False,
                "adaptive_monitor": False,
            }
            record.update(gpu_snapshot(device))
            record.update(_health(record))
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                value, val_used = _validation_loss(
                    model,
                    data,
                    train_tokens,
                    total_tokens,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    batches=eval_batches,
                    device=device,
                    amp_enabled=amp_enabled,
                )
                record["val_loss"] = value
                record["validation_tokens_evaluated"] = val_used
                if value < best_val:
                    best_val = value
                    best_step = step
            logger.log(record)
            records.append(record)
            if step % log_interval == 0 or step == 1 or step == max_steps:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % save_interval == 0 or step == max_steps:
                save_checkpoint(
                    ckpt_dir / f"{checkpoint_prefix}_step_{step:05d}.pt",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler_state=scheduler.state_dict(),
                    config={**cfg, "fresh_data": True, "replay_enabled": False, "adaptive_monitor": False},
                    data_meta=metadata,
                    last_record=record,
                    extra_state={"data_cursor": stream.cursor, "tokens_consumed": stream.tokens_consumed},
                )
    last = records[-1]
    val_values = [float(r["val_loss"]) for r in records if r.get("val_loss") is not None]
    summary = {
        "run": "day6_336m_500m_fresh",
        "smoke": smoke,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "param_count": model.num_parameters(),
        "max_steps": max_steps,
        "end_step": int(last["step"]),
        "tokens_per_optimizer_step": tokens_per_step,
        "processed_tokens": int(last["tokens"]),
        "unique_corpus_tokens_consumed": int(last["unique_corpus_tokens_consumed"]),
        "train_tokens_available": train_tokens,
        "val_tokens_available": val_tokens,
        "final_loss": float(last["loss"]),
        "final_val_loss": float(last["val_loss"]) if last.get("val_loss") is not None else None,
        "best_val_loss": min(val_values) if val_values else None,
        "best_val_step": best_step,
        "train_val_gap": (float(last["loss"]) - float(last["val_loss"])) if last.get("val_loss") is not None else None,
        "epochs_equivalent": int(last["unique_corpus_tokens_consumed"]) / train_tokens,
        "elapsed_s": time.time() - start_time,
        "mean_tokens_per_sec": float(np.mean([r["tokens_per_sec"] for r in records])),
        "peak_memory_mb": max(float(r["peak_memory_mb"]) for r in records if r.get("peak_memory_mb") is not None),
        "precision": last["precision"],
        "grad_accum_steps": grad_accum_steps,
        "attention_backend": last["attention_backend"],
        "fresh_data": True,
        "replay_enabled": False,
        "adaptive_monitor": False,
        "data_cursor_final": int(stream.cursor),
        "data_cursor_monotonic": all(
            records[i]["data_cursor_start"] >= records[i - 1]["data_cursor_end"] for i in range(1, len(records))
        ),
        "log_path": str(log_path),
        "checkpoint_dir": str(ckpt_dir),
    }
    summary_path = root / str(cfg.get("summary_name", "day6_336m_500m_fresh_summary.json"))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, root=Path(cfg.get("output_dir", ".")), max_steps_override=args.max_steps, smoke=args.smoke)


if __name__ == "__main__":
    main()
