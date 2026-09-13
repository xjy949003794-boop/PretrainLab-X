"""A compact but production-shaped single-GPU pretraining trainer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .scheduler import WarmupCosineScheduler
from monitor.adaptive_health import AdaptiveTrainingMonitor
from monitor.gradient import clip as clip_gradients
from monitor.gpu import snapshot as gpu_snapshot
from monitor.logger import MetricLogger


def _batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64, copy=False)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64, copy=False)) for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    generator: torch.Generator,
    amp_enabled: bool,
) -> float:
    was_training = model.training
    model.eval()
    values = []
    for _ in range(max(1, batches)):
        x, y = _batch(data, batch_size, seq_len, device, generator)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            _, loss = model(x, y)
        values.append(float(loss.detach()))
    if was_training:
        model.train()
    return float(np.mean(values))


class Trainer:
    """Trainer with BF16 autocast, accumulation, clipping, resume and JSONL logs."""

    def __init__(
        self,
        model: nn.Module,
        train_data: np.ndarray,
        val_data: np.ndarray,
        config: dict,
        *,
        root: str | Path = ".",
        data_meta: dict | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = dict(config)
        self.root = Path(root)
        self.data_meta = data_meta or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = int(self.cfg.get("batch_size", 8))
        self.seq_len = int(self.cfg.get("max_seq_len", 256))
        self.grad_accum_steps = int(self.cfg.get("gradient_accumulation_steps", self.cfg.get("grad_accum_steps", 1)))
        self.max_steps = int(self.cfg.get("max_steps", 100))
        self.precision = str(self.cfg.get("precision", "bf16"))
        self.amp_enabled = self.device.type == "cuda" and self.precision == "bf16"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.cfg.get("weight_decay", 0.1)),
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            self.max_steps,
            int(self.cfg.get("warmup_steps", 0)),
            float(self.cfg.get("learning_rate", 3e-4)),
            float(self.cfg.get("min_learning_rate", 3e-5)),
        )
        self.rng = torch.Generator().manual_seed(int(self.cfg.get("seed", 1337)) + 1)
        self.eval_rng = torch.Generator().manual_seed(int(self.cfg.get("seed", 1337)) + 2)
        self.log_path = self.root / "logs" / str(self.cfg.get("log_name", "day2_day3_metrics.jsonl"))
        # Keep large run-specific checkpoints on a configurable volume.  The
        # project root can live on a nearly-full persistent data disk, while
        # a system/local volume may have enough headroom for the atomic
        # temporary file created during ``torch.save``.
        checkpoint_dir = self.cfg.get("checkpoint_dir", "checkpoints")
        self.ckpt_dir = Path(checkpoint_dir)
        if not self.ckpt_dir.is_absolute():
            self.ckpt_dir = self.root / self.ckpt_dir
        self.checkpoint_prefix = str(self.cfg.get("checkpoint_prefix", "day2_day3"))
        self.summary_name = str(self.cfg.get("summary_name", "day2_day3_summary.json"))
        self.health_monitor = None
        if bool(self.cfg.get("adaptive_monitor", False)):
            self.health_monitor = AdaptiveTrainingMonitor(
                warmup_records=int(self.cfg.get("health_warmup_records", 5)),
                loss_spike_ratio=float(self.cfg.get("health_loss_spike_ratio", 3.0)),
                max_grad_norm=float(self.cfg.get("health_max_grad_norm", 10.0)),
                min_gpu_utilization_pct=float(self.cfg.get("health_min_gpu_utilization_pct", 15.0)),
                throughput_drop_ratio=float(self.cfg.get("health_throughput_drop_ratio", 0.5)),
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self, resume: str | Path | None = None) -> dict:
        start_step = 0
        if resume:
            payload = load_checkpoint(
                resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            start_step = int(payload.get("step", 0))
            extra_state = payload.get("extra_state", {})
            if extra_state.get("batch_rng") is not None:
                self.rng.set_state(extra_state["batch_rng"].cpu())
            if extra_state.get("eval_rng") is not None:
                self.eval_rng.set_state(extra_state["eval_rng"].cpu())
        mode = "a" if resume else "w"
        start_time = time.time()
        records = []
        self.model.train()
        with MetricLogger(
            self.log_path,
            append=bool(resume),
            wandb_enabled=bool(self.cfg.get("wandb_enabled", False)),
            wandb_project=str(self.cfg.get("wandb_project", "PretrainLab-X")),
            wandb_run_name=self.cfg.get("wandb_run_name"),
            config=self.cfg,
        ) as metric_logger:
            for step in range(start_step + 1, self.max_steps + 1):
                lr = self.scheduler.step(step)
                self.optimizer.zero_grad(set_to_none=True)
                losses = []
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                for _ in range(self.grad_accum_steps):
                    x, y = _batch(self.train_data, self.batch_size, self.seq_len, self.device, self.rng)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled):
                        _, loss = self.model(x, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
                    losses.append(float(loss.detach()))
                    (loss / self.grad_accum_steps).backward()
                grad_norm = clip_gradients(
                    self.model.parameters(), float(self.cfg.get("grad_clip", self.cfg.get("max_grad_norm", 1.0)))
                )
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
                self.optimizer.step()
                elapsed = time.time() - start_time
                tokens = step * self.batch_size * self.seq_len * self.grad_accum_steps
                record = {
                    "step": step,
                    "loss": float(np.mean(losses)),
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens": tokens,
                    "tokens_per_sec": tokens / max(elapsed, 1e-6),
                    "elapsed_s": elapsed,
                    "precision": self.precision if self.amp_enabled else "fp32",
                    "grad_accum_steps": self.grad_accum_steps,
                    "attention_backend": getattr(self.model, "attention_backend", "unknown"),
                }
                record.update(gpu_snapshot(self.device))
                if self.health_monitor is not None:
                    record.update(self.health_monitor.observe(record))
                if step % int(self.cfg.get("eval_interval", 50)) == 0 or step == start_step + 1:
                    record["val_loss"] = _evaluate(
                        self.model,
                        self.val_data,
                        self.batch_size,
                        self.seq_len,
                        int(self.cfg.get("eval_batches", 2)),
                        self.device,
                        self.eval_rng,
                        self.amp_enabled,
                    )
                metric_logger.log(record)
                records.append(record)
                if step % int(self.cfg.get("log_interval", 10)) == 0 or step == start_step + 1:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if step % int(self.cfg.get("save_interval", 50)) == 0 or step == self.max_steps:
                    save_checkpoint(
                        self.ckpt_dir / f"{self.checkpoint_prefix}_step_{step:05d}.pt",
                        step=step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler_state=self.scheduler.state_dict(),
                        config=self.cfg,
                        data_meta=self.data_meta,
                        last_record=record,
                        extra_state={"batch_rng": self.rng.get_state(), "eval_rng": self.eval_rng.get_state()},
                    )
        wandb_info = dict(metric_logger.run_info)
        last = records[-1] if records else {"step": start_step}
        # Keep aggregate counters in the summary as well as in JSONL.  This
        # makes a report reproducible without having to infer totals from the
        # last log line and remains correct when a run is resumed.
        tps_values = [float(row["tokens_per_sec"]) for row in records if row.get("tokens_per_sec") is not None]
        peak_memory_values = [float(row["peak_memory_mb"]) for row in records if row.get("peak_memory_mb") is not None]
        summary = {
            "device": str(self.device),
            "cuda_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "param_count": sum(p.numel() for p in self.model.parameters()),
            "start_step": start_step,
            "end_step": int(last["step"]),
            "max_steps": self.max_steps,
            "final_loss": last.get("loss"),
            "final_val_loss": last.get("val_loss"),
            "elapsed_s": time.time() - start_time,
            "processed_tokens": last.get("tokens"),
            "mean_tokens_per_sec": float(np.mean(tps_values)) if tps_values else None,
            "peak_memory_mb": max(peak_memory_values, default=None),
            "precision": last.get("precision"),
            "grad_accum_steps": self.grad_accum_steps,
            "attention_backend": last.get("attention_backend"),
            "data": self.data_meta,
            "log_path": str(self.log_path),
        }
        if self.health_monitor is not None:
            summary["training_health"] = self.health_monitor.summary()
        if wandb_info:
            summary["wandb"] = wandb_info
        with (self.root / self.summary_name).open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
