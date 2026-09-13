"""Adaptive training-health checks for the Day 5 experiments.

The monitor is deliberately conservative: it never changes optimizer state or
silently stops a run.  It annotates each JSONL record and produces a separate
report so a failure can be inspected after the fact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class AdaptiveTrainingMonitor:
    """Track loss, gradients, throughput and GPU utilization online."""

    def __init__(
        self,
        *,
        warmup_records: int = 5,
        loss_spike_ratio: float = 3.0,
        max_grad_norm: float = 10.0,
        min_gpu_utilization_pct: float = 15.0,
        throughput_drop_ratio: float = 0.5,
    ) -> None:
        self.warmup_records = max(1, int(warmup_records))
        self.loss_spike_ratio = float(loss_spike_ratio)
        self.max_grad_norm = float(max_grad_norm)
        self.min_gpu_utilization_pct = float(min_gpu_utilization_pct)
        self.throughput_drop_ratio = float(throughput_drop_ratio)
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.throughputs: list[float] = []
        self.utilizations: list[float] = []
        self.events: list[dict[str, Any]] = []

    def observe(self, record: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        loss = _finite_float(record.get("loss"))
        grad = _finite_float(record.get("grad_norm")) if "grad_norm" in record else None
        tps = _finite_float(record.get("tokens_per_sec")) if "tokens_per_sec" in record else None
        util = _finite_float(record.get("gpu_utilization_pct")) if "gpu_utilization_pct" in record else None
        if loss is None:
            reasons.append("nonfinite_loss")
        if "grad_norm" in record and grad is None:
            reasons.append("nonfinite_grad_norm")
        elif grad > self.max_grad_norm:
            reasons.append("gradient_spike")
        if self.losses and loss is not None:
            baseline = sum(self.losses[: self.warmup_records]) / min(len(self.losses), self.warmup_records)
            if len(self.losses) >= 2 and loss > max(1e-8, baseline) * self.loss_spike_ratio:
                reasons.append("loss_spike")
        if self.throughputs and tps is not None:
            baseline_tps = sum(self.throughputs[: self.warmup_records]) / min(
                len(self.throughputs), self.warmup_records
            )
            if len(self.throughputs) >= self.warmup_records and tps < baseline_tps * self.throughput_drop_ratio:
                reasons.append("throughput_drop")
        # A low utilization reading is a warning, not a failure: nvidia-smi
        # can legitimately sample between kernels or be unavailable.
        if util is not None and util < self.min_gpu_utilization_pct:
            reasons.append("low_gpu_utilization")

        status = "critical" if any(r in reasons for r in ("nonfinite_loss", "nonfinite_grad_norm")) else (
            "warning" if reasons else "ok"
        )
        event = {
            "step": int(record.get("step", len(self.events) + 1)),
            "status": status,
            "reasons": reasons,
        }
        self.events.append(event)
        if loss is not None:
            self.losses.append(loss)
        if grad is not None:
            self.grad_norms.append(grad)
        if tps is not None:
            self.throughputs.append(tps)
        if util is not None:
            self.utilizations.append(util)
        return {"health_status": status, "health_reasons": reasons}

    def summary(self) -> dict[str, Any]:
        warning_events = [event for event in self.events if event["status"] != "ok"]
        return {
            "steps": len(self.events),
            "status": "critical" if any(e["status"] == "critical" for e in self.events) else (
                "warning" if warning_events else "ok"
            ),
            "warning_steps": [event["step"] for event in warning_events],
            "reasons": sorted({reason for event in warning_events for reason in event["reasons"]}),
            "max_grad_norm": max(self.grad_norms, default=None),
            "mean_tokens_per_sec": sum(self.throughputs) / len(self.throughputs) if self.throughputs else None,
            "mean_gpu_utilization_pct": sum(self.utilizations) / len(self.utilizations)
            if self.utilizations
            else None,
        }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def diagnose(path: str | Path) -> dict[str, Any]:
    monitor = AdaptiveTrainingMonitor()
    for record in read_records(path):
        monitor.observe(record)
    return monitor.summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="training_health_report.json")
    args = parser.parse_args()
    report = diagnose(args.log)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
