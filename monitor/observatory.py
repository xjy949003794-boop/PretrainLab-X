"""Turn JSONL training events into an auditable run report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_records(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def diagnose(records: list[dict]) -> dict:
    if not records:
        return {"steps": 0, "passed": False, "anomalies": ["empty_log"]}
    losses = [float(r["loss"]) for r in records if "loss" in r]
    grads = [float(r["grad_norm"]) for r in records if "grad_norm" in r]
    lrs = [float(r["lr"]) for r in records if "lr" in r]
    anomalies: list[str] = []
    if any(not math.isfinite(v) for v in losses):
        anomalies.append("nonfinite_loss")
    if any(not math.isfinite(v) for v in grads):
        anomalies.append("nonfinite_grad_norm")
    if len(losses) >= 3:
        baseline = max(1e-8, sum(losses[: min(5, len(losses))]) / min(5, len(losses)))
        if max(losses) > baseline * 3.0:
            anomalies.append("loss_spike")
    if grads and max(grads) > 10.0:
        anomalies.append("gradient_explosion_risk")
    if lrs and max(lrs) > 1e-2:
        anomalies.append("learning_rate_too_high")
    throughput = [float(r["tokens_per_sec"]) for r in records if "tokens_per_sec" in r]
    utilization = [float(r["gpu_utilization_pct"]) for r in records if r.get("gpu_utilization_pct") is not None]
    report = {
        "steps": int(records[-1].get("step", len(records))),
        "records": len(records),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "best_val_loss": min((float(r["val_loss"]) for r in records if "val_loss" in r), default=None),
        "max_grad_norm": max(grads, default=None),
        "mean_tokens_per_sec": sum(throughput) / len(throughput) if throughput else None,
        "mean_gpu_utilization_pct": sum(utilization) / len(utilization) if utilization else None,
        "min_gpu_utilization_pct": min(utilization, default=None),
        "peak_memory_mb": max((float(r["peak_memory_mb"]) for r in records if "peak_memory_mb" in r), default=None),
        "precision": records[-1].get("precision"),
        "attention_backend": records[-1].get("attention_backend"),
        "anomalies": anomalies,
        "passed": not anomalies and len(records) > 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/day2_day3_metrics.jsonl")
    parser.add_argument("--out", default="day2_day3_observatory.json")
    args = parser.parse_args()
    report = diagnose(read_records(args.log))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
