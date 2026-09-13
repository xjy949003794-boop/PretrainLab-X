"""Produce a small machine-checkable Day-1 run audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=".")
    args = p.parse_args()
    root = Path(args.run_dir)
    checks = {
        "tokenized_data": (root / "data/tokenized/train.npy").exists(),
        "data_metadata": (root / "data/tokenized/metadata.json").exists(),
        "train_log": (root / "logs/train.jsonl").exists(),
        "checkpoint": any((root / "checkpoints").glob("step_*.pt")),
        "run_summary": (root / "run_summary.json").exists(),
    }
    records = []
    if checks["train_log"]:
        with (root / "logs/train.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    losses = [float(x["loss"]) for x in records if "loss" in x]
    report = {"checks": checks, "steps_logged": len(records), "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None, "loss_decreased": bool(losses and losses[-1] < losses[0])}
    report["passed"] = all(checks.values()) and report["loss_decreased"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (root / "run_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

