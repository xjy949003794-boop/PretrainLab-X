"""Create a tiny deterministic loss-spike case and show the diagnosis path."""

from __future__ import annotations

import json
from pathlib import Path

from observatory import diagnose


def main() -> None:
    records = []
    for step in range(1, 9):
        records.append(
            {
                "step": step,
                "loss": 2.0 if step < 5 else (9.5 if step == 5 else 8.0),
                "lr": 3e-3,
                "grad_norm": 18.0 if step >= 5 else 1.2,
                "tokens_per_sec": 1000.0,
                "precision": "bf16",
            }
        )
    path = Path("logs/anomaly_demo.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    report = diagnose(records)
    Path("anomaly_demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
