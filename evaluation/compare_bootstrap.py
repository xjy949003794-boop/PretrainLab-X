"""Paired document bootstrap comparison for Day5 and Day6 evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def weighted(rows: list[dict]) -> tuple[float, int, float]:
    total = float(sum(float(r["nll_sum"]) for r in rows))
    tokens = int(sum(int(r["tokens"]) for r in rows))
    return total / max(tokens, 1), tokens, total


def paired_bootstrap(day5_path: Path, day6_path: Path, *, draws: int, seed: int) -> dict:
    a = {int(r["doc_id"]): r for r in read_jsonl(day5_path)}
    b = {int(r["doc_id"]): r for r in read_jsonl(day6_path)}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise RuntimeError("no paired document ids")
    tokens = np.asarray([min(int(a[i]["tokens"]), int(b[i]["tokens"])) for i in ids], dtype=np.float64)
    d5 = np.asarray([float(a[i]["nll_sum"]) for i in ids], dtype=np.float64)
    d6 = np.asarray([float(b[i]["nll_sum"]) for i in ids], dtype=np.float64)
    # Both evaluations use the same token stream and score mask. Keep the
    # shared count explicit in case a future run has a partial final document.
    d5 = d5 * (tokens / np.maximum(np.asarray([int(a[i]["tokens"]) for i in ids]), 1))
    d6 = d6 * (tokens / np.maximum(np.asarray([int(b[i]["tokens"]) for i in ids]), 1))
    exact_delta = (float(d5.sum()) - float(d6.sum())) / max(float(tokens.sum()), 1.0)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, len(ids), size=len(ids))
        samples.append(float((d5[idx].sum() - d6[idx].sum()) / max(tokens[idx].sum(), 1.0)))
    q = np.quantile(np.asarray(samples), [0.025, 0.5, 0.975]).tolist()
    return {"kind": "day6.6_paired_document_bootstrap", "draws": draws, "seed": seed, "paired_documents": len(ids), "paired_tokens": int(tokens.sum()), "day5_nll": float(d5.sum() / max(tokens.sum(), 1.0)), "day6_nll": float(d6.sum() / max(tokens.sum(), 1.0)), "delta_nll_day5_minus_day6": exact_delta, "delta_perplexity": float(math.exp(float(d5.sum() / max(tokens.sum(), 1.0))) - math.exp(float(d6.sum() / max(tokens.sum(), 1.0)))), "delta_nll_ci95": {"low": q[0], "median": q[1], "high": q[2]}, "interpretation": "positive delta means Day6 fresh-data checkpoint has lower held-out NLL"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day5-doc-metrics", required=True, type=Path)
    p.add_argument("--day6-doc-metrics", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    result = paired_bootstrap(a.day5_doc_metrics, a.day6_doc_metrics, draws=a.draws, seed=a.seed)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with a.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in result.items():
            if isinstance(v, dict):
                for sub, value in v.items():
                    w.writerow([f"{k}.{sub}", value])
            else:
                w.writerow([k, v])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

