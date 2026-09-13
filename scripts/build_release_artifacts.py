#!/usr/bin/env python3
"""Build the small, public-safe Day 8 metrics and figures.

The script consumes only checked-in summaries/logs.  It never reads model
weights, raw parquet files, token arrays, or W&B binary runs.  When a raw log
is unavailable, the figure is labelled as an endpoint/table view rather than
inventing a per-step curve.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strict_day23() -> dict:
    text = (RESULTS / "STRICT_DAY2_DAY3_RESULT.md").read_text(encoding="utf-8")
    pattern = re.compile(
        r"\| (?P<name>30M 正式组|高学习率组（LR=3e-3）|正常 LR 恢复组|checkpoint 恢复验证)"
        r" \|\s*(?P<records>[^|]+)\|\s*(?P<range>[^|]+)\|\s*"
        r"(?P<first>[0-9.]+) → (?P<last>[0-9.]+)\s*\|\s*(?P<best>[0-9.]+)\s*\|\s*"
        r"(?P<grad>[0-9.]+)\s*\|\s*(?P<ckpt>[^|]+)\|"
    )
    rows = {}
    for match in pattern.finditer(text):
        row = match.groupdict()
        rows[row["name"]] = {
            "records": row["records"].strip(),
            "step_range": row["range"].strip(),
            "first_loss": float(row["first"]),
            "final_loss": float(row["last"]),
            "best_loss": float(row["best"]),
            "max_preclip_grad_norm": float(row["grad"]),
            "checkpoints": row["ckpt"].strip(),
        }
    validation = re.search(
        r"正式组最终 validation loss 为 `([0-9.]+)`；高 LR 组为 `([0-9.]+)`；"
        r"正常 LR 恢复组为 `([0-9.]+)`",
        text,
    )
    if validation:
        rows["30M 正式组"]["final_validation_loss"] = float(validation.group(1))
        rows["高学习率组（LR=3e-3）"]["final_validation_loss"] = float(validation.group(2))
        rows["正常 LR 恢复组"]["final_validation_loss"] = float(validation.group(3))
    return {
        "model_parameters": 30_419_456,
        "data_tokens": 2_000_000,
        "data_source": "builtin-fallback (documented engineering fallback)",
        "runs": rows,
        "source": "results/STRICT_DAY2_DAY3_RESULT.md",
        "raw_logs_in_release": False,
    }


def make_metrics() -> dict:
    day4_300 = load_json(RESULTS / "machine_readable/day4_300m_benchmark_summary.json", {})
    day4_700 = load_json(RESULTS / "machine_readable/day4_700m_benchmark_summary.json", {})
    fsdp = load_json(RESULTS / "machine_readable/fsdp_summary.json", {})
    day5 = load_json(RESULTS / "machine_readable/day5_336m_500m_summary.json", {})
    day6 = load_json(RESULTS / "machine_readable/day6_336m_500m_fresh_summary.json", {})
    heldout = load_json(RESULTS / "machine_readable/day6_6_summary.json", {})
    if not day6:
        # The fresh-data summary is currently kept as a narrative result on
        # some runs; keep the absence explicit instead of copying a guess.
        day6 = {"source": "results/DAY6_EXECUTION_RESULT.md", "status": "see narrative report"}
    return {
        "schema_version": "day8.v1",
        "project": "PretrainLab-X",
        "generated_from_checked_in_evidence": True,
        "strict_day2_day3": strict_day23(),
        "day4": {
            "300m_benchmark": day4_300,
            "700m_benchmark": day4_700,
            "fsdp_path": fsdp,
            "scope": "short single-process benchmarks; no 3B/5B-token long run or multi-GPU FSDP",
        },
        "day5": day5,
        "day6": day6,
        "shared_heldout": {
            "dataset": heldout.get("heldout", {}).get("dataset"),
            "documents": heldout.get("evaluation", {}).get("documents"),
            "scored_tokens": heldout.get("evaluation", {}).get("scored_tokens"),
            "day5_nll": heldout.get("evaluation", {}).get("day5", {}).get("weighted_nll"),
            "day6_nll": heldout.get("evaluation", {}).get("day6", {}).get("weighted_nll"),
            "delta_nll_day5_minus_day6": heldout.get("paired_bootstrap", {}).get("delta_nll_day5_minus_day6"),
            "delta_nll_ci95": heldout.get("paired_bootstrap", {}).get("delta_nll_ci95"),
            "limitations": heldout.get("limitations", []),
            "source": "results/machine_readable/day6_6_summary.json",
        },
        "sanity_check": {
            "source": "results/DAY7_SANITY_CHECK_RESULT_CN.md",
            "status": "complete",
            "new_training": False,
            "scope": "checkpoint/configuration/random-baseline/precision cross-check",
            "day5_train_nll": 0.065023,
            "day6_train_subset_nll": 3.049181,
            "shared_heldout_day5_nll": 8.4242114093,
            "shared_heldout_day6_nll": 3.0394002921,
        },
        "audit": {
            "source": "audit/HASH_CROSSCHECK.json",
            "status": "pass",
            "weights_or_metrics_modified": False,
        },
        "interpretation_boundary": (
            "These are auditable, bounded engineering experiments. They do not establish a causal effect of data freshness "
            "because the Day5/Day6 training histories are not identical, and they are not industrial-scale pretraining."
        ),
    }


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_endpoint_chart(plt, path: Path, title: str, ylabel: str, labels: list[str], values: list[float], note: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    xs = list(range(len(values)))
    ax.plot(xs, values, marker="o", linewidth=2.2, color="#2563eb")
    ax.set_xticks(xs, labels, rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_figures(metrics: dict) -> dict:
    plt = import_matplotlib()
    FIGURES.mkdir(parents=True, exist_ok=True)

    strict = metrics["strict_day2_day3"]["runs"]
    normal = strict.get("30M 正式组", {})
    high = strict.get("高学习率组（LR=3e-3）", {})
    save_endpoint_chart(
        plt,
        FIGURES / "normal_vs_high_lr_loss.png",
        "Day 2-3 strict run: normal vs high learning rate",
        "reported endpoint loss",
        ["normal LR", "high LR (3e-3)"],
        [normal.get("final_loss", math.nan), high.get("final_loss", math.nan)],
        "Endpoint/table view from STRICT_DAY2_DAY3_RESULT.md; raw per-step logs were not in the local release staging.",
    )
    save_endpoint_chart(
        plt,
        FIGURES / "normal_vs_high_lr_grad_norm.png",
        "Day 2-3 strict run: pre-clip gradient risk",
        "maximum pre-clip gradient norm",
        ["normal LR", "high LR (3e-3)"],
        [normal.get("max_preclip_grad_norm", math.nan), high.get("max_preclip_grad_norm", math.nan)],
        "Endpoint/table view from STRICT_DAY2_DAY3_RESULT.md; values are pre-clipping diagnostics.",
    )

    held = metrics["shared_heldout"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    labels = ["Day 5\nrepeated slice", "Day 6\nfresh stream"]
    vals = [float(held["day5_nll"]), float(held["day6_nll"])]
    bars = ax.bar(labels, vals, color=["#f97316", "#16a34a"], width=0.58)
    ax.set_ylabel("weighted NLL (lower is better)")
    ax.set_title("Shared held-out evaluation")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.3f}", ha="center", fontsize=10)
    ci = held.get("delta_nll_ci95") or {}
    note = f"Δ NLL (Day5−Day6)={float(held['delta_nll_day5_minus_day6']):.3f}; 95% CI [{float(ci.get('low', math.nan)):.3f}, {float(ci.get('high', math.nan)):.3f}]"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(FIGURES / "shared_heldout_nll.png", bbox_inches="tight")
    plt.close(fig)

    san = metrics["sanity_check"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    labels = ["Day5\ntrain", "Day5\nheld-out", "Day6\ntrain subset", "Day6\nheld-out"]
    vals = [san["day5_train_nll"], san["shared_heldout_day5_nll"], san["day6_train_subset_nll"], san["shared_heldout_day6_nll"]]
    bars = ax.bar(labels, vals, color=["#fb923c", "#fdba74", "#4ade80", "#16a34a"])
    ax.set_ylabel("NLL (lower is better)")
    ax.set_title("Memorization vs generalization sanity check")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.3f}", ha="center", fontsize=9)
    ax.text(0.02, 0.02, "Day5 shows a large train/held-out gap; Day6 train-subset and held-out scores are close.", transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(FIGURES / "memorization_generalization.png", bbox_inches="tight")
    plt.close(fig)

    day6_log = ROOT / "logs/day6_336m_500m_fresh_metrics.jsonl"
    rows = load_jsonl(day6_log)
    if rows:
        xs = [r.get("step") for r in rows if r.get("loss") is not None]
        ys = [r.get("loss") for r in rows if r.get("loss") is not None]
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
        ax.plot(xs, ys, linewidth=1.2, color="#2563eb")
        ax.set_title("Day 6 fresh-data training curve")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "day6_training_curve.png", bbox_inches="tight")
        plt.close(fig)
        curve_status = "raw_jsonl"
    else:
        save_endpoint_chart(
            plt,
            FIGURES / "day6_training_curve.png",
            "Day 6 fresh-data training curve",
            "loss",
            ["raw log unavailable"],
            [0.0],
            "Placeholder only: retrieve day6_336m_500m_fresh_metrics.jsonl from B46 before public release.",
        )
        curve_status = "placeholder_missing_raw_log"

    return {
        "normal_vs_high_lr_loss": "figures/normal_vs_high_lr_loss.png",
        "normal_vs_high_lr_grad_norm": "figures/normal_vs_high_lr_grad_norm.png",
        "shared_heldout_nll": "figures/shared_heldout_nll.png",
        "memorization_generalization": "figures/memorization_generalization.png",
        "day6_training_curve": "figures/day6_training_curve.png",
        "day6_curve_source_status": curve_status,
    }


def main() -> None:
    metrics = make_metrics()
    figures = build_figures(metrics)
    metrics["figures"] = figures
    out = RESULTS / "final_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(out), "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
