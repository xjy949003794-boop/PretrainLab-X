"""Generate the Day 5 resume-facing project report from recorded artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: str | Path | None, default: Any) -> Any:
    if not path:
        return default
    candidate = Path(path)
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else default


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _loss_chart(log_paths: list[Path], output: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plotted = False
    for path in log_paths:
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records and all("loss" in row for row in records):
            plt.plot([row["step"] for row in records], [row["loss"] for row in records], label=path.stem)
            plotted = True
    if not plotted:
        plt.close()
        return None
    plt.xlabel("optimizer step")
    plt.ylabel("training loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def build_report(
    *,
    output: Path,
    run_summaries: list[tuple[str, Path]],
    throughput: dict,
    memory_off: dict,
    memory_on: dict,
    health: dict,
    wandb_runs: list[dict],
    log_paths: list[Path],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, KeepTogether, Paragraph, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

    output.parent.mkdir(parents=True, exist_ok=True)
    chart = _loss_chart(log_paths, output.with_suffix(".png"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=7.5, leading=9, textColor=colors.HexColor("#475569")))
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = [
        Paragraph("PretrainLab-X: Day 5 Final Polish", styles["Title"]),
        Paragraph(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}", styles["Caption"]),
        Spacer(1, 0.12 * inch),
        Paragraph(
            "A production-shaped LLaMA-style foundation-model pretraining stack with real FineWeb-Edu data, BF16 training, activation-checkpointing comparison, throughput measurement, adaptive health monitoring, and reproducible experiment artifacts.",
            styles["BodyText"],
        ),
        Spacer(1, 0.14 * inch),
        Paragraph("Scope boundary", styles["Heading2"]),
        Paragraph(
            "These are controlled validation runs. The Day 5 suite includes a 336M-parameter run targeted at approximately 500M tokens, plus bounded memory and throughput experiments. This remains a small-scale pretraining validation, not a claim of completing the 3B/5B-token Day 4 targets.",
            styles["BodyText"],
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("1. Architecture and training system", styles["Heading2"]),
        Paragraph(
            "The model exposes RMSNorm, RoPE, SwiGLU, grouped-query attention, PyTorch SDPA, tied/untied embeddings, and activation checkpointing. The engine provides BF16 autocast, gradient accumulation, clipping, warmup-cosine scheduling, atomic checkpoint/resume, JSONL metrics, optional W&B logging, and GPU snapshots.",
            styles["Small"],
        ),
        Spacer(1, 0.12 * inch),
    ]
    arch_rows = [["Component", "Evidence"]]
    for row in [
        ("Data", "FineWeb-Edu sample-10BT slice; 10M tokens; deterministic filters and exact-text dedup"),
        ("Precision", "BF16 autocast on CUDA"),
        ("Memory", "Activation checkpointing switch and paired on/off runs"),
        ("Observability", "Loss, LR, gradient norm, throughput, GPU memory/utilization, health status"),
        ("Recovery", "Atomic checkpoint with optimizer, scheduler, RNG and data provenance"),
    ]:
        arch_rows.append([row[0], row[1]])
    arch = Table(arch_rows, colWidths=[1.65 * inch, 5.15 * inch])
    arch.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAB7C4")), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])]))
    story.append(arch)
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("2. Recorded training runs", styles["Heading2"]))
    run_table = [["Run", "Parameters", "Steps", "Tokens", "Final loss", "Val loss", "Runtime (s)"]]
    for name, path in run_summaries:
        summary = _load(path, {})
        run_table.append([
            name,
            f'{int(summary.get("param_count", summary.get("parameters", 0))):,}',
            _fmt(summary.get("end_step", summary.get("steps"))),
            _fmt(summary.get("tokens", summary.get("processed_tokens"))),
            _fmt(summary.get("final_loss", summary.get("last_loss")), 4),
            _fmt(summary.get("final_val_loss"), 4),
            _fmt(summary.get("elapsed_s"), 1),
        ])
    run_table = Table(run_table, repeatRows=1, colWidths=[1.2 * inch, 1.05 * inch, 0.5 * inch, 0.8 * inch, 0.78 * inch, 0.72 * inch, 0.75 * inch])
    run_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAB7C4")), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])]))
    story.append(run_table)
    if chart:
        story.extend([
            Spacer(1, 0.08 * inch),
            KeepTogether([
                Image(str(chart), width=6.9 * inch, height=2.15 * inch),
                Paragraph("Observed loss curves; short runs are not capability evaluations.", styles["Caption"]),
            ]),
        ])

    story.extend([PageBreak(), Paragraph("3. Efficiency and memory trade-off", styles["Heading2"])])
    throughput_rows = [["Batch", "Seq", "Status", "Tokens/s", "Peak memory (MB)"]]
    for row in throughput.get("rows", []):
        throughput_rows.append([_fmt(row.get("batch_size")), _fmt(row.get("seq_len")), row.get("status", "-"), _fmt(row.get("tokens_per_sec"), 1), _fmt(row.get("peak_memory_mb"), 1)])
    if len(throughput_rows) == 1:
        throughput_rows.append(["-", "-", "not recorded", "-", "-"])
    table = Table(throughput_rows, repeatRows=1, colWidths=[0.75 * inch, 0.75 * inch, 1.0 * inch, 1.05 * inch, 1.3 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAB7C4")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])]))
    story.append(Paragraph("Throughput benchmark", styles["Heading3"]))
    story.append(table)
    story.append(Spacer(1, 0.14 * inch))
    memory_rows = [["Mode", "Steps", "Peak memory (MB)", "Mean tokens/s", "Status"]]
    for name, report in (("checkpoint off", memory_off), ("checkpoint on", memory_on)):
        rows = report.get("rows", [])
        ok = [row for row in rows if row.get("status") == "ok"]
        peak = max((row.get("peak_memory_mb") for row in ok if row.get("peak_memory_mb") is not None), default=None)
        tps = sum(row.get("tokens_per_sec", 0.0) for row in ok) / len(ok) if ok else None
        memory_rows.append([name, _fmt(report.get("measure_steps", report.get("steps", 3))), _fmt(peak, 1), _fmt(tps, 1), "ok" if ok else "not recorded"])
    mem_table = Table(memory_rows, repeatRows=1, colWidths=[1.25 * inch, 0.8 * inch, 1.35 * inch, 1.25 * inch, 1.0 * inch])
    mem_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAB7C4")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])]))
    story.append(Paragraph("Activation-checkpointing comparison", styles["Heading3"]))
    story.append(mem_table)
    story.append(Paragraph("The intended engineering trade-off is lower activation memory in exchange for extra recomputation; the table reports only what the recorded run measured.", styles["Caption"]))

    story.extend([Spacer(1, 0.18 * inch), Paragraph("4. Adaptive training health", styles["Heading2"])])
    story.append(Paragraph(json.dumps(health, ensure_ascii=False, indent=2), styles["Code"]))
    story.extend([Spacer(1, 0.14 * inch), Paragraph("5. W&B reproducibility", styles["Heading2"])])
    story.append(Paragraph("Training logs are mirrored to W&B offline run directories when the SDK is available. The same metrics remain in JSONL so the experiment is auditable without a network connection.", styles["Small"]))
    if wandb_runs:
        rows = [["Run", "Mode", "Local directory"]]
        for row in wandb_runs:
            rows.append([str(row.get("name") or row.get("id") or "-"), str(row.get("mode", "offline")), str(row.get("dir", "-"))])
        wb_table = Table(rows, repeatRows=1, colWidths=[1.8 * inch, 0.8 * inch, 4.2 * inch])
        wb_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAB7C4")), ("FONTSIZE", (0, 0), (-1, -1), 7.5)]))
        story.append(wb_table)
    story.extend([Spacer(1, 0.14 * inch), Paragraph("6. Limitations", styles["Heading2"]), Paragraph("The dataset slice is 10M tokens, the training runs are bounded, the FSDP validation is single-process unless a multi-GPU result is explicitly supplied, and an online W&B URL is not inferred from offline files. These boundaries should be stated in interviews.", styles["BodyText"])])
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report/PretrainLab-X_Final_Report.pdf")
    parser.add_argument("--throughput", default="report/throughput_benchmark.json")
    parser.add_argument("--memory-off", default="report/day5_653m_memory_off_benchmark.json")
    parser.add_argument("--memory-on", default="report/day5_653m_memory_on_benchmark.json")
    parser.add_argument("--health", default="report/training_health_report.json")
    parser.add_argument("--run", action="append", default=[], help="name=summary.json")
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--wandb-json", default="report/wandb_runs.json")
    args = parser.parse_args()
    run_summaries = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in args.run]
    build_report(
        output=Path(args.out),
        run_summaries=run_summaries,
        throughput=_load(args.throughput, {}),
        memory_off=_load(args.memory_off, {}),
        memory_on=_load(args.memory_on, {}),
        health=_load(args.health, {"status": "not recorded"}),
        wandb_runs=_load(args.wandb_json, []),
        log_paths=[Path(path) for path in args.log],
    )
    print(json.dumps({"pdf": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
