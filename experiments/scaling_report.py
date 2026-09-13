"""Build a compact, source-grounded scaling report from run summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(run_specs: list[str]) -> list[dict]:
    results = []
    for spec in run_specs:
        name, raw_path = spec.split("=", 1)
        summary_path = Path(raw_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        log_path = summary_path.parent / summary.get("log_path", "")
        if not log_path.exists():
            log_path = summary_path.parent / "logs" / f"{name}_metrics.jsonl"
        records = _read_jsonl(log_path)
        tokens = int(records[-1].get("tokens", 0)) if records else 0
        results.append(
            {
                "name": name,
                "summary_path": str(summary_path),
                "parameters": int(summary.get("param_count", summary.get("parameters", 0))),
                "steps": int(summary.get("end_step", summary.get("steps", len(records)))),
                "tokens": tokens,
                "final_loss": summary.get("final_loss", summary.get("last_loss")),
                "final_val_loss": summary.get("final_val_loss"),
                "elapsed_s": float(summary.get("elapsed_s", 0.0)),
                "precision": summary.get("precision"),
                "world_size": summary.get("world_size", 1),
                "records": len(records),
            }
        )
    return results


def write_pdf(results: list[dict], output: Path, chart: Path | None = None) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch)
    story = [Paragraph("PretrainLab-X Day 4 Scaling Analysis", styles["Title"]), Spacer(1, 0.12 * inch)]
    story.append(
        Paragraph(
            "This report summarizes the recorded experiments. It does not infer model quality beyond the observed loss, token count, and runtime.",
            styles["BodyText"],
        )
    )
    table_data = [["Run", "Parameters", "Steps", "Tokens", "Final loss", "Runtime (s)"]]
    for row in results:
        table_data.append(
            [
                row["name"],
                f'{row["parameters"]:,}',
                str(row["steps"]),
                f'{row["tokens"]:,}',
                "-" if row["final_loss"] is None else f'{float(row["final_loss"]):.5f}',
                f'{row["elapsed_s"]:.1f}',
            ]
        )
    table = Table(table_data, repeatRows=1, colWidths=[1.45 * inch, 0.9 * inch, 0.55 * inch, 1.0 * inch, 0.8 * inch, 0.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([Spacer(1, 0.18 * inch), table])
    if chart and chart.exists():
        story.extend([Spacer(1, 0.18 * inch), Paragraph("Observed training loss", styles["Heading2"]), Image(str(chart), width=6.9 * inch, height=3.1 * inch)])
    story.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Interpretation", styles["Heading2"]),
            Paragraph(
                "A larger parameter count is an engineering scaling axis, not proof of better capability. Compare runs only when data, token budget, optimizer settings, and evaluation protocol are explicitly recorded.",
                styles["BodyText"],
            ),
        ]
    )
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=summary.json; repeat for multiple runs")
    parser.add_argument("--output", default="report/scaling_analysis.pdf")
    parser.add_argument("--json", default="report/scaling_metrics.json")
    args = parser.parse_args()
    results = collect(args.run)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_path = None
    try:
        import matplotlib.pyplot as plt

        for row in results:
            summary_path = Path(row["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            log_path = summary_path.parent / summary.get("log_path", "")
            if not log_path.exists():
                log_path = summary_path.parent / "logs" / f'{row["name"]}_metrics.jsonl'
            records = _read_jsonl(log_path)
            if records:
                plt.plot([r["step"] for r in records], [r["loss"] for r in records], label=row["name"])
        if results:
            plt.xlabel("optimizer step")
            plt.ylabel("training loss")
            plt.grid(alpha=0.25)
            plt.legend()
            chart_path = Path(args.output).with_suffix(".png")
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=160)
            plt.close()
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"chart skipped: {exc}")
    write_pdf(results, Path(args.output), chart_path)
    print(json.dumps({"runs": results, "pdf": args.output, "json": args.json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
