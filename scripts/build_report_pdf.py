#!/usr/bin/env python3
"""Create a compact PDF companion to the final Markdown technical report."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdf/PretrainLab-X_Day8_Technical_Report.pdf"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(42, 24, "PretrainLab-X | Day 8 evidence package")
    canvas.drawRightString(A4[0] - 42, 24, f"page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"), spaceAfter=18))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#475569")))
    styles["BodyText"].fontSize = 9.3
    styles["BodyText"].leading = 12
    story = []
    story.append(Paragraph("PretrainLab-X", styles["TitleCenter"]))
    story.append(Paragraph("Day 8 Final Technical Report", styles["Heading1"]))
    story.append(Paragraph("Auditable, engineering-scale mini foundation-model pretraining stack", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Date: 2026-09-14. This report summarizes checked-in evidence. It does not claim industrial-scale pretraining or a causal effect of data freshness alone.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Key results", styles["Heading2"]))
    rows = [
        ["Evidence", "Result", "Boundary"],
        ["Strict Day 2-3", "30.42M params; 500 steps", "byte-level fallback corpus"],
        ["Day 4 benchmarks", "336M/653M short runs", "no 3B/5B-token long run"],
        ["Day 5", "336M; 500,006,912 positions; loss 0.069339", "9.9M slice replayed"],
        ["Day 6", "336M; 500,006,912 unique tokens", "single-GPU bounded run"],
        ["Shared held-out", "NLL 8.424211 vs 3.039400; Δ 5.384811", "not a causal proof"],
        ["Day 7", "load/config/random/precision checks pass", "no weight edits"],
    ]
    table = Table(rows, colWidths=[1.25 * inch, 2.55 * inch, 2.45 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))
    sections = [
        ("1. System", "The model is a pre-normalized decoder-only Transformer using RMSNorm, RoPE, SwiGLU and GQA (16 query heads / 4 key-value heads). Causal attention uses PyTorch SDPA with the recorded sdpa_auto backend. The trainer implements BF16 autocast, accumulation, clipping, warmup-cosine scheduling, atomic checkpoints and complete optimizer/scheduler/RNG restoration."),
        ("2. Data", "The real-data path uses FineWeb-Edu sample-10BT and the TinyLlama tokenizer. Day 4 retains a 10M-token slice; Day 6 uses a separate shard and document-boundary split. Raw Parquet, token arrays, checkpoints and W&B binaries remain private."),
        ("3. Interpretation", "Day 5 replays a 9.9M-token slice while Day 6 consumes a fresh stream. Their shared held-out difference is stable under paired bootstrap and sanity checks, but the histories are not identical, so the result is not a freshness-only causal estimate."),
        ("4. Audit", "results/final_metrics.json is generated from checked-in summaries. audit/ contains the canonical SHA-256 manifest and cross-check. scripts/verify_release.py rejects forbidden large or credential-like artifacts."),
        ("5. Reproduction", "Start with the CPU smoke test in REPRODUCIBILITY.md, then use the explicitly named short GPU benchmarks. Long-run reproduction requires private inputs and storage; no public command downloads third-party raw data automatically."),
    ]
    for title, body in sections:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 5))
    story.append(PageBreak())
    story.append(Paragraph("Figures", styles["Heading1"]))
    for name, caption in [
        ("normal_vs_high_lr_loss.png", "Strict Day 2-3 endpoint loss comparison"),
        ("normal_vs_high_lr_grad_norm.png", "Strict Day 2-3 pre-clip gradient risk"),
        ("shared_heldout_nll.png", "Shared held-out NLL"),
        ("memorization_generalization.png", "Memorization versus generalization"),
        ("day6_training_curve.png", "Day 6 fresh-data training curve"),
    ]:
        path = ROOT / "figures" / name
        if not path.exists():
            continue
        block = [Paragraph(caption, styles["Heading3"])]
        img = Image(str(path))
        img._restrictSize(6.5 * inch, 3.4 * inch)
        block.extend([img, Spacer(1, 10)])
        story.append(KeepTogether(block))
    story.append(Spacer(1, 8))
    story.append(Paragraph("See README.md, README_CN.md and LIMITATIONS.md for the public release boundary. Values and source paths are machine-readable in results/final_metrics.json.", styles["Small"]))
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42, title="PretrainLab-X Day 8 Technical Report", author="PretrainLab-X")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
