#!/usr/bin/env python3
"""Fail-closed audit for a public-safe PretrainLab-X release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "README_CN.md",
    "LIMITATIONS.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "environment.txt",
    "results/final_metrics.json",
    "audit/CANONICAL_HASH_MANIFEST.json",
    "audit/HASH_CROSSCHECK.json",
    "figures/normal_vs_high_lr_loss.png",
    "figures/normal_vs_high_lr_grad_norm.png",
    "figures/shared_heldout_nll.png",
    "figures/memorization_generalization.png",
    "figures/day6_training_curve.png",
]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".parquet", ".npy", ".int32", ".wandb"}
MAX_FILE_BYTES = 100 * 1024 * 1024


def main() -> int:
    errors: list[str] = []
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required file: {rel}")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 100 MiB: {rel}")
        if path.name in {".env", ".npmrc"} or "secret" in path.name.lower():
            errors.append(f"secret-like filename: {rel}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Build markers without embedding the exact token in this auditor;
        # otherwise the auditor would flag its own source file.
        needles = ("ghp" + "_", "github_pat" + "_", "s" + "k-", "AK" + "IA")
        for needle in needles:
            if needle in text:
                errors.append(f"possible credential marker {needle!r}: {rel}")
    metrics = ROOT / "results/final_metrics.json"
    if metrics.exists():
        try:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            if payload.get("generated_from_checked_in_evidence") is not True:
                errors.append("final_metrics.json is not marked evidence-derived")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid final_metrics.json: {exc}")
    manifest = ROOT / "audit/CANONICAL_HASH_MANIFEST.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            entries = data.get("files", {})
            if isinstance(entries, dict):
                rows = entries.items()
            elif isinstance(entries, list):
                rows = ((str(i), row) for i, row in enumerate(entries))
            else:
                errors.append("canonical manifest files must be an object or list")
                rows = []
            for name, row in rows:
                digest = row.get("sha256", "") if isinstance(row, dict) else ""
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                    errors.append(f"canonical hash is not 64-char hex: {name}")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid canonical manifest: {exc}")
    if errors:
        print("RELEASE_AUDIT_FAIL")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"RELEASE_AUDIT_PASS files={sum(p.is_file() for p in ROOT.rglob('*'))} bytes={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
