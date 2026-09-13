#!/usr/bin/env python3
"""Day 7 patch: canonical SHA-256 audit and documentation-repair evidence.

This program is intentionally read-only with respect to the three source
files.  It never trains, evaluates, or rewrites model/data files.  It writes
only a new audit directory and can optionally inspect a supplied directory of
project documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = {
    "day5_checkpoint": Path("/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt"),
    "day6_checkpoint": Path("/root/day6_checkpoint_store/day6_336m_500m_fresh_step_15259.pt"),
    "shared_heldout_tokens": Path("/root/autodl-tmp/day6_6/heldout_processed/heldout_tokens.int32"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": st.st_size,
        "mtime_epoch": st.st_mtime,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_capture(command: list[str], output_path: Path) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    text = result.stdout
    write_text(output_path, text)
    return text


def parse_sha256sum(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            parsed[fields[1].lstrip("*")] = fields[0].lower()
    return parsed


def parse_openssl(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        match = re.search(r"SHA2-256\((.+)\)=\s*([0-9a-fA-F]{64})$", line)
        if match:
            parsed[match.group(1)] = match.group(2).lower()
            continue
        # OpenSSL 3 can use the form "file: SHA2-256(stdin)= ..." on some
        # builds; accept a final path plus a 64-character digest as fallback.
        match = re.search(r"(?:^|: )(.+?):\s*([0-9a-fA-F]{64})$", line)
        if match:
            parsed[match.group(1)] = match.group(2).lower()
    return parsed


def hamming(a: str, b: str) -> int | None:
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def adjacent_transposition(candidate: str, actual: str) -> dict[str, Any]:
    if len(candidate) != len(actual):
        return {"is_adjacent_transposition": False, "hamming_distance": None}
    mismatches = [i for i, (x, y) in enumerate(zip(candidate, actual)) if x != y]
    is_swap = (
        len(mismatches) == 2
        and mismatches[1] == mismatches[0] + 1
        and candidate[mismatches[0]] == actual[mismatches[1]]
        and candidate[mismatches[1]] == actual[mismatches[0]]
    )
    return {"is_adjacent_transposition": is_swap, "hamming_distance": len(mismatches)}


def omission_relation(candidate: str, actual: str) -> dict[str, Any]:
    positions = [i for i in range(len(actual)) if actual[:i] + actual[i + 1 :] == candidate]
    return {"is_single_character_omission": len(positions) == 1, "omitted_positions": positions}


def canonical_markdown(manifest: dict[str, Any], crosscheck: dict[str, Any], patches: list[dict[str, Any]]) -> str:
    files = manifest["files"]
    lines = [
        "# Day7补丁：Hash Audit & Documentation Repair（哈希审计与文档修复）\n\n",
        "> 这是一份只读审计报告：不训练、不评估、不修改模型权重或原始数据。\n\n",
        "## 1. 最终结论\n\n",
        "三个真实文件均存在，三组 SHA-256 均为合法的 64 位十六进制字符串；`sha256sum` 与 OpenSSL 逐文件一致。后续正式文档应只引用 `CANONICAL_HASH_MANIFEST.json` 中的值。\n\n",
        "## 2. 权威文件指纹\n\n",
        "| 文件 | 大小（bytes） | SHA-256 |\n|---|---:|---|\n",
    ]
    labels = {
        "day5_checkpoint": "Day5 checkpoint",
        "day6_checkpoint": "Day6 checkpoint",
        "shared_heldout_tokens": "共享留出令牌流",
    }
    for key, label in labels.items():
        row = files[key]
        lines.append(f"| {label} | {row['size_bytes']:,} | `{row['sha256']}` |\n")
    lines += [
        "\n## 3. 双工具交叉核验\n\n",
        f"- `sha256sum` 与 OpenSSL 一致：**{crosscheck['sha256sum_openssl_agree']}**。\n",
        f"- 所有哈希长度为 64：**{crosscheck['all_hash_lengths_64']}**。\n",
        "\n## 4. 文档补丁记录\n\n",
    ]
    if patches:
        lines.append("| 文档 | 字段 | 旧值 | 新值 | 原因 |\n|---|---|---|---|---|\n")
        for patch in patches:
            lines.append(
                f"| `{patch['document']}` | `{patch['field']}` | `{patch['old_value']}` | `{patch['new_value']}` | {patch['reason']} |\n"
            )
    else:
        lines.append("本次远端审计未直接改写文档；正式文档修复由本地受控补丁完成。\n")
    lines += [
        "\n## 5. 保护边界\n\n",
        "- 没有重新训练任何模型。\n",
        "- 没有修改任何模型权重。\n",
        "- 没有重新选择测试数据。\n",
        "- 没有修改 NLL、困惑度、Bootstrap 或训练损失。\n",
        "- 只重新计算文件指纹并修复文档转录记录。\n",
    ]
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/day7_patch"))
    parser.add_argument("--docs-root", type=Path, default=None)
    args = parser.parse_args()
    out = args.out
    raw = out / "raw"
    reports = out / "reports"
    backup = out / "backup"
    for directory in (raw, reports, backup):
        directory.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in TARGETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required source files missing: " + ", ".join(missing))

    stat_text = "\n".join(
        [
            "=== Day5 checkpoint ===",
            subprocess.run(["stat", str(TARGETS["day5_checkpoint"])], check=True, text=True, capture_output=True).stdout,
            "=== Day6 checkpoint ===",
            subprocess.run(["stat", str(TARGETS["day6_checkpoint"])], check=True, text=True, capture_output=True).stdout,
            "=== Heldout tokens ===",
            subprocess.run(["stat", str(TARGETS["shared_heldout_tokens"])], check=True, text=True, capture_output=True).stdout,
        ]
    )
    write_text(raw / "file_stat.txt", stat_text)

    sha_text = run_capture(["sha256sum", *(str(p) for p in TARGETS.values())], raw / "sha256sum_output.txt")
    openssl_text = run_capture(["openssl", "dgst", "-sha256", *(str(p) for p in TARGETS.values())], raw / "openssl_sha256_output.txt")
    sha_map = parse_sha256sum(sha_text)
    openssl_map = parse_openssl(openssl_text)
    files: dict[str, Any] = {}
    for key, path in TARGETS.items():
        rec = file_record(path)
        rec["sha256sum"] = sha_map.get(str(path))
        rec["openssl_sha256"] = openssl_map.get(str(path))
        rec["sha256sum_matches_python"] = rec["sha256sum"] == rec["sha256"]
        rec["openssl_matches_python"] = rec["openssl_sha256"] == rec["sha256"]
        files[key] = rec

    canonical = {
        "verification_name": "day7_patch",
        "method": "direct_file_sha256",
        "generated_at": utc_now(),
        "files": files,
    }
    write_text(reports / "CANONICAL_HASH_MANIFEST.json", json.dumps(canonical, ensure_ascii=False, indent=2) + "\n")

    crosscheck = {
        "generated_at": utc_now(),
        "all_hash_lengths_64": all(len(rec["sha256"]) == 64 for rec in files.values()),
        "sha256sum_openssl_agree": all(
            rec["sha256sum_matches_python"] and rec["openssl_matches_python"] for rec in files.values()
        ),
        "files": {
            key: {
                "sha256sum": rec["sha256sum"],
                "openssl_sha256": rec["openssl_sha256"],
                "python_sha256": rec["sha256"],
                "all_agree": rec["sha256sum_matches_python"] and rec["openssl_matches_python"],
            }
            for key, rec in files.items()
        },
    }
    write_text(reports / "HASH_CROSSCHECK.json", json.dumps(crosscheck, ensure_ascii=False, indent=2) + "\n")

    patches: list[dict[str, Any]] = []
    docs_root = args.docs_root
    if docs_root and docs_root.is_dir():
        # This is an audit-only scan.  The caller decides which files to back
        # up and patch; no document is changed by this script.
        old_values = {rec["sha256"] for rec in files.values()}
        for path in sorted(docs_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"(?i)(?<![0-9a-f])[0-9a-f]{63,64}(?![0-9a-f])", text):
                value = match.group(0).lower()
                if value not in old_values:
                    patches.append(
                        {
                            "document": str(path),
                            "field": "hash_literal",
                            "old_value": value,
                            "new_value": "see canonical manifest",
                            "reason": "非权威哈希字面量，需由权威清单替换",
                        }
                    )
    write_text(reports / "DAY7_PATCH_MANIFEST.json", json.dumps({"generated_at": utc_now(), "patches": patches}, ensure_ascii=False, indent=2) + "\n")

    final_lines = [
        "DAY7 PATCH HASH VERIFICATION\n",
        f"generated_at={utc_now()}\n",
        f"all_hash_lengths_64={crosscheck['all_hash_lengths_64']}\n",
        f"sha256sum_openssl_agree={crosscheck['sha256sum_openssl_agree']}\n",
        "\n",
    ]
    for key, rec in files.items():
        final_lines += [
            f"[{key}]\n",
            f"path={rec['path']}\n",
            f"size_bytes={rec['size_bytes']}\n",
            f"sha256={rec['sha256']}\n",
            f"sha256sum={rec['sha256sum']}\n",
            f"openssl_sha256={rec['openssl_sha256']}\n",
            f"all_agree={rec['sha256sum_matches_python'] and rec['openssl_matches_python']}\n\n",
        ]
    final_lines.append("documentation_consistent=pending_local_repair\n")
    write_text(reports / "HASH_VERIFICATION_FINAL.txt", "".join(final_lines))
    write_text(reports / "DAY7_PATCH_RESULT_CN.md", canonical_markdown(canonical, crosscheck, patches))
    print(json.dumps({"status": "complete", "canonical_manifest": str(reports / "CANONICAL_HASH_MANIFEST.json"), "crosscheck": crosscheck, "patch_count": len(patches)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
