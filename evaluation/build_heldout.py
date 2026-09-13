"""Build one shared held-out token stream from an unused FineWeb-Edu shard.

The builder is deliberately conservative: it keeps complete documents, uses
the same cleaning/tokenizer path as Day 6, and records what could and could
not be reconstructed about overlap with the earlier API-sourced Day 5 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fresh_preprocessing import clean_text  # noqa: E402
from data.tokenizer import build_tokenizer  # noqa: E402


WINDOW = 64
STRIDE = 16


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def token_window_hashes(ids: np.ndarray, *, window: int = WINDOW, stride: int = STRIDE) -> Iterable[bytes]:
    """Yield cryptographic hashes of long token windows without copying the corpus."""
    if ids.size < window:
        return
    for start in range(0, int(ids.size) - window + 1, stride):
        # int32 is the on-disk representation; the tokenizer is identical for
        # all three runs, so this is a useful exact token-span screen.
        yield hashlib.sha256(np.asarray(ids[start : start + window], dtype=np.int32).tobytes()).digest()


def load_existing_window_hashes(path: Path, *, max_tokens: int | None = None) -> set[bytes]:
    arr = np.load(path, mmap_mode="r")
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return set(token_window_hashes(np.asarray(arr)))


def iter_rows(paths: Iterable[Path], batch_size: int = 1024):
    import pyarrow.parquet as pq

    wanted = ("text", "url", "id", "dump", "language", "language_score", "token_count")
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        columns = [c for c in wanted if c in pf.schema.names]
        if "text" not in columns:
            raise RuntimeError(f"{path} has no text column")
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            values = {name: batch.column(name).to_pylist() for name in columns}
            for i in range(batch.num_rows):
                yield {name: values[name][i] for name in columns}


def build(
    *,
    raw_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    day5_train: Path | None,
    day5_source_status: str,
    day6_source_shards: list[str],
    target_tokens: int,
    min_chars: int,
    max_chars: int,
    batch_size: int,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "heldout_tokens.int32"
    index_path = output_dir / "heldout_doc_index.jsonl"
    manifest_path = output_dir / "heldout_manifest.json"
    for path in (token_path, index_path, manifest_path):
        if path.exists():
            path.unlink()

    paths = sorted(raw_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shard in {raw_dir}")
    tok = build_tokenizer(tokenizer_name)
    prior_hashes: set[bytes] = set()
    day5_screen_status = "not_run"
    day5_screen_tokens = 0
    if day5_train is not None and day5_train.exists():
        prior_hashes = load_existing_window_hashes(day5_train)
        arr = np.load(day5_train, mmap_mode="r")
        day5_screen_tokens = int(arr.size)
        day5_screen_status = "token_window_screening"
    else:
        day5_screen_status = "unavailable_day5_token_stream"

    seen_text: set[str] = set()
    docs_seen = docs_kept = filtered = duplicate_text = overlap_rejected = 0
    chars_kept = tokens_kept = 0
    started = time.time()
    doc_records: list[dict] = []
    next_progress = 1_000_000

    with token_path.open("wb") as out, index_path.open("w", encoding="utf-8") as index:
        for row in iter_rows(paths, batch_size=batch_size):
            docs_seen += 1
            text = clean_text(row.get("text"), min_chars=min_chars, max_chars=max_chars)
            if text is None:
                filtered += 1
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen_text:
                duplicate_text += 1
                continue
            seen_text.add(text_hash)
            ids = tok.encode(text)
            if not ids:
                filtered += 1
                continue
            ids.append(tok.eos_id)
            arr = np.asarray(ids, dtype=np.int32)
            # Exact token-span screen against the only reconstructable Day 5
            # stream. A source-shard disjointness check is recorded separately
            # for Day 6; it is not advertised as semantic zero overlap.
            if prior_hashes and any(h in prior_hashes for h in token_window_hashes(arr)):
                overlap_rejected += 1
                continue
            if tokens_kept >= target_tokens:
                break
            start = tokens_kept
            arr.tofile(out)
            tokens_kept += int(arr.size)
            docs_kept += 1
            chars_kept += len(text)
            record = {
                "doc_id": docs_kept - 1,
                "token_start": start,
                "token_end": tokens_kept,
                "text_sha256": text_hash,
                "source_shard": Path(str(row.get("_source_shard", paths[0].name))).name,
                "url": row.get("url"),
                "row_id": row.get("id"),
                "dump": row.get("dump"),
                "language": row.get("language"),
            }
            index.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_records.append(record)
            if tokens_kept >= next_progress:
                print(json.dumps({"documents_seen": docs_seen, "documents_kept": docs_kept, "tokens": tokens_kept, "elapsed_s": round(time.time() - started, 1)}), flush=True)
                next_progress += 1_000_000

    if tokens_kept < target_tokens:
        raise RuntimeError(f"candidate shard ended at {tokens_kept} tokens; target is {target_tokens}")
    token_digest = sha256_file(token_path)
    shard_meta = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths]
    manifest = {
        "kind": "day6.6_shared_heldout",
        "dataset_source": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "source_shards": shard_meta,
        "target_tokens": target_tokens,
        "total_tokens": tokens_kept,
        "documents_seen": docs_seen,
        "documents_kept": docs_kept,
        "filtered_documents": filtered,
        "duplicate_text_documents": duplicate_text,
        "day5_token_window_overlap_rejected": overlap_rejected,
        "characters_kept": chars_kept,
        "tokenizer": tokenizer_name,
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "dtype": "int32",
        "token_file": token_path.name,
        "doc_index_file": index_path.name,
        "sha256_tokens": token_digest,
        "seed": seed,
        "filters": {"min_chars": min_chars, "max_chars": max_chars, "control_chars": "replaced", "exact_text_dedup_within_candidate": True},
        "overlap_audit": {
            "day5_exact_document_status": "not_reconstructed_api_source",
            "day5_token_window_status": day5_screen_status,
            "day5_train_tokens_screened": day5_screen_tokens,
            "day5_token_window": {"window_tokens": WINDOW, "stride_tokens": STRIDE},
            "day6_source_shard_disjoint": not any(x["name"] in day6_source_shards for x in shard_meta),
            "day6_exact_document_status": "source_shard_disjoint_only",
            "url_overlap_status": "not_reconstructed_for_day5_api_source",
            "semantic_overlap_status": "not_proven",
            "limitation": "Day5 used dataset-server API rows without retained shard/URL manifest; zero semantic overlap cannot be claimed.",
        },
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--day5-train", type=Path, default=None)
    p.add_argument("--day5-source-status", default="api_source_without_shard_manifest")
    p.add_argument("--day6-source-shard", action="append", default=["000_00000.parquet"])
    p.add_argument("--target-tokens", type=int, default=10_000_000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    build(raw_dir=a.raw_dir, output_dir=a.output_dir, tokenizer_name=a.tokenizer, day5_train=a.day5_train, day5_source_status=a.day5_source_status, day6_source_shards=a.day6_source_shard, target_tokens=a.target_tokens, min_chars=a.min_chars, max_chars=a.max_chars, batch_size=a.batch_size, seed=a.seed)


if __name__ == "__main__":
    main()

