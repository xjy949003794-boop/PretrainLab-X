"""Build a document-disjoint, fresh-token corpus for the Day 6 run.

The writer deliberately uses a contiguous ``int32`` binary file instead of a
Python list or two copied ``.npy`` arrays.  This keeps memory bounded while
retaining a single, auditable train/validation boundary.  Each document is
assigned wholly to either train or validation; no document is truncated or
re-used to fill the other split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .tokenizer import TokenizerAdapter, build_tokenizer


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: object, *, min_chars: int = 200, max_chars: int = 200_000) -> str | None:
    """Apply deterministic, document-level quality filters."""

    if not isinstance(value, str):
        return None
    text = _CONTROL_RE.sub(" ", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) < min_chars:
        return None
    # A long page is capped deterministically, but never split between train
    # and validation. The resulting token sequence is still one document.
    return text[:max_chars]


def _iter_parquet_rows(paths: Iterable[Path], batch_size: int = 1024):
    import pyarrow.parquet as pq

    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        if "text" not in pf.schema.names:
            raise RuntimeError(f"{path} has no text column; refusing an ambiguous corpus")
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
            for value in batch.column("text").to_pylist():
                yield value, path.name


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_fresh_corpus(
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    tokenizer: TokenizerAdapter,
    target_train_tokens: int = 505_000_000,
    target_val_tokens: int = 5_000_000,
    min_chars: int = 200,
    max_chars: int = 200_000,
    parquet_batch_size: int = 1024,
    seed: int = 1337,
) -> dict:
    """Tokenize fresh parquet rows until both split targets are met.

    The output ``tokens.int32`` contains ``[train region][validation region]``
    and ``metadata.json`` records exact counts and provenance. Documents are
    hashed before tokenization, so exact duplicates are removed without ever
    silently replaying a row.
    """

    raw = Path(raw_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(raw.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {raw}")
    token_path = output / "tokens.int32"
    # A stale partial file is dangerous: it could look like a valid corpus.
    if token_path.exists():
        token_path.unlink()
    meta_path = output / "metadata.json"
    if meta_path.exists():
        meta_path.unlink()

    seen: set[str] = set()
    documents_seen = documents_kept = duplicates_removed = filtered = 0
    train_tokens = val_tokens = chars_kept = 0
    train_documents = val_documents = 0
    next_progress = 10_000_000
    started = time.time()

    with token_path.open("wb") as out:
        for raw_value, source_file in _iter_parquet_rows(paths, batch_size=parquet_batch_size):
            documents_seen += 1
            text = clean_text(raw_value, min_chars=min_chars, max_chars=max_chars)
            if text is None:
                filtered += 1
                continue
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                duplicates_removed += 1
                continue
            seen.add(fingerprint)
            ids = tokenizer.encode(text)
            if not ids:
                filtered += 1
                continue
            ids.append(tokenizer.eos_id)
            arr = np.asarray(ids, dtype=np.int32)

            if train_tokens < target_train_tokens:
                # Keep the complete document. Thus the actual train count may
                # exceed the target by at most one document, but is never below
                # the requested lower bound.
                arr.tofile(out)
                train_tokens += int(arr.size)
                train_documents += 1
            elif val_tokens < target_val_tokens:
                arr.tofile(out)
                val_tokens += int(arr.size)
                val_documents += 1
            else:
                break
            documents_kept += 1
            chars_kept += len(text)
            if train_tokens + val_tokens >= next_progress:
                print(
                    json.dumps(
                        {
                            "documents_seen": documents_seen,
                            "documents_kept": documents_kept,
                            "train_tokens": train_tokens,
                            "val_tokens": val_tokens,
                            "elapsed_s": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
                next_progress += 10_000_000

    if train_tokens < target_train_tokens or val_tokens < target_val_tokens:
        raise RuntimeError(
            f"fresh shards exhausted: train={train_tokens} (target {target_train_tokens}), "
            f"val={val_tokens} (target {target_val_tokens}); refusing to replay"
        )

    total_tokens = train_tokens + val_tokens
    digest = _sha256_file(token_path)
    metadata = {
        "dataset_source": "HuggingFaceFW/fineweb-edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "source_files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": _sha256_file(p)} for p in paths
        ],
        "real_data": True,
        "fresh_data": True,
        "replay_enabled": False,
        "document_boundary_split": True,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total_tokens,
        "train_documents": train_documents,
        "val_documents": val_documents,
        "documents_seen": documents_seen,
        "documents_kept": documents_kept,
        "filtered_documents": filtered,
        "duplicates_removed": duplicates_removed,
        "characters_kept": chars_kept,
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dtype": "int32",
        "token_file": token_path.name,
        "sha256_tokens": digest,
        "seed": seed,
        "filters": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "control_chars": "replaced",
            "exact_text_dedup": True,
        },
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - started,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target-train-tokens", type=int, default=505_000_000)
    parser.add_argument("--target-val-tokens", type=int, default=5_000_000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--parquet-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    tok = build_tokenizer(args.tokenizer, cache_dir=args.cache_dir)
    build_fresh_corpus(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        tokenizer=tok,
        target_train_tokens=args.target_train_tokens,
        target_val_tokens=args.target_val_tokens,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        parquet_batch_size=args.parquet_batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
