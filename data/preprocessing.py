"""Stream and tokenize a real FineWeb-Edu sample into a reproducible corpus."""

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
    """Apply transparent, deterministic document-level filters."""

    if not isinstance(value, str):
        return None
    text = _CONTROL_RE.sub(" ", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) < min_chars:
        return None
    return text[:max_chars]


def _datasets_stream(dataset: str, config: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("datasets is required for real Day 4 data") from exc
    return load_dataset(dataset, name=config, split=split, streaming=True)


def _dataset_server_stream(
    dataset: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
    timeout: float = 60.0,
):
    """Stream rows through Hugging Face's read-only datasets-server API.

    Some rental images deliberately block direct outbound connections from the
    container.  The API endpoint is still useful when the host has a proxy or
    when this script is run on a workstation.  It returns the same ``text``
    field as the regular ``datasets`` streaming loader, without downloading a
    parquet shard or requiring ``pyarrow``.
    """

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("requests is required for the datasets-server fallback") from exc
    endpoint = "https://datasets-server.huggingface.co/rows"
    offset = 0
    while True:
        # The public endpoint is rate-limited.  A bounded retry keeps a long
        # corpus build reproducible without hammering the service.
        for attempt in range(6):
            response = requests.get(
                endpoint,
                params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": page_size},
                timeout=timeout,
            )
            if response.status_code != 429:
                response.raise_for_status()
                break
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(1.0, min(30.0, float(retry_after))) if retry_after else min(30.0, 2.0 ** attempt)
            except ValueError:
                delay = min(30.0, 2.0 ** attempt)
            print(f"[data] datasets-server rate limit at offset={offset}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
        else:
            response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows") or []
        if not rows:
            break
        for item in rows:
            row = item.get("row", item) if isinstance(item, dict) else item
            if isinstance(row, dict):
                yield row
        offset += len(rows)
        # Stay below the public viewer's per-minute request budget.
        time.sleep(2.2)
        # A short page is the server's explicit end-of-split signal.
        if len(rows) < page_size:
            break


def _dataset_stream(dataset: str, config: str, split: str, source_mode: str):
    if source_mode == "api":
        return _dataset_server_stream(dataset, config, split)
    if source_mode == "datasets":
        return _datasets_stream(dataset, config, split)
    # auto: prefer the canonical library, but fall back only on an immediate
    # import/connection failure.  Never substitute synthetic text.
    try:
        return _datasets_stream(dataset, config, split)
    except Exception as exc:
        print(f"[data] datasets streaming unavailable ({exc}); trying datasets-server API", flush=True)
        return _dataset_server_stream(dataset, config, split)


def _write_split(tokens: list[int], output_dir: Path, val_fraction: float, seed: int) -> tuple[int, int, str]:
    if len(tokens) < 1000:
        raise RuntimeError(f"only {len(tokens)} tokens survived preprocessing; refusing a fake Day 4 corpus")
    array = np.asarray(tokens, dtype=np.int32)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    # Keep validation deterministic and preserve contiguous language-model context.
    cut = max(1, int(len(array) * (1.0 - val_fraction)))
    train = array[:cut]
    val = array[cut:]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "tokens.npy", array)
    np.save(output_dir / "train_split.npy", train)
    np.save(output_dir / "val_split.npy", val)
    return len(train), len(val), digest


def build_corpus(
    *,
    dataset: str,
    config: str,
    split: str,
    tokenizer: TokenizerAdapter,
    output_dir: str | Path,
    target_tokens: int,
    max_documents: int | None = None,
    min_chars: int = 200,
    max_chars: int = 200_000,
    val_fraction: float = 0.01,
    seed: int = 1337,
    source_mode: str = "auto",
) -> dict:
    """Stream documents until ``target_tokens`` and write token arrays + provenance."""

    output = Path(output_dir)
    tokens: list[int] = []
    seen: set[str] = set()
    documents_seen = documents_kept = duplicates = 0
    chars_kept = 0
    stream = _dataset_stream(dataset, config, split, source_mode)
    for row in stream:
        documents_seen += 1
        if max_documents is not None and documents_seen > max_documents:
            break
        text = clean_text(row.get("text") if isinstance(row, dict) else None, min_chars=min_chars, max_chars=max_chars)
        if text is None:
            continue
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        ids = tokenizer.encode(text)
        if not ids:
            continue
        documents_kept += 1
        chars_kept += len(text)
        tokens.extend(ids)
        tokens.append(tokenizer.eos_id)
        if len(tokens) >= target_tokens:
            tokens = tokens[:target_tokens]
            break

    train_tokens, val_tokens, digest = _write_split(tokens, output, val_fraction, seed)
    metadata = {
        "dataset_source": "HuggingFaceFW/fineweb-edu",
        "dataset": dataset,
        "config": config,
        "split": split,
        "source_mode": source_mode,
        "real_data": True,
        "documents_seen": documents_seen,
        "documents_kept": documents_kept,
        "duplicates_removed": duplicates,
        "characters_kept": chars_kept,
        "tokens": len(tokens),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "seed": seed,
        "sha256_tokens": digest,
        "filters": {"min_chars": min_chars, "max_chars": max_chars, "control_chars": "replaced", "exact_text_dedup": True},
        "created_at_unix": time.time(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", default="data/fineweb_10m")
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-mode", choices=("auto", "datasets", "api"), default="auto")
    args = parser.parse_args()
    tok = build_tokenizer(args.tokenizer, cache_dir=args.cache_dir)
    metadata = build_corpus(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        tokenizer=tok,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        max_documents=args.max_documents,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        val_fraction=args.val_fraction,
        seed=args.seed,
        source_mode=args.source_mode,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
