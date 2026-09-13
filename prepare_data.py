"""Create a small, reproducible tokenized corpus for the Day-1 smoke run.

The script deliberately caps both examples and tokens. It is not intended to
download a full pretraining corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


FALLBACK_TEXT = [
    "A language model learns statistical regularities from sequences of tokens.",
    "Large scale pretraining combines a decoder transformer, an optimizer, and a carefully measured data stream.",
    "The training loss is a diagnostic signal, not a complete measure of model quality.",
    "Reproducible experiments record the random seed, model configuration, data source, and checkpoint metadata.",
    "Attention lets each position use information from earlier positions while a causal mask prevents looking ahead.",
    "RMSNorm, rotary position embeddings, and gated feed forward layers are common components of modern decoder models.",
]


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_hf_text(dataset_name: str, max_records: int) -> Iterable[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_records:
            break
        text = row.get("text") if isinstance(row, dict) else None
        if text and isinstance(text, str):
            text = " ".join(text.split())
            if len(text) >= 40:
                yield text


def collect_text(cfg: dict) -> tuple[list[str], str]:
    max_records = int(cfg.get("data_records", 20000))
    preferred = str(cfg.get("dataset_preference", "fineweb-edu"))
    candidates = []
    if preferred == "fineweb-edu":
        candidates.append("HuggingFaceFW/fineweb-edu")
    candidates.append("roneneldan/TinyStories")

    for name in candidates:
        try:
            rows = list(iter_hf_text(name, max_records))
            if rows:
                return rows, name
        except Exception as exc:  # network/auth/version failures are logged below
            print(f"[data] {name} unavailable: {type(exc).__name__}: {exc}")

    # A deterministic local fallback keeps the engineering smoke test runnable
    # even if the dataset service is temporarily unavailable.
    rows = (FALLBACK_TEXT * ((max_records // len(FALLBACK_TEXT)) + 1))[:max_records]
    return rows, "builtin-fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)

    out_root = Path(cfg.get("output_dir", "."))
    token_dir = out_root / cfg.get("data_dir", "data/tokenized")
    token_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, source = collect_text(cfg)
    # Keep a human-readable audit sample, never the full streaming dataset.
    with (raw_dir / "sample.jsonl").open("w", encoding="utf-8") as f:
        for row in rows[: min(len(rows), 1000)]:
            f.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")

    from transformers import AutoTokenizer  # type: ignore

    tokenizer_name = str(cfg.get("tokenizer_name", "gpt2"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name} has no eos_token_id")

    cap = int(cfg.get("data_tokens", 2_000_000))
    ids: list[int] = []
    for row in rows:
        encoded = tokenizer(row, add_special_tokens=False)["input_ids"]
        ids.extend(encoded)
        ids.append(eos_id)
        if len(ids) >= cap:
            ids = ids[:cap]
            break
    if len(ids) < int(cfg.get("max_seq_len", 256)) * 20:
        raise RuntimeError(f"Only {len(ids)} tokens available; need a larger smoke corpus")

    arr = np.asarray(ids, dtype=np.int32)
    np.save(token_dir / "train.npy", arr)
    split = max(int(len(arr) * 0.9), 1)
    np.save(token_dir / "train_split.npy", arr[:split])
    np.save(token_dir / "val_split.npy", arr[split:])

    meta = {
        "dataset_source": source,
        "records_seen": len(rows),
        "tokens": int(len(arr)),
        "tokenizer": tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "seed": seed,
    }
    with (token_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

