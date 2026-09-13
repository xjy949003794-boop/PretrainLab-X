"""Evaluate a saved PretrainLab-X checkpoint on the shared held-out stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import LlamaConfig, LlamaForCausalLM  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_model(checkpoint: Path, *, device: torch.device, cfg_values: dict) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg_values = dict(cfg_values)
    saved_cfg = payload.get("config") or {}
    # The training config has the architecture fields; prefer the checkpoint
    # copy when present and keep the evaluator's max sequence length explicit.
    for key in ("dim", "n_layers", "n_heads", "n_kv_heads", "multiple_of", "dropout", "rope_base", "tie_embeddings", "gradient_checkpointing", "max_seq_len"):
        if key in saved_cfg:
            model_cfg_values[key] = saved_cfg[key]
    meta = payload.get("data_meta") or {}
    vocab_size = int(meta.get("vocab_size", model_cfg_values.get("vocab_size", 32000)))
    model_cfg = LlamaConfig.from_dict(vocab_size, model_cfg_values)
    model = LlamaForCausalLM(model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    step = int(payload.get("step", payload.get("last_record", {}).get("step", -1)))
    del payload
    gc.collect()
    return model, {"step": step, "param_count": model.num_parameters(), "model_config": model_cfg.__dict__, "checkpoint_sha256": sha256_file(checkpoint)}


def read_docs(index_path: Path) -> list[dict]:
    docs: list[dict] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


def evaluate(
    *,
    checkpoint: Path,
    tokens_path: Path,
    index_path: Path,
    output_json: Path,
    doc_metrics_path: Path,
    max_tokens: int,
    batch_windows: int,
    seq_len: int,
    model_cfg: dict,
    device_name: str = "cuda",
) -> dict:
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    arr = np.memmap(tokens_path, mode="r", dtype=np.int32)
    docs = read_docs(index_path)
    model, ckpt_meta = load_checkpoint_model(checkpoint, device=device, cfg_values=model_cfg)
    eval_docs: list[dict] = []
    remaining = max_tokens
    for doc in docs:
        length = int(doc["token_end"]) - int(doc["token_start"])
        if remaining <= 1:
            break
        scored = min(length - 1, remaining)
        if scored <= 0:
            continue
        item = dict(doc)
        item["score_tokens"] = scored
        item["partial_document"] = scored < length - 1
        eval_docs.append(item)
        remaining -= scored
    if not eval_docs:
        raise RuntimeError("held-out index has no scoreable tokens")

    # Construct fixed-size causal windows. Padding is only at the right edge;
    # masked positions are never included in the loss.
    windows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for doc_i, doc in enumerate(eval_docs):
        start = int(doc["token_start"])
        score_tokens = int(doc["score_tokens"])
        ids = np.asarray(arr[start : start + score_tokens + 1], dtype=np.int64)
        for pos in range(0, score_tokens, seq_len):
            n = min(seq_len, score_tokens - pos)
            x = ids[pos : pos + n]
            y = ids[pos + 1 : pos + n + 1]
            windows.append((doc_i, x, y))

    sums = np.zeros(len(eval_docs), dtype=np.float64)
    counts = np.zeros(len(eval_docs), dtype=np.int64)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_windows):
            group = windows[offset : offset + batch_windows]
            x = torch.zeros((len(group), seq_len), dtype=torch.long, device=device)
            y = torch.zeros((len(group), seq_len), dtype=torch.long, device=device)
            mask = torch.zeros((len(group), seq_len), dtype=torch.bool, device=device)
            for row, (_, xi, yi) in enumerate(group):
                n = len(xi)
                x[row, :n] = torch.from_numpy(xi).to(device)
                y[row, :n] = torch.from_numpy(yi).to(device)
                mask[row, :n] = True
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(x, None)
            token_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").reshape(len(group), seq_len)
            token_loss = token_loss * mask
            batch_sums = token_loss.sum(dim=1).detach().cpu().numpy()
            batch_counts = mask.sum(dim=1).detach().cpu().numpy()
            for row, (doc_i, _, _) in enumerate(group):
                sums[doc_i] += float(batch_sums[row])
                counts[doc_i] += int(batch_counts[row])

    elapsed = time.time() - started
    total_nll = float(sums.sum())
    total_tokens = int(counts.sum())
    weighted_nll = total_nll / max(total_tokens, 1)
    result = {
        "kind": "day6.6_shared_heldout_evaluation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
        "checkpoint_step": ckpt_meta["step"],
        "param_count": ckpt_meta["param_count"],
        "model_config": ckpt_meta["model_config"],
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "eval_mode": "model.eval + torch.inference_mode; no backward or optimizer",
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "seq_len": seq_len,
        "batch_windows": batch_windows,
        "max_tokens_requested": max_tokens,
        "tokens_evaluated": total_tokens,
        "documents_evaluated": len(eval_docs),
        "documents_partial": sum(bool(d["partial_document"]) for d in eval_docs),
        "total_nll": total_nll,
        "weighted_nll": weighted_nll,
        "perplexity": float(math.exp(weighted_nll)),
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
        "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else None,
        "heldout_tokens_sha256": sha256_file(tokens_path),
        "doc_metrics_path": str(doc_metrics_path),
    }
    doc_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with doc_metrics_path.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(eval_docs):
            record = {"doc_id": doc["doc_id"], "token_start": doc["token_start"], "token_end": doc["token_end"], "tokens": int(counts[i]), "nll_sum": float(sums[i]), "nll": float(sums[i] / max(counts[i], 1)), "partial_document": bool(doc["partial_document"]), "text_sha256": doc.get("text_sha256"), "source_shard": doc.get("source_shard"), "url": doc.get("url")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--tokens", required=True, type=Path)
    p.add_argument("--doc-index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--doc-metrics", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--multiple-of", type=int, default=256)
    p.add_argument("--gradient-checkpointing", action="store_true")
    a = p.parse_args()
    cfg = {"dim": a.dim, "n_layers": a.n_layers, "n_heads": a.n_heads, "n_kv_heads": a.n_kv_heads, "multiple_of": a.multiple_of, "max_seq_len": a.seq_len, "dropout": 0.0, "rope_base": 10000.0, "tie_embeddings": False, "gradient_checkpointing": a.gradient_checkpointing}
    evaluate(checkpoint=a.checkpoint, tokens_path=a.tokens, index_path=a.doc_index, output_json=a.output, doc_metrics_path=a.doc_metrics, max_tokens=a.max_tokens, batch_windows=a.batch_windows, seq_len=a.seq_len, model_cfg=cfg)


if __name__ == "__main__":
    main()

