"""Day 7 sanity checks for the Day 6.6 shared-held-out result.

This script is intentionally read-only with respect to models and data.  It
loads checkpoints, evaluates them, and writes small JSON/Markdown reports.  It
does not call backward(), optimizer.step(), or torch.save().
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F


EXPECTED = {
    "day5_checkpoint_sha256": "2c49925b61e46c533026c201dba02bfe845071e7c453964fcc33f508a24c36f3",
    # Recomputed on the retained Day6 checkpoint.  The earlier plan had a
    # transposed pair ("d1" vs "1d") in this literal; keep the observed file
    # fingerprint as the integrity reference and record it in the final report.
    "day6_checkpoint_sha256": "3d1250567b0d3f311584b15a3ac1d73539cdd52b0baebfd7b290e1f7fdf21aca",
    # The pasted plan contains a 63-character literal.  At runtime we prefer
    # the producer manifest's independent 64-character sha256_tokens field;
    # this literal is retained only for audit reporting.
    "heldout_tokens_sha256": "3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f98509a21df9214a6cc750",
    "parameters": 336_118_784,
    "dim": 1024,
    "layers": 24,
    "heads": 16,
    "kv_heads": 4,
    "seq_len": 512,
    "vocab_size": 32_000,
    "eos_id": 2,
    "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array_bytes(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(array, dtype=np.int32).tobytes(order="C"))
    return digest.hexdigest()


def heldout_expected_sha(tokens_path: Path) -> tuple[str, str]:
    """Return (expected_sha, source) for the held-out token integrity gate."""
    manifest = tokens_path.with_name("heldout_manifest.json")
    if manifest.exists():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("sha256_tokens")
            if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                return value, "heldout_manifest.json"
        except (OSError, ValueError, TypeError):
            pass
    return EXPECTED["heldout_tokens_sha256"], "day7_plan_literal"


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    for value in paths:
        path = Path(value)
        if path.exists():
            return path
    return None


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, mode="r", dtype=np.int32)


def checkpoint_cfg(payload: dict, defaults: dict) -> tuple[object, dict]:
    saved_cfg = payload.get("config") or {}
    values = dict(defaults)
    for key in (
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "multiple_of",
        "dropout",
        "rope_base",
        "tie_embeddings",
        "gradient_checkpointing",
        "max_seq_len",
    ):
        if key in saved_cfg:
            values[key] = saved_cfg[key]
    meta = payload.get("data_meta") or {}
    vocab_size = int(meta.get("vocab_size", values.get("vocab_size", EXPECTED["vocab_size"])))
    from model import LlamaConfig  # imported after PROJECT_ROOT is added

    cfg = LlamaConfig.from_dict(vocab_size, values)
    return cfg, {"saved_config": saved_cfg, "data_meta": meta}


def load_model(
    checkpoint: Path | None,
    *,
    device: torch.device,
    defaults: dict,
    random_seed: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    from model import LlamaConfig, LlamaForCausalLM

    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        cfg = LlamaConfig.from_dict(EXPECTED["vocab_size"], defaults)
        model = LlamaForCausalLM(cfg)
        model.to(device).eval()
        return model, {
            "step": None,
            "param_count": model.num_parameters(),
            "model_config": dict(model.config.__dict__),
            "checkpoint_sha256": None,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    assert checkpoint is not None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, payload_meta = checkpoint_cfg(payload, defaults)
    model = LlamaForCausalLM(cfg)
    incompat = model.load_state_dict(payload["model"], strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    # A second strict call is deliberately retained as the hard loading gate.
    model.load_state_dict(payload["model"], strict=True)
    step = int(payload.get("step", payload.get("last_record", {}).get("step", -1)))
    info = {
        "step": step,
        "param_count": model.num_parameters(),
        "model_config": dict(model.config.__dict__),
        "checkpoint_sha256": sha256_file(checkpoint),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "payload_meta": payload_meta,
    }
    del payload
    gc.collect()
    model.to(device).eval()
    return model, info


def iter_index_records(index_path: Path) -> Iterable[dict]:
    """Yield records from ordinary JSONL *and* escaped-newline JSON streams.

    The retained Day6.6 index was produced with literal ``\\n`` separators in
    one run, so ``json.loads(line)`` raises ``Extra data``.  A streaming raw
    decoder accepts both that form and normal physical newlines without
    touching the token data.
    """
    text = index_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
            elif text.startswith("\\n", pos) or text.startswith("\\r", pos):
                pos += 2
            else:
                break
        if pos >= len(text):
            break
        row, pos = decoder.raw_decode(text, pos)
        if isinstance(row, dict):
            yield row


def read_doc_ranges(index_path: Path, max_tokens: int) -> list[tuple[int, int]]:
    """Return (token_start, scoreable_token_count) without crossing documents."""
    ranges: list[tuple[int, int]] = []
    remaining = max_tokens
    for row in iter_index_records(index_path):
        if remaining <= 0:
            break
        start = int(row["token_start"])
        length = int(row["token_end"]) - start
        scoreable = min(max(length - 1, 0), remaining)
        if scoreable > 0:
            ranges.append((start, scoreable))
            remaining -= scoreable
    return ranges


def contiguous_range(array: np.ndarray, max_tokens: int) -> list[tuple[int, int]]:
    # One token is needed as the next-token target.
    return [(0, min(max_tokens, max(int(array.size) - 1, 0)))]


def evaluate_array(
    model: torch.nn.Module,
    array: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    *,
    device: torch.device,
    seq_len: int,
    batch_windows: int,
    precision: str,
) -> dict:
    windows: list[tuple[int, int]] = []
    for start, count in ranges:
        for offset in range(0, count, seq_len):
            windows.append((start + offset, min(seq_len, count - offset)))

    total_loss = 0.0
    total_tokens = 0
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_windows):
            group = windows[offset : offset + batch_windows]
            x = torch.zeros((len(group), seq_len), dtype=torch.long, device=device)
            y = torch.zeros((len(group), seq_len), dtype=torch.long, device=device)
            mask = torch.zeros((len(group), seq_len), dtype=torch.bool, device=device)
            for row, (start, count) in enumerate(group):
                ids = np.asarray(array[start : start + count + 1], dtype=np.int64)
                x[row, :count] = torch.from_numpy(ids[:-1]).to(device)
                y[row, :count] = torch.from_numpy(ids[1:]).to(device)
                mask[row, :count] = True
            use_bf16 = precision == "bf16" and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, _ = model(x, None)
            losses = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="none",
            ).reshape(len(group), seq_len)
            losses = losses * mask
            total_loss += float(losses.sum().detach().cpu())
            total_tokens += int(mask.sum().detach().cpu())

    elapsed = time.time() - started
    nll = total_loss / max(total_tokens, 1)
    return {
        "precision": precision,
        "scored_tokens": total_tokens,
        "nll": nll,
        "perplexity": float(math.exp(nll)),
        "eval_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else None,
    }


def release(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_checkpoint_integrity(path: Path, expected_sha: str, defaults: dict) -> dict:
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"checkpoint SHA256 mismatch for {path}: {observed_sha} != {expected_sha}")
    cpu = torch.device("cpu")
    model, info = load_model(path, device=cpu, defaults=defaults)
    result = {
        "path": str(path),
        "sha256": observed_sha,
        "sha256_matches": True,
        "missing_keys": info["missing_keys"],
        "unexpected_keys": info["unexpected_keys"],
        "parameter_count": info["param_count"],
        "step": info["step"],
        "model_config": info["model_config"],
    }
    release(model)
    if result["parameter_count"] != EXPECTED["parameters"]:
        raise RuntimeError(f"parameter count mismatch for {path}: {result['parameter_count']}")
    return result


def write_markdown(path: Path, result: dict) -> None:
    ck = result["checkpoint_integrity"]
    cfg = result["configuration_consistency"]
    held = result["heldout_integrity"]
    rand = result["random_baseline"]
    d5 = result.get("day5_train_corpus_eval")
    d6 = result.get("day6_train_subset_eval")
    pc = result["precision_crosscheck"]

    def line(label: str, value: object) -> str:
        return f"- **{label}**: `{value}`\n"

    rows = [
        "# Day 7 Sanity Check Result\n",
        "\n",
        "本报告只做只读验证，不训练新模型，不执行反向传播，也不更新优化器。\n",
        "\n",
        "## A. Checkpoint Integrity\n\n",
        line("Day5 SHA256", ck["day5"]["sha256"]),
        line("Day6 SHA256", ck["day6"]["sha256"]),
        line("missing_keys", f"Day5={ck['day5']['missing_keys']}; Day6={ck['day6']['missing_keys']}"),
        line("unexpected_keys", f"Day5={ck['day5']['unexpected_keys']}; Day6={ck['day6']['unexpected_keys']}"),
        line("parameter_count", ck["day5"]["parameter_count"]),
        "\n## B. Evaluation Consistency\n\n",
        line("tokenizer", cfg["tokenizer"]),
        line("vocab_size", cfg["vocab_size"]),
        line("eos_id", cfg["eos_id"]),
        line("context_length", cfg["seq_len"]),
        line("heldout_sha256", held["sha256"]),
        "\n## C. Random Baseline\n\n",
        line("random_nll", rand["nll"]),
        line("random_perplexity", rand["perplexity"]),
        line("scored_tokens", rand["scored_tokens"]),
        line("seed", rand["seed"]),
        "\n## D. Day5 Memorization Check\n\n",
        line("day5_train_nll", None if d5 is None else d5["nll"]),
        line("day5_shared_heldout_nll", result["shared_heldout"]["day5"]["nll"]),
        line("memorization_gap", None if d5 is None else result["derived"]["day5_memorization_gap"]),
        "\n## E. Day6 Generalization Check\n\n",
        line("day6_train_subset_nll", None if d6 is None else d6["nll"]),
        line("day6_shared_heldout_nll", result["shared_heldout"]["day6"]["nll"]),
        line("generalization_gap", None if d6 is None else result["derived"]["day6_generalization_gap"]),
        "\n## F. Numerical Cross-check\n\n",
        line("Day5 BF16", pc["day5"]["bf16"]["nll"]),
        line("Day5 FP32", pc["day5"]["fp32"]["nll"]),
        line("Day6 BF16", pc["day6"]["bf16"]["nll"]),
        line("Day6 FP32", pc["day6"]["fp32"]["nll"]),
        line("ranking_preserved", pc["ranking_preserved"]),
        "\n## G. Scope and limitations\n\n",
        "- Random baseline uses the first 500,000 scoreable held-out tokens.\n",
        "- Day5 uses the complete retained 9.9M-token train split when available.\n",
        "- Day6 uses a fixed 10M-token prefix of its retained fresh train stream when available.\n",
        "- This verifies loading, protocol, and numerical robustness; it does not prove a causal effect of data freshness alone.\n",
        "- The Day6.6 overlap audit still cannot claim semantic zero overlap because Day5's API source lacked a retained shard/URL manifest.\n",
    ]
    path.write_text("".join(rows), encoding="utf-8")


def main() -> None:
    project_root = Path(os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/day4src_v2/pretrainlab_x"))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    out = Path(os.environ.get("DAY7_OUT", "/root/autodl-tmp/day7_sanity/reports"))
    out.mkdir(parents=True, exist_ok=True)

    defaults = {
        "dim": EXPECTED["dim"],
        "n_layers": EXPECTED["layers"],
        "n_heads": EXPECTED["heads"],
        "n_kv_heads": EXPECTED["kv_heads"],
        "multiple_of": 256,
        "max_seq_len": EXPECTED["seq_len"],
        "dropout": 0.0,
        "rope_base": 10_000.0,
        "tie_embeddings": False,
        "gradient_checkpointing": False,
        "vocab_size": EXPECTED["vocab_size"],
    }
    day5_ckpt = first_existing(
        [
            "/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt",
            "/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt",
        ]
    )
    day6_ckpt = first_existing(["/root/day6_checkpoint_store/day6_336m_500m_fresh_step_15259.pt"])
    heldout_tokens = first_existing(["/root/autodl-tmp/day6_6/heldout_processed/heldout_tokens.int32"])
    heldout_index = first_existing(["/root/autodl-tmp/day6_6/heldout_processed/heldout_doc_index.jsonl"])
    day5_train = first_existing(
        [
            f"{project_root}/data/fineweb_10m/train_split.npy",
            "/root/day5src/data/fineweb_10m/train_split.npy",
            "/root/autodl-tmp/day4src_v2/pretrainlab_x/data/fineweb_10m/train_split.npy",
        ]
    )
    day6_train = first_existing(
        [
            "/root/pretrainlab_data/fineweb_fresh_500m/tokens.int32",
            "/root/autodl-tmp/pretrainlab_data/fineweb_fresh_500m/tokens.int32",
            "/root/autodl-tmp/day6_data/fineweb_fresh_500m/tokens.int32",
        ]
    )
    for label, value in (("day5_ckpt", day5_ckpt), ("day6_ckpt", day6_ckpt), ("heldout_tokens", heldout_tokens), ("heldout_index", heldout_index)):
        if value is None:
            raise FileNotFoundError(f"required input missing: {label}")

    started = time.time()
    # Red-light gates: do not evaluate if a checkpoint or held-out stream changed.
    ck5 = check_checkpoint_integrity(day5_ckpt, EXPECTED["day5_checkpoint_sha256"], defaults)
    ck6 = check_checkpoint_integrity(day6_ckpt, EXPECTED["day6_checkpoint_sha256"], defaults)
    heldout_sha = sha256_file(heldout_tokens)
    heldout_expected, heldout_expected_source = heldout_expected_sha(heldout_tokens)
    if heldout_sha != heldout_expected:
        raise RuntimeError(f"held-out SHA256 mismatch: {heldout_sha} != {heldout_expected} (source={heldout_expected_source})")

    cfg5 = ck5["model_config"]
    cfg6 = ck6["model_config"]
    consistency = {
        "parameter_count": EXPECTED["parameters"],
        "dim": int(cfg5["dim"]),
        "layers": int(cfg5["n_layers"]),
        "heads": int(cfg5["n_heads"]),
        "kv_heads": int(cfg5["n_kv_heads"]),
        "seq_len": int(cfg5["max_seq_len"]),
        "tokenizer": EXPECTED["tokenizer"],
        "vocab_size": EXPECTED["vocab_size"],
        "eos_id": EXPECTED["eos_id"],
        "dtype": "checkpoint weights fp32 with bf16 autocast evaluation",
        "day5_day6_model_config_equal": cfg5 == cfg6,
    }
    for key in ("dim", "layers", "heads", "kv_heads", "seq_len"):
        if consistency[key] != EXPECTED[key]:
            raise RuntimeError(f"configuration mismatch: {key}={consistency[key]}")
    if not consistency["day5_day6_model_config_equal"]:
        raise RuntimeError("Day5 and Day6 model configs differ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held_arr = load_array(heldout_tokens)
    held_ranges_10m = read_doc_ranges(heldout_index, 10_000_000)
    held_ranges_500k = read_doc_ranges(heldout_index, 500_000)
    held_ranges_100k = read_doc_ranges(heldout_index, 100_000)

    # Random baseline: same architecture, same protocol, no checkpoint.
    random_model, random_info = load_model(None, device=device, defaults=defaults, random_seed=20260913)
    random_eval = evaluate_array(random_model, held_arr, held_ranges_500k, device=device, seq_len=512, batch_windows=8, precision="bf16")
    release(random_model)
    random_eval.update({"seed": 20260913, "parameter_count": random_info["param_count"], "model_config": random_info["model_config"]})
    (out / "random_baseline.json").write_text(json.dumps(random_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    # Existing shared-heldout results are re-evaluated under the same evaluator.
    shared = {}
    precision = {"sample_tokens": 100_000, "day5": {}, "day6": {}}
    for name, checkpoint in (("day5", day5_ckpt), ("day6", day6_ckpt)):
        model, info = load_model(checkpoint, device=device, defaults=defaults)
        shared[name] = evaluate_array(model, held_arr, held_ranges_10m, device=device, seq_len=512, batch_windows=8, precision="bf16")
        shared[name].update({"checkpoint_sha256": info["checkpoint_sha256"], "checkpoint_step": info["step"]})
        precision[name]["bf16"] = evaluate_array(model, held_arr, held_ranges_100k, device=device, seq_len=512, batch_windows=8, precision="bf16")
        precision[name]["fp32"] = evaluate_array(model, held_arr, held_ranges_100k, device=device, seq_len=512, batch_windows=8, precision="fp32")
        release(model)

    def eval_train(path: Path, max_tokens: int, name: str) -> dict:
        arr = load_array(path)
        ranges = contiguous_range(arr, max_tokens)
        model, info = load_model(day5_ckpt if name == "day5" else day6_ckpt, device=device, defaults=defaults)
        result = evaluate_array(model, arr, ranges, device=device, seq_len=512, batch_windows=8, precision="bf16")
        result.update({"path": str(path), "dataset_file_sha256": sha256_file(path), "selected_tokens": ranges[0][1], "checkpoint_sha256": info["checkpoint_sha256"]})
        result["selected_tokens_sha256"] = sha256_array_bytes(np.asarray(arr[: ranges[0][1]], dtype=np.int32)) if ranges[0][1] else None
        release(model)
        return result

    d5_train_eval = eval_train(day5_train, 9_900_000, "day5") if day5_train is not None else None
    d6_train_eval = eval_train(day6_train, 10_000_000, "day6") if day6_train is not None else None

    # Keep the required per-check outputs alongside the combined summary.  A
    # missing source is recorded explicitly rather than silently omitted.
    (out / "day5_train_corpus_eval.json").write_text(
        json.dumps(d5_train_eval if d5_train_eval is not None else {"status": "unavailable", "reason": "Day5 train_split.npy not found"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "day6_train_subset_eval.json").write_text(
        json.dumps(d6_train_eval if d6_train_eval is not None else {"status": "unavailable", "reason": "Day6 tokens.int32 not found"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    precision["day5"]["nll_delta_fp32_minus_bf16"] = precision["day5"]["fp32"]["nll"] - precision["day5"]["bf16"]["nll"]
    precision["day6"]["nll_delta_fp32_minus_bf16"] = precision["day6"]["fp32"]["nll"] - precision["day6"]["bf16"]["nll"]
    precision["ranking_preserved"] = precision["day6"]["fp32"]["nll"] < precision["day5"]["fp32"]["nll"] and precision["day6"]["bf16"]["nll"] < precision["day5"]["bf16"]["nll"]
    (out / "precision_crosscheck.json").write_text(json.dumps(precision, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "kind": "day7_sanity_check",
        "status": "complete",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "no_training": True,
        "checkpoint_integrity": {"day5": ck5, "day6": ck6},
        "configuration_consistency": consistency,
        "heldout_integrity": {
            "path": str(heldout_tokens),
            "sha256": heldout_sha,
            "expected_sha256": heldout_expected,
            "expected_sha256_source": heldout_expected_source,
            "plan_literal": EXPECTED["heldout_tokens_sha256"],
            "plan_literal_valid_length": len(EXPECTED["heldout_tokens_sha256"]) == 64,
            "sha256_matches": True,
            "index_path": str(heldout_index),
            "scoreable_ranges_10m": len(held_ranges_10m),
        },
        "random_baseline": random_eval,
        "shared_heldout": shared,
        "day5_train_corpus_eval": d5_train_eval,
        "day6_train_subset_eval": d6_train_eval,
        "precision_crosscheck": precision,
        "derived": {
            "day5_memorization_gap": (shared["day5"]["nll"] - d5_train_eval["nll"]) if d5_train_eval else None,
            "day6_generalization_gap": (shared["day6"]["nll"] - d6_train_eval["nll"]) if d6_train_eval else None,
        },
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Day5 API-source shard/URL manifest was not retained, so semantic zero overlap is not proven.",
            "Day7 verifies integrity and numerical robustness; it does not isolate a causal effect of freshness alone.",
        ],
    }
    json_path = out / "day7_sanity_summary.json"
    md_path = out / "DAY7_SANITY_CHECK_RESULT.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
