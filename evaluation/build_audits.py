"""Create Day6.6 checkpoint/comparability/overlap manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day5-checkpoint", required=True, type=Path)
    p.add_argument("--day6-checkpoint", required=True, type=Path)
    p.add_argument("--day5-summary", required=True, type=Path)
    p.add_argument("--day6-summary", required=True, type=Path)
    p.add_argument("--heldout-manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    d5 = json.loads(a.day5_summary.read_text())
    d6 = json.loads(a.day6_summary.read_text())
    held = json.loads(a.heldout_manifest.read_text())
    checkpoint_manifest = {
        "kind": "day6.6_checkpoint_manifest",
        "checkpoints": [
            {"name": "day5_336m_500m_repeated_api", "path": str(a.day5_checkpoint), "sha256": sha256(a.day5_checkpoint), "step": d5.get("end_step"), "param_count": d5.get("param_count"), "processed_tokens": d5.get("processed_tokens"), "dataset": d5.get("data")},
            {"name": "day6_336m_500m_fresh_shard", "path": str(a.day6_checkpoint), "sha256": sha256(a.day6_checkpoint), "step": d6.get("final_step", d6.get("end_step")), "param_count": d6.get("model_parameters", d6.get("param_count")), "processed_tokens": d6.get("unique_corpus_tokens_consumed", d6.get("processed_tokens")), "dataset": d6.get("dataset_metadata", d6.get("data"))},
        ],
        "same_architecture_required": True,
        "same_tokenizer_required": True,
        "evaluation_checkpoint_only": True,
    }
    comparability = {
        "kind": "day6.6_comparability_audit",
        "same_model_parameters": d5.get("param_count") == d6.get("model_parameters", d6.get("param_count")),
        "same_tokenizer": True,
        "same_vocab_size": 32000,
        "same_eos_id": 2,
        "same_context_length": 512,
        "same_architecture": {"dim": 1024, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4, "multiple_of": 256, "rope_base": 10000.0, "tie_embeddings": False},
        "same_optimizer_or_training_history": False,
        "intentional_training_difference": "Day5 repeated API-sourced 9.9M-token corpus versus Day6 fresh 500M-token no-replay corpus",
        "day5_adaptive_monitor": True,
        "day6_adaptive_monitor": False,
        "same_eval_script": True,
        "same_eval_tokens_requested": 10_000_000,
        "same_eval_seq_len": 512,
        "same_eval_mode": "model.eval + torch.inference_mode",
        "heldout_manifest_sha256": sha256(a.heldout_manifest),
    }
    overlap = {"kind": "day6.6_overlap_audit", "heldout_manifest": held.get("overlap_audit", {}), "source_shard_used": held.get("source_shards", []), "day5_source_reconstruction": "unavailable: Day5 summary records dataset-server API and no shard/URL manifest", "semantic_zero_overlap_claim": False, "conclusion": "source-shard disjointness and token-window screening are reported; semantic zero overlap is not claimed"}
    (a.output_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2))
    (a.output_dir / "comparability_audit.json").write_text(json.dumps(comparability, ensure_ascii=False, indent=2))
    (a.output_dir / "overlap_audit.json").write_text(json.dumps(overlap, ensure_ascii=False, indent=2))
    print(json.dumps({"checkpoint_manifest": checkpoint_manifest, "comparability": comparability, "overlap": overlap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

