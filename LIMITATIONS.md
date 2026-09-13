# Limitations and claim boundary

This file is part of the evidence, not an apology. The project is useful precisely because its scope is explicit.

1. **Scale.** The longest demonstrated run is a 336M-parameter single-GPU experiment with a bounded 500,006,912-token budget. Day 4's 3B/5B-token targets and multi-GPU FSDP run were not completed.
2. **Data replay.** Day 5 processes 500M positions by repeatedly sampling a 9.9M-token slice. It must not be described as 500M independent web tokens.
3. **Causal interpretation.** Day 5 and Day 6 have different training histories and data streams. The shared held-out evaluation is controlled evidence, but it does not isolate data freshness as a causal variable.
4. **Day 5 provenance.** Day 5 used dataset-server API rows without a retained URL/shard manifest. Semantic zero overlap with the held-out set is therefore not proven.
5. **FSDP.** The recorded FSDP path has `world_size=1`. It verifies initialization and forward/backward plumbing, not multi-GPU communication or memory sharding.
6. **Attention backend.** Training used PyTorch SDPA (`sdpa_auto`). The third-party `flash-attn` wheel did not finish compiling, so no claim of a direct flash-attn integration is made.
7. **Evaluation.** Day 6's training-subset sanity check is a fixed prefix, not the complete training stream. Validation metrics and NLL are not model capability benchmarks.
8. **W&B.** Offline W&B runs are retained privately. The public release uses JSONL and small summaries as its primary audit trail; no online dashboard URL is inferred from an offline run id.
9. **Reproducibility.** Exact byte-for-byte reruns require the recorded environment, tokenizer, data provenance and private checkpoints. The public package provides code/config/report paths and a bounded CPU smoke test, not the large private inputs.
