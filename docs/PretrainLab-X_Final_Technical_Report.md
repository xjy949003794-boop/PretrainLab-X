# PretrainLab-X Final Technical Report

**Date:** 2026-09-14  
**Project type:** auditable, engineering-scale foundation-model pretraining stack

## Abstract

PretrainLab-X connects a LLaMA-style decoder-only model, a BF16 training engine, real-data preprocessing, checkpoint recovery, monitoring, shared held-out evaluation, and hash-based evidence auditing. The longest experiments are bounded single-GPU runs. The central diagnostic compares a 336M model trained on a repeatedly sampled 9.9M-token slice (Day 5) with the same-sized model trained on a fresh token stream (Day 6). On one shared held-out set, weighted NLL is 8.424211 for Day 5 and 3.039400 for Day 6; the paired bootstrap gap is 5.384811 with a 95% interval of [5.367100, 5.402861]. This is controlled evidence from two different training histories, not a causal estimate of data freshness alone.

## 1. Engineering question

Can a small but production-shaped stack make every important pretraining decision inspectable: data provenance, architecture, numerical format, optimizer state, checkpoint recovery, monitoring signals, evaluation protocol and artifact integrity?

## 2. System

The model is a pre-normalized decoder-only Transformer. RMSNorm stabilizes activations, RoPE supplies relative position information, SwiGLU provides a gated feed-forward path, and GQA uses 16 query heads with 4 key/value heads. Causal attention is routed through PyTorch SDPA (`sdpa_auto`). The trainer adds BF16 autocast, accumulation, clipping, warmup + cosine scheduling, atomic checkpoint writes and complete optimizer/scheduler/RNG restoration.

## 3. Data

The real-data pipeline uses FineWeb-Edu `sample-10BT` and the TinyLlama tokenizer. The retained Day 4 slice contains 10,000,000 tokens (9.9M train / 0.1M validation) from 8,479 observed documents, with one exact duplicate removed. Day 6 uses a separate shard and records 505,007,770 training tokens and 5,000,809 validation tokens after document-level splitting. Raw Parquet and token arrays are private and are not republished in this release.

## 4. Experiments and results

### Strict Day 2-3

The 30,419,456-parameter run completed 500 steps with the modern architecture and full trainer instrumentation. The high-learning-rate stress run (`3e-3`) ended at loss 0.044620 versus 0.015530 for the normal-LR run; maximum pre-clip gradient norms were 19.08514 and 15.01231 respectively. Resume from step 400 to 500 was verified. The corpus for this strict run is a documented deterministic byte-level fallback, not FineWeb-Edu.

### Day 4 benchmarks

The 336M benchmark completed 8 steps and the 653M benchmark completed 4 steps on BF16 CUDA. The FSDP entry point completed a 16-step `world_size=1` path check. No 3B/5B-token long run or multi-GPU FSDP claim is made.

### Day 5 and Day 6

Both main runs use 336,118,784 parameters and 15,259 optimizer steps. Day 5 processed 500,006,912 positions by replaying a 9.9M-token slice; its final training loss was 0.069339. Day 6 consumed 500,006,912 unique fresh-stream tokens without replay and reached best validation loss 3.142760 at step 15,000.

### Shared evaluation and sanity checks

The same 8,666-document held-out stream contributes 9,992,428 scored tokens to both models. Day 7 additionally verified strict checkpoint loading, equal model configurations, a random baseline, a fixed Day 6 training prefix, and BF16/FP32 ranking consistency. The hash patch changed documentation references only; it did not alter weights, metrics or training history.

## 5. Figures

![Normal versus high-LR loss](../figures/normal_vs_high_lr_loss.png)

![Normal versus high-LR gradient risk](../figures/normal_vs_high_lr_grad_norm.png)

![Shared held-out NLL](../figures/shared_heldout_nll.png)

![Memorization versus generalization](../figures/memorization_generalization.png)

![Day 6 training curve](../figures/day6_training_curve.png)

The Day 2-3 plots are endpoint views sourced from the strict report because those raw logs are not part of the public staging. The Day 6 curve uses the retrieved raw JSONL log.

## 6. Reproduction and audit

Start with the CPU smoke test in [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md), then use the explicitly named short GPU benchmarks. `scripts/verify_release.py` rejects checkpoints, raw arrays, Parquet, W&B binaries, oversized files and credential-like markers. `audit/` contains the canonical SHA-256 manifest and the independent cross-check record.

## 7. Limitations

This report does not claim industrial model quality, billion-token independent corpora, multi-GPU scaling, third-party FlashAttention integration, or a causal effect of data freshness. See [`LIMITATIONS.md`](../LIMITATIONS.md) for the complete boundary.

## 8. Reusable interview statement

> I built an auditable mini foundation-model pretraining stack. The model uses RMSNorm, RoPE, SwiGLU and GQA; the trainer handles BF16, accumulation, clipping, warmup-cosine scheduling and full-state checkpoint resume. I then compared a replayed 9.9M-token slice with a fresh stream under a shared held-out protocol. The result is a large but carefully bounded NLL gap, and I added sanity checks and SHA-256 audits so I can defend the measurement without claiming a causal or industrial-scale result.
