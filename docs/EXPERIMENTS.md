# Experiment index

| Stage | Configuration/evidence | What it demonstrates |
|---|---|---|
| Strict Day 2–3 | `results/STRICT_DAY2_DAY3_RESULT.md` | modern architecture, 500-step trainer, high-LR stress, resume |
| Day 4 | `experiments/benchmark_300m.yaml`, `benchmark_700m.yaml`, `fsdp.yaml` | 336M/653M forward-backward path and `world_size=1` FSDP plumbing |
| Day 5 | `experiments/day5_336m_500m.yaml`, `results/DAY5_EXECUTION_RESULT.md` | 336M bounded long run, replayed 9.9M slice, offline W&B |
| Day 5 memory | `day5_653m_memory_off/on.yaml` | activation checkpointing on/off comparison |
| Day 6 | `experiments/day6_336m_500m_fresh.yaml`, `results/DAY6_EXECUTION_RESULT.md` | fresh-data token cursor and document-disjoint validation |
| Day 6.6 | `results/DAY6_6_SHARED_EVAL_RESULT_CN.md` | paired shared held-out evaluation and bootstrap CI |
| Day 7 | `results/DAY7_SANITY_CHECK_RESULT_CN.md` | load/config/random-baseline/precision sanity checks |
| Day 7 patch | `results/DAY7_PATCH_RESULT_CN.md`, `audit/` | hash audit and documentation repair |

Short benchmarks are named as benchmarks. Long-run configurations preserve the intended target budgets but were not falsely reported as completed when resources were insufficient.
