# Results archive map

The release keeps the compact, human-facing reports and machine-readable summaries needed to audit the project:

- `results/DAY1_RESULT.md`
- `results/DAY2_DAY3_RESULT.md`, `STRICT_DAY2_DAY3_RESULT.md`, `STRICT_DAY2_DAY3_EXECUTION.md`
- `results/DAY4_EXECUTION_RESULT.md`
- `results/DAY5_EXECUTION_RESULT.md`
- `results/DAY6_EXECUTION_RESULT.md`
- `results/DAY6_6_SHARED_EVAL_RESULT.md` and `_CN.md`
- `results/DAY7_SANITY_CHECK_RESULT_CN.md`
- `results/DAY7_PATCH_RESULT_CN.md`
- `results/machine_readable/` and `results/pdf/`

The discarded layer is deliberate: model checkpoints, raw Parquet/token arrays, W&B binary runs, caches and pre-patch backups stay private. The original files remain outside this public staging directory.
