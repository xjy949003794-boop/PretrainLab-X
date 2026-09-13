# Reproducibility guide

## Reproduction levels

### Level 1: CPU smoke test (minutes)

This checks imports, configuration parsing, model forward/backward and report writing. It does not reproduce the reported losses.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py --config configs/debug.yaml
python train.py --config configs/debug.yaml
python verify_run.py --run-dir .
```

### Level 2: single-GPU architecture benchmark (minutes)

Provide a local `data/fineweb_10m/` directory containing `tokens.npy`, `train_split.npy`, `val_split.npy`, and `metadata.json` with the tokenizer metadata. Then run the short, explicitly named benchmarks:

```bash
PYTHONPATH=. WANDB_MODE=offline python train_day4.py --config experiments/benchmark_300m.yaml
PYTHONPATH=. WANDB_MODE=offline python train_day4.py --config experiments/benchmark_700m.yaml
PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=1 \
  distributed/run_fsdp.py --config experiments/fsdp.yaml
```

These reproduce the engineering path, not the long-run numbers.

### Level 3: long-run evidence (hours)

The Day 5/Day 6 configurations are preserved in `experiments/`. They require the private token arrays and enough checkpoint storage. Run only after checking the disk budget and use `WANDB_MODE=offline` on an air-gapped machine. Keep the resulting checkpoints outside the public repository.

## Determinism checklist

- Keep the recorded seed (`1337` for the main runs).
- Keep the exact model dimensions, tokenizer vocabulary and EOS id in the metadata.
- Keep document-boundary train/validation splitting for fresh-data experiments.
- Record the GPU name, PyTorch/CUDA versions, precision, batch size, accumulation and sequence length.
- Preserve JSONL metrics and the final summary before shutting down a rented GPU.

## Evidence versus teaching code

The reports in `results/` are the authoritative record of what was actually run. The commands above are reproducibility entry points. Missing private inputs, checkpoint files and raw logs are intentional release boundaries, not silently replaced by synthetic numbers.
