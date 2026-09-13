# System design notes

## Data path

The data path is explicit: source rows are filtered, control characters are cleaned, exact duplicate texts are removed, documents are tokenized with a recorded tokenizer, EOS markers are inserted, and train/validation arrays are split at document boundaries. Metadata records source, configuration, counts, vocabulary, seed and content hashes.

## Model path

`model/llama.py` defines a decoder-only stack. Each block is pre-normalized with RMSNorm, applies RoPE inside grouped-query attention, then uses a SwiGLU feed-forward module. `model/attention.py` delegates the causal attention operation to PyTorch SDPA. `n_kv_heads < n_heads` makes the key/value projection cheaper than a full multi-head implementation.

## Training path

`engine/trainer.py` owns the optimizer loop. It accumulates micro-batch losses, runs BF16 autocast on CUDA, clips the pre-clip gradient norm, advances a warmup-cosine scheduler, writes JSONL metrics, and atomically saves model/optimizer/scheduler/RNG/data metadata. Resume restores the same state instead of only loading model weights.

## Evidence path

Every headline number in `results/final_metrics.json` points to a checked-in summary or report. Day 6.6 compares two checkpoints on one shared held-out stream; Day 7 adds random-baseline, strict-load, configuration and BF16/FP32 checks. The Day 7 patch records that hash corrections touched documentation references only, not weights or metrics.
