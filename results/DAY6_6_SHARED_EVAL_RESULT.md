# Day 6.6 — Shared Held-out Evaluation

**Status: complete (no retraining).**  This stage evaluates the existing Day 5
repeated-corpus 336M checkpoint and the existing Day 6 fresh-corpus 336M
checkpoint on one shared held-out stream.  The model weights and training
history were not changed.

## Scope and controls

| Item | Fixed choice |
|---|---|
| Model | 336,118,784 parameters; dim 1024; 24 layers; 16 query heads; 4 KV heads; SwiGLU/RoPE/RMSNorm; context 512 |
| Tokenizer | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, vocabulary 32,000, EOS id 2 |
| Evaluation | identical script, `model.eval()`, `torch.inference_mode()`, BF16 autocast, no backward pass or optimizer |
| Loss | token-weighted causal cross-entropy (one next-token target is unavailable at each document boundary) |
| Requested budget | 10,000,000 held-out tokens |
| Scored budget | 9,992,428 tokens across 8,666 complete documents; the difference is the document-boundary next-token mask |
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA |

The Day 5 checkpoint is `/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt`.
The Day 6 checkpoint is `/root/day6_checkpoint_store/day6_336m_500m_fresh_step_15259.pt`.
Both are at step 15,259 and have the same parameter count and architecture.

## Shared held-out construction

The candidate was FineWeb-Edu `sample-10BT`, shard `013_00000.parquet`.
After the same cleaning/tokenizer path, the stream contained 10,000,094
tokens from 8,666 documents (8,681 rows inspected).  Candidate-level exact
text deduplication removed 0 rows.  The token-stream SHA-256 is
`3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f98509a21df9214a6cc750`.

The candidate was screened against the reconstructable Day 5 9.9M-token
stream using SHA-256 hashes of 64-token windows at stride 16; 15 candidate
documents were rejected by that exact token-window screen.  The candidate
source shard is different from Day 6's training shard `000_00000.parquet`.
The processed token stream and document index were retained; the original
516-MB parquet file was deleted after processing to reduce billed storage.

**Overlap boundary:** Day 5 was produced through the dataset-server API and
did not retain source-shard/URL identifiers.  Therefore exact Day 5
document/URL reconstruction and semantic zero-overlap cannot be proven.  The
report intentionally claims only the recorded token-window screen and
source-shard disjointness.

## Evaluation results

| Checkpoint | Scored tokens | Documents | Weighted NLL | Perplexity | Eval time | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Day 5 repeated API corpus | 9,992,428 | 8,666 | 8.4242114093 | 4,556.0505025110 | 62.649 s | 159,497.94 tok/s |
| Day 6 fresh shard corpus | 9,992,428 | 8,666 | 3.0394002921 | 20.8927099547 | 62.418 s | 160,089.71 tok/s |

Raw NLL sums were 84,178,325.96462083 (Day 5) and 30,370,988.58282305
(Day 6).  Peak allocated GPU memory was 2,614.41 MiB for each evaluation.

## Paired bootstrap comparison

The same 8,666 document IDs were paired and resampled 10,000 times (seed
1337), preserving each document's token count.  Day 5 minus Day 6 weighted
NLL was **5.3848111722**, with a percentile 95% interval of
**[5.3670995529, 5.4028614536]**.  The corresponding perplexity difference was
4,535.1577952563; Day 6's perplexity was 99.5414% lower relative to Day 5.

This is a controlled held-out comparison, not a causal claim that data
freshness alone explains the gap: the two checkpoints also have different
training histories (Day 5's repeated API stream versus Day 6's 500M-token
fresh stream).

## Artifacts

Implementation in this repository:

- `evaluation/build_heldout.py` — held-out builder and overlap audit.
- `evaluation/shared_heldout_eval.py` — identical checkpoint evaluator.
- `evaluation/compare_bootstrap.py` — paired document bootstrap.
- `evaluation/build_audits.py` — checkpoint/comparability/overlap manifests.

Remote run artifacts (AutoDL, retained on the data disk):

- `/root/autodl-tmp/day6_6/heldout_processed/heldout_tokens.int32`
- `/root/autodl-tmp/day6_6/heldout_processed/heldout_doc_index.jsonl`
- `/root/autodl-tmp/day6_6/heldout_processed/heldout_manifest.json`
- `/root/autodl-tmp/day6_6/reports/day5_formal_10m.json`
- `/root/autodl-tmp/day6_6/reports/day6_formal_10m.json`
- `/root/autodl-tmp/day6_6/reports/paired_bootstrap_10k.json`
- `/root/autodl-tmp/day6_6/reports/checkpoint_manifest.json`
- `/root/autodl-tmp/day6_6/reports/comparability_audit.json`
- `/root/autodl-tmp/day6_6/reports/overlap_audit.json`

No online W&B upload was required for this evaluation; all evidence is in the
JSON/JSONL artifacts above.  The GPU instance is to be released immediately
after the final report is verified.

## Day7 patch note

The final Day7 hash audit recomputed the three source-file SHA-256 values
directly with `sha256sum`, OpenSSL, and Python. It found two early report
transcription errors: the held-out token hash had a missing character, and the
Day6 checkpoint hash had an adjacent-character transposition. This patch only
repairs file fingerprints and audit documentation; model weights, metrics,
training history, and evaluation outputs were not changed. The canonical
values are recorded in `report/CANONICAL_HASH_MANIFEST.json`.
