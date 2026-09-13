<div align="center">

# PretrainLab-X

### A production-style LLaMA pretraining stack with controlled data experiments and auditable evaluation

**336M long-run pretraining · ~500M processed token positions · FineWeb-Edu · fault injection · shared held-out evaluation · reproducibility audit**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![Precision](https://img.shields.io/badge/Precision-BF16-111827)
![Model](https://img.shields.io/badge/Model-LLaMA--style-111827)
![Data](https://img.shields.io/badge/Data-FineWeb--Edu-F2C94C)

**English** · [简体中文](README_CN.md) · [Technical Report](results/pdf/PretrainLab-X_Day8_Technical_Report.pdf) · [Reproducibility](REPRODUCIBILITY.md) · [Canonical Metrics](results/final_metrics.json)

</div>

---

## At a glance

| | |
|---|---:|
| **Completed long-run model** | 336,118,784 parameters |
| **Largest model path validated** | 653,477,120 parameters |
| **Long-run training budget** | 15,259 optimizer steps |
| **Processed token positions** | 500,006,912 |
| **Non-replayed training corpus** | 505,007,770 train tokens |
| **Shared held-out evaluation** | 8,666 documents / 9,992,428 scored tokens |
| **Held-out NLL** | 8.424 → **3.039** |
| **Paired NLL gap** | 5.3848 |
| **95% paired-bootstrap interval** | [5.3671, 5.4029] |
| **Primary long-run hardware** | 1 × NVIDIA RTX PRO 6000 96GB |

<p align="center">
  <img src="figures/shared_heldout_nll.png" width="820" alt="Shared held-out evaluation">
</p>

## Project arc

PretrainLab-X covers the core path of a foundation-model pretraining project: **model architecture, data pipelines, training infrastructure, stability diagnostics, checkpoint recovery, scale-up validation, long-running pretraining, controlled evaluation, statistical testing, and artifact auditing**. The stack was progressively scaled from small validation runs to 336M / 653M configurations, culminating in two 336M runs of 15,259 optimizer steps and roughly 500M processed token positions each.

The first 336M run exposed a more important issue than whether training loss continued to decrease: **roughly 500M token positions were being consumed from an underlying training slice of only 9.9M tokens.** The corpus was therefore being replayed many times. That observation shifted the next experiment toward data coverage. The pipeline was rebuilt around approximately **505M FineWeb-Edu training tokens**, and the 336M run was repeated at essentially the same model scale and processed-token budget without dataset replay.

The two resulting checkpoints were then evaluated under a common protocol rather than compared through their training losses. Model configuration, tokenizer, context length, evaluation code, and held-out corpus were fixed across both runs. Next-token evaluation covered **8,666 documents and 9,992,428 scored tokens**, followed by **10,000 paired bootstrap resamples**. The resulting weighted NLL was **8.424 for Day 5 and 3.039 for Day 6**.

That gap became the next engineering question: whether it reflected generalization behavior or an artifact of checkpoint integrity, evaluation code, numerical precision, data overlap, or training-set memorization. The project therefore added strict checkpoint loading, configuration matching, a random-init baseline, Day 5 training-corpus evaluation, Day 6 training-subset evaluation, BF16 / FP32 cross-checks, held-out overlap screening, and multi-implementation SHA-256 verification.

The result is not just a training run, but an end-to-end experimental chain:

**train → diagnose → redesign → rerun → evaluate → challenge the result → audit the evidence**

The project progression can be summarized as:

```text
model + trainer
      ↓
real-data pipeline
      ↓
scale-up validation
      ↓
336M repeated-corpus long run
      ↓
data-replay problem identified
      ↓
~505M-token non-replayed corpus rebuild
      ↓
336M long run repeated at matched token budget
      ↓
shared held-out exam
      ↓
bootstrap + sanity checks + audit
```

## System implementation

### Model
- Decoder-only Transformer with **RMSNorm**, **RoPE**, **SwiGLU**, and **grouped-query attention (GQA)**.
- 16 query heads / 4 KV heads in the main 336M configuration.
- PyTorch scaled-dot-product attention through the `sdpa_auto` backend.
- Activation checkpointing support for larger configurations.
- 336M main configuration: `dim=1024`, `24` layers, `16` query heads, `4` KV heads, context length `512`.

### Training engine
- **BF16** mixed-precision training.
- Gradient accumulation and gradient clipping.
- Warmup + cosine learning-rate schedule.
- Stateful checkpoints containing model, optimizer, scheduler, and RNG state.
- Checkpoint resume validated from step 400 → 500.
- JSONL telemetry for loss, learning rate, pre-clip gradient norm, GPU memory/utilization, and throughput.
- Observability is separated from intervention: the monitor records and flags abnormal behavior but does not silently rewrite LR, batch size, or optimizer state.

### Data and evaluation
- FineWeb-Edu ingestion, filtering, exact-text deduplication, document-level splitting, tokenization, and packing.
- Tokenizer provenance and token-file SHA-256.
- Shared held-out next-token evaluation for frozen checkpoints.
- Token-weighted NLL and perplexity.
- Document-level paired bootstrap.
- Random-init baseline, precision cross-check, overlap audit, and checkpoint/configuration integrity checks.

```mermaid
flowchart LR
    A[FineWeb-Edu] --> B[Filter + clean + deduplicate]
    B --> C[Document-level split]
    C --> D[Tokenize + pack]
    D --> E[LLaMA-style decoder]
    E --> F[BF16 trainer]
    F --> G[Checkpoint + telemetry]
    G --> H[Shared held-out evaluation]
    H --> I[Bootstrap + sanity checks]
    I --> J[SHA-256 audit]
```

## Data engineering: from a 10M-token slice to a 510M-token corpus

The data pipeline was expanded in two stages.

### Stage 1 — real-data integration

The first real-data pipeline used FineWeb-Edu `sample-10BT` and produced a 10M-token corpus:

| Item | Value |
|---|---:|
| Documents inspected | 8,479 |
| Documents retained | 8,478 |
| Exact duplicates removed | 1 |
| Cleaned characters | 39,822,471 |
| Total tokens | 10,000,000 |
| Train / validation | 9,900,000 / 100,000 |
| Tokenizer | TinyLlama tokenizer, vocab 32,000, EOS 2 |

This stage established the real-data preprocessing path and later became the underlying slice used by the Day 5 repeated-corpus run.

### Stage 2 — non-replayed long-run corpus

For Day 6, the pipeline was rebuilt from a 2.15 GB FineWeb-Edu parquet source with the train/validation split performed at document boundaries before packing:

| Item | Value |
|---|---:|
| Train documents | 427,260 |
| Validation documents | 4,243 |
| Exact duplicates removed | 395 |
| Train tokens | 505,007,770 |
| Validation tokens | 5,000,809 |
| Total tokens | 510,008,579 |
| Packed token file | 2,040,034,316 bytes (`int32`) |

The raw parquet was integrity-checked before preprocessing. The packed token stream and metadata were retained with SHA-256 provenance; the raw source file was removed after verification to avoid unnecessary storage use.

## Scale-up and distributed-path validation

Before the long runs, larger model paths were exercised on the real FineWeb-Edu pipeline.

| Configuration | Parameters | Steps | Precision | Outcome |
|---|---:|---:|---|---|
| 336M benchmark | 336,118,784 | 8 | BF16 | forward/backward/update path passed |
| 653M benchmark | 653,477,120 | 4 | BF16 | larger model path passed |
| FSDP path | 336,118,784 | 16 | BF16 | init/wrap/forward/backward/optimizer path passed |

The FSDP run used `world_size=1`; PyTorch therefore operated in `NO_SHARD`. This validates the code path, not multi-GPU sharding or communication.

A scaling report and machine-readable summaries were generated from these runs.

## Long-run pretraining

Both formal long runs use the same 336,118,784-parameter architecture and the same 15,259-step / 500,006,912-position budget.

| | Day 5 — repeated corpus | Day 6 — non-replayed corpus |
|---|---:|---:|
| Model | 336,118,784 params | 336,118,784 params |
| Steps | 15,259 | 15,259 |
| Tokens / optimizer step | 32,768 | 32,768 |
| Processed token positions | 500,006,912 | 500,006,912 |
| Underlying train corpus | 9.9M tokens | 505,007,770 tokens |
| Data regime | slice replayed repeatedly | consumed without dataset replay |
| Precision | BF16 | BF16 |
| Wall time | ~3 h 31 m | ~3.55 h |
| Checkpoint | ~4.03 GB | ~4.03 GB |

Day 5 averaged **39,424.75 tokens/s**, **77.24% GPU utilization**, and **7.3 GB peak allocated training memory** in its recorded summary. The run completed normally; sampled low-utilization warnings were retained rather than hidden.

Day 6 first passed a **100-step smoke run**, then completed the formal run. Its best recorded validation loss was **3.14276 at step 15,000**, and the release includes the long-run JSONL trajectory.

<p align="center">
  <img src="figures/day6_training_curve.png" width="900" alt="Day 6 non-replayed training curve">
</p>

## Shared held-out evaluation

The Day 5 and Day 6 checkpoints were frozen and evaluated with the same model-loading and next-token scoring path. Both were at step 15,259 and use the same architecture, tokenizer, vocabulary, EOS ID, and context length.

### Held-out construction and overlap controls

The shared evaluation set was built from FineWeb-Edu `sample-10BT`, candidate shard `013_00000.parquet`:

- 8,681 candidate rows inspected.
- 8,666 complete documents retained.
- 10,000,094 tokens produced before next-token boundary masking.
- Same cleaning and tokenizer path as the training data.
- Exact-text deduplication inside the candidate set.
- 64-token-window SHA-256 screening at stride 16 against the reconstructable Day 5 9.9M-token stream.
- **15 candidate documents rejected** by the exact token-window screen.
- Candidate shard differs from the Day 6 training shard (`000_00000.parquet`).

The Day 5 data path did not preserve original shard/URL provenance, so this is intentionally reported as **token-window screening + source-shard separation**, not as proof of semantic or URL-level zero overlap.

### Identical evaluation protocol

For both checkpoints:

- `model.eval()`
- `torch.inference_mode()`
- BF16 autocast
- no backward pass
- no optimizer update
- token-weighted causal cross-entropy
- identical scoring code and held-out stream

The requested 10M-token budget yields **9,992,428 scored tokens** because each complete document loses one unavailable next-token target at its boundary.

| Checkpoint | Documents | Scored tokens | Weighted NLL | Perplexity |
|---|---:|---:|---:|---:|
| **Day 5 repeated-corpus** | 8,666 | 9,992,428 | **8.424211** | 4,556.051 |
| **Day 6 non-replayed** | 8,666 | 9,992,428 | **3.039400** | 20.893 |

<p align="center">
  <img src="figures/shared_heldout_nll.png" width="820" alt="Shared held-out evaluation">
</p>

A paired bootstrap resampled the same 8,666 document IDs **10,000 times** (seed `1337`) while preserving document token counts:

- Day 5 − Day 6 weighted NLL: **5.384811**
- 95% percentile interval: **[5.367100, 5.402861]**

The NLL gap is the primary reported statistic. Perplexity is retained as an auxiliary metric because it is exponential in NLL.

## The large gap was treated as suspicious, not as a conclusion

The held-out difference was large enough to justify trying to break the result before trusting it. Day 7 added a read-only sanity-check layer with **no retraining and no weight updates**.

### 1. Checkpoint integrity
Both 336M checkpoints were re-hashed and loaded with `strict=True`.

- `missing_keys = []`
- `unexpected_keys = []`
- parameter count = 336,118,784 for both
- both checkpoints at step 15,259

### 2. Configuration equality
The comparison verified identical:

- hidden size `1024`
- `24` Transformer layers
- `16` query heads
- `4` KV heads
- context length `512`
- vocabulary `32,000`
- EOS ID `2`
- TinyLlama tokenizer

`day5_day6_model_config_equal = true`.

### 3. Random-init baseline
A randomly initialized model with the same architecture and evaluation path was scored on 500,000 held-out targets:

- NLL: **10.574821**
- Perplexity: ~**39,100**

This verifies that the evaluation pipeline is sensitive to model quality rather than returning a nearly fixed score.

### 4. Training-side re-evaluation
The two trained checkpoints were also evaluated back on training-side data:

| Checkpoint | Training-side NLL | Shared held-out NLL | Interpretation |
|---|---:|---:|---|
| **Day 5** | **0.065023** on the full 9.9M-token training slice | **8.424211** | very large train/held-out gap |
| **Day 6** | **3.049181** on a fixed 10M-token training prefix | **3.039400** | training-subset and held-out scores are close |

<p align="center">
  <img src="figures/memorization_generalization.png" width="820" alt="Memorization vs generalization sanity check">
</p>

This is the strongest evidence for the **memorization vs. generalization** interpretation inside this experiment. The Day 6 training subset is a fixed prefix, not the complete 505M-token training corpus.

### 5. BF16 / FP32 numerical cross-check
The first 100,000 scored held-out tokens were re-evaluated in both BF16 and FP32:

| Model | BF16 NLL | FP32 NLL | FP32 − BF16 |
|---|---:|---:|---:|
| Day 5 | ~8.424 | ~8.424 | −0.0001667 |
| Day 6 | ~3.039 | ~3.039 | −0.0000198 |

The ordering is unchanged and the numerical differences are tiny, making BF16 evaluation error an implausible explanation for the observed gap.

### 6. Evaluation-chain consistency
The same shared held-out result was reproduced during the sanity-check stage: Day 5 remained at ~8.424 NLL and Day 6 at ~3.039 NLL.

Together, these checks substantially reduce the likelihood that the headline result is explained by a broken checkpoint, mismatched architecture, random baseline artifact, precision issue, or asymmetric evaluation code.

## Training stability, fault injection, and recovery

The training engine was also exercised under deliberately abnormal optimization conditions rather than only the nominal configuration.

### Normal vs. high-learning-rate stress test

A 30.42M-parameter model was trained for 500 steps under both the normal configuration and an independent high-LR (`3e-3`) stress configuration.

| Run | Final loss | Best loss | Max pre-clip grad norm | Validation loss |
|---|---:|---:|---:|---:|
| **Normal LR** | **0.015530** | 0.014527 | 15.012 | 0.015689 |
| **High LR (`3e-3`)** | **0.044620** | 0.042047 | **19.085** | 0.046229 |

The high-LR run remained numerically finite, but converged to a substantially worse loss and produced larger pre-clipping gradient norms.

The observability path records global gradient norm **before** clipping, while the trainer applies `grad_clip=1.0` before the optimizer update. This keeps the diagnostic signal visible without allowing the same magnitude to propagate directly into the parameter update.

### Deterministic anomaly injection

A separate anomaly demo exercised the diagnostic path with deliberately abnormal traces and reproducibly triggered:

- `loss_spike`
- `gradient_explosion_risk`

This tests the monitor as a diagnostic system rather than using the monitor itself to modify training.

### Recovery run

After the stress configuration, a full 500-step recovery run restored the normal learning rate, warmup, and clipping setup:

- final loss: `0.015530`
- best loss: `0.014527`
- validation loss: `0.015689`

The recovery run returned to the normal baseline.

### Stateful checkpoint resume

Checkpoints were written every 100 steps with model, optimizer, scheduler, and RNG state. A step-400 checkpoint was reloaded and training continued to step 500, reproducing the original step-500 loss of `0.015530`.

This validates stateful training recovery rather than weight-only loading.

<details>
<summary><b>Diagnostic endpoint views</b></summary>

<br>

<p align="center">
  <img src="figures/normal_vs_high_lr_loss.png" width="720" alt="Normal vs high learning-rate endpoint loss">
</p>

<p align="center">
  <img src="figures/normal_vs_high_lr_grad_norm.png" width="720" alt="Normal vs high learning-rate gradient norm">
</p>

</details>

## Reproducibility and artifact audit

The project keeps machine-readable results alongside narrative reports rather than relying on screenshots or hand-copied numbers.

After the sanity checks, the three source artifacts used in the final comparison were re-hashed directly with **three independent implementations**:

- `sha256sum`
- OpenSSL
- Python `hashlib`

All three implementations agreed on all three files:

- Day 5 checkpoint
- Day 6 checkpoint
- shared held-out token stream

That audit caught two earlier documentation transcription errors: one missing character in the held-out hash and one adjacent-character transposition in the Day 6 checkpoint hash. The patch changed **only the recorded fingerprints and documentation**; weights, training history, metrics, and evaluation outputs were not modified. Pre-patch copies were retained for audit comparison.

Canonical evidence is kept in:

- [`results/final_metrics.json`](results/final_metrics.json)
- [`audit/CANONICAL_HASH_MANIFEST.json`](audit/CANONICAL_HASH_MANIFEST.json)
- [`audit/HASH_CROSSCHECK.json`](audit/HASH_CROSSCHECK.json)
- [`results/`](results/)
- [`figures/`](figures/)
- [`results/pdf/PretrainLab-X_Day8_Technical_Report.pdf`](results/pdf/PretrainLab-X_Day8_Technical_Report.pdf)

The release itself also passed component-level checks for RMSNorm, RoPE, SwiGLU, GQA, full-model forward, and training-observability logic, plus a release audit for required files, JSON validity, hash format, forbidden large artifacts, and obvious secret patterns.

## Repository map

```text
model/          LLaMA-style model and attention components
engine/         trainer, scheduler, optimizer and checkpoint logic
data/           preprocessing and tokenizer adapters
distributed/    FSDP entry point and wrapper
evaluation/     held-out evaluation, bootstrap, sanity and audit code
monitor/        JSONL logging, GPU/gradient probes and diagnostics
experiments/    benchmark and long-run configurations/scripts
results/        reports and machine-readable experiment evidence
figures/        release figures generated from checked-in evidence
audit/          canonical hash manifests and cross-checks
docs/           design, data, experiment and evaluation notes
tests/          component-level checks
```

## Reproduce a smoke run

The public release intentionally excludes raw corpora, packed token arrays, and large model checkpoints. A small smoke configuration can be run with prepared local data:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py --config configs/debug.yaml
python train.py --config configs/debug.yaml
python verify_run.py --run-dir .
```

For exact experiment boundaries, data assumptions, and GPU configurations, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Scope

The **336M** configuration is the completed long-run experiment. The **653M** configuration is a short path-validation benchmark, not a completed long run. The FSDP path was exercised with `world_size=1`, so no multi-GPU scaling claim is made. The recorded attention backend is **PyTorch SDPA**, not a third-party `flash-attn` installation.

The Day 5 / Day 6 comparison supports a strong memorization/generalization contrast under this experimental setup. It does not establish a universal causal law that non-replayed or “fresh” data is always superior: the checkpoints have different full training histories, and Day 5 did not retain enough original source provenance to prove semantic- or URL-level zero overlap with the held-out set.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full interpretation boundary.

---

<div align="center">

**PretrainLab-X** · pretraining systems, data pipelines, failure diagnostics, and evidence-driven evaluation

</div>
