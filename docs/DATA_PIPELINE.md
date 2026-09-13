# Data pipeline and provenance

The real-data path uses `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, and the openly downloadable TinyLlama tokenizer. The retained Day 4 slice contains 10,000,000 tokens (9.9M train / 0.1M validation) from 8,479 observed documents, with one exact duplicate removed. Day 6 uses a separate shard and records 505,007,770 training tokens plus 5,000,809 validation tokens after document-level splitting.

The public release intentionally excludes the raw Parquet source, `tokens.npy`, `tokens.int32`, and private held-out arrays. Their sizes and hashes remain in the human-readable reports and small summary JSON. This keeps the repository reviewable and avoids republishing a large third-party corpus.

Day 5's 500M-token run repeatedly sampled a 9.9M-token slice. Day 6 consumed a fresh token stream without replay. Because their histories are not identical, the shared held-out gap is evidence for a difference between these two training recipes, not a causal estimate of “freshness alone.”
