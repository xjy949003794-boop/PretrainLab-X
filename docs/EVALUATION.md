# Evaluation protocol

Day 6.6 evaluates the Day 5 and Day 6 checkpoints in `model.eval()` and `torch.inference_mode()` on the same 8,666-document held-out stream. It scores 9,992,428 tokens, computes weighted NLL/perplexity, and uses paired bootstrap resampling (10,000 draws, seed 1337) to quantify the document-level difference.

Day 7 verifies that both checkpoints load strictly with no missing or unexpected keys, use equal model configurations, beat a random baseline, and preserve the same ranking under BF16 and FP32 evaluation. It also compares Day 5's training slice with the shared held-out set and Day 6's fixed training prefix with that held-out set.

The evaluation is diagnostic evidence. It is not a downstream benchmark, a human preference test, or a causal claim that one data property alone produced the observed gap.
