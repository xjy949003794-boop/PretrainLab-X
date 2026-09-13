# Day 7 Sanity Check Result

本报告只做只读验证，不训练新模型，不执行反向传播，也不更新优化器。

## A. Checkpoint Integrity

- **Day5 SHA256**: `2c49925b61e46c533026c201dba02bfe845071e7c453964fcc33f508a24c36f3`
- **Day6 SHA256**: `3d1250567b0d3f311584b15a3ac1d73539cdd52b0baebfd7b290e1f7fdf21aca`
- **missing_keys**: `Day5=[]; Day6=[]`
- **unexpected_keys**: `Day5=[]; Day6=[]`
- **parameter_count**: `336118784`

## B. Evaluation Consistency

- **tokenizer**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **vocab_size**: `32000`
- **eos_id**: `2`
- **context_length**: `512`
- **heldout_sha256**: `3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f9f8509a21df9214a6c750`

## C. Random Baseline

- **random_nll**: `10.57482137109375`
- **random_perplexity**: `39136.913155554816`
- **scored_tokens**: `500000`
- **seed**: `20260913`

## D. Day5 Memorization Check

- **day5_train_nll**: `0.065023332899258`
- **day5_shared_heldout_nll**: `8.424211414250058`
- **memorization_gap**: `8.3591880813508`

## E. Day6 Generalization Check

- **day6_train_subset_nll**: `3.0491799106445314`
- **day6_shared_heldout_nll**: `3.0394002915912224`
- **generalization_gap**: `-0.009779619053309041`

## F. Numerical Cross-check

- **Day5 BF16**: `8.324128438110352`
- **Day5 FP32**: `8.32396173095703`
- **Day6 BF16**: `2.965548766784668`
- **Day6 FP32**: `2.965439006347656`
- **ranking_preserved**: `True`

## G. Scope and limitations

- Random baseline uses the first 500,000 scoreable held-out tokens.
- Day5 uses the complete retained 9.9M-token train split when available.
- Day6 uses a fixed 10M-token prefix of its retained fresh train stream when available.
- This verifies loading, protocol, and numerical robustness; it does not prove a causal effect of data freshness alone.
- The Day6.6 overlap audit still cannot claim semantic zero overlap because Day5's API source lacked a retained shard/URL manifest.
