# Day 7 Sanity Check（合理性检查）结果

> 执行日期：2026-09-13  
> 状态：**已完成（complete）**  
> 说明：本阶段是只读验证，不重新训练模型，不修改模型权重。

## 1. 一句话结论

Day 7 的 6 项检查全部通过：两个 checkpoint（检查点：训练中保存的模型快照）可以完整加载；模型配置一致；随机模型基线正常；Day 5 和 Day 6 都在同一份共享留出集上完成了约 1,000 万令牌评估；Day 6 的训练子集也完成了真实评估；BF16（16 位脑浮点数）与 FP32（32 位浮点数）的结果差异很小，模型排名保持不变。

这证明了 Day 6.6 的巨大差异不是简单的“文件加载失败”或“评估脚本崩了”。它仍然**不能单独证明数据新鲜度的因果效果**，详见第 8 节限制。

## 2. 本次执行概况

| 项目 | 结果 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| CUDA（GPU 计算平台） | 已启用 |
| 实际 GPU 计算时间 | 约 287 秒，约 4.8 分钟 |
| 是否训练新模型 | 否 |
| 是否执行反向传播 | 否 |
| 是否更新优化器/权重 | 否 |
| 输出状态 | `complete` |

## 3. Checkpoint Integrity（检查点完整性）

两个模型都通过了 SHA-256（文件指纹）校验、严格权重加载和参数量校验。

| 模型 | 训练步数 | 参数量 | SHA-256 |
|---|---:|---:|---|
| Day 5 | 15,259 | 336,118,784 | `2c49925b61e46c533026c201dba02bfe845071e7c453964fcc33f508a24c36f3` |
| Day 6 | 15,259 | 336,118,784 | `3d1250567b0d3f311584b15a3ac1d73539cdd52b0baebfd7b290e1f7fdf21aca` |

- `missing_keys`（缺少的权重键）：Day 5、Day 6 均为空。
- `unexpected_keys`（多出来的权重键）：Day 5、Day 6 均为空。
- 两个 checkpoint 都通过了 `strict=True`（严格模式）加载。

输入文件：

```text
/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt
/root/day6_checkpoint_store/day6_336m_500m_fresh_step_15259.pt
```

## 4. Configuration Consistency（配置一致性）

Day 5 和 Day 6 使用完全相同的模型配置，避免把结构差异误当成数据差异。

| 配置项 | 值 |
|---|---:|
| 隐藏维度 `dim` | 1024 |
| Transformer 层数 | 24 |
| Query Head（查询头） | 16 |
| KV Head（键值头） | 4 |
| 上下文长度 | 512 |
| 词表大小 | 32,000 |
| EOS（序列结束标记）编号 | 2 |
| Tokenizer（分词器） | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 参数量 | 336,118,784 |

结论：`day5_day6_model_config_equal = true`。

## 5. Random Baseline（随机模型基线）

使用相同结构、随机种子 `20260913`、相同评估协议，在共享留出集的前 500,000 个可评分令牌上测试：

| 指标 | 结果 |
|---|---:|
| NLL（负对数似然，越低越好） | 约 10.574821 |
| Perplexity（困惑度，越低越好） | 约 3.91 万 |
| 可评分令牌 | 500,000 |
| 随机种子 | 20260913 |

随机模型的损失显著高于两个已训练模型，说明评估脚本确实对模型能力敏感，而不是无论输入什么都输出固定分数。

## 6. 共享留出集正式评估

两个 checkpoint 在同一份 shared held-out set（共享留出集：训练时不使用、专门用来考试的数据）上评估。两边的可评分令牌数和文档数完全一致。

| 模型 | 可评分令牌 | 加权 NLL | Perplexity |
|---|---:|---:|---:|
| Day 5 | 9,992,428 | 8.424211 | 4,556.051 |
| Day 6 | 9,992,428 | 3.039400 | 20.893 |

Day 6 相比 Day 5：

- NLL 低约 **5.384811**。
- 困惑度从约 4,556 降到约 20.9。
- 这个结果与 Day 6.6 的共享评估结果一致。

留出令牌文件：

```text
/root/autodl-tmp/day6_6/heldout_processed/heldout_tokens.int32
```

实际使用的文件 SHA-256（由 `heldout_manifest.json` 提供并再次核对）：

```text
3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f98509a21df9214a6cc750
```

## 7. 训练数据上的 Sanity Check（合理性检查）

### 7.1 Day 5 Memorization Check（记忆检查）

Day 5 在自己使用过的完整 9.9M-token 训练切片上重新评估：

| 指标 | 结果 |
|---|---:|
| 可评分令牌 | 9,899,999 |
| 训练集 NLL | 约 0.065023 |
| 训练集 Perplexity | 约 1.067 |
| 共享留出集 NLL | 8.424211 |
| 记忆差距（留出 NLL − 训练 NLL） | 约 8.359188 |

训练集分数非常低、留出集分数明显高，符合“模型对重复训练语料过度记忆”的现象。

### 7.2 Day 6 Generalization Check（泛化检查）

Day 6 在其保留的新训练流前 10M tokens（1,000 万令牌）上重新评估：

| 指标 | 结果 |
|---|---:|
| 可评分令牌 | 10,000,000 |
| 训练子集 NLL | 约 3.049181 |
| 训练子集 Perplexity | 约 21.098 |
| 共享留出集 NLL | 3.039400 |
| 泛化差距（留出 NLL − 训练 NLL） | 约 −0.009781 |

Day 6 的训练子集与共享留出集分数接近，没有出现 Day 5 那样的巨大记忆差距。这里的“训练子集”是保留训练流的固定前缀，不等同于完整 Day 6 训练语料。

## 8. BF16 / FP32 Numerical Cross-check（数值交叉核验）

在同一份留出数据前 100,000 个可评分令牌上，分别用 BF16 自动混合精度和 FP32 评估：

| 模型 | BF16 NLL | FP32 NLL | FP32 − BF16 |
|---|---:|---:|---:|
| Day 5 | 约 8.424 | 约 8.424 | −0.0001667 |
| Day 6 | 约 3.039 | 约 3.039 | −0.0000198 |

结论：

- 数值差异很小。
- Day 6 优于 Day 5 的排序在 BF16 和 FP32 下都保持不变。
- 因此结果不太可能是 BF16 数值误差单独造成的。

## 9. 这次检查排除了什么、没有排除什么

### 已经排除或大幅降低可能性

1. checkpoint 文件被替换或损坏。
2. Day 5 和 Day 6 参数量、层数、注意力头配置不一致。
3. 评估脚本对两个模型使用了不同协议。
4. 随机初始化模型也能得到相同的“好成绩”。
5. BF16 数值误差改变了模型排名。
6. Day 6 训练子集评估缺失或被错误写成 `None`。

### 仍然不能声称

1. 不能仅凭 Day 7 证明“数据新鲜度必然造成提升”。两个模型的训练历史仍不同。
2. 不能声称 Day 5 与留出集在语义层面绝对零重叠。Day 5 当时通过 API（程序接口）读取数据，没有保留完整的来源分片/URL 清单。
3. 不能把这次实验称作大规模工业预训练；它是 336M 模型、固定令牌预算下的受控 sanity check。

## 10. 生成的机器可读结果

远端结果目录：

```text
/root/autodl-tmp/day7_sanity/reports/
```

其中包含：

```text
random_baseline.json
day5_train_corpus_eval.json
day6_train_subset_eval.json
precision_crosscheck.json
day7_sanity_summary.json
DAY7_SANITY_CHECK_RESULT.md
```

结果归档：

```text
/root/autodl-tmp/day7_sanity/day7_sanity_results.tgz
```

本地复现实验代码：

```text
pretrainlab_x/evaluation/day7_sanity.py
```

## 11. 面试中的准确说法

可以这样说：

> 我为 Day 6.6 的共享留出评估增加了一个只读 sanity-check 阶段。首先对两个 336M checkpoint 做 SHA-256、严格加载和配置一致性校验；然后加入随机模型基线、训练集/训练子集评估，以及 100K token 的 BF16/FP32 数值交叉核验。结果显示 Day 5 在自身训练切片上接近记忆，而 Day 6 的训练子集和共享留出集分数接近；BF16 与 FP32 的差异很小且排名一致。这个结果支持评估链路和数值稳定性，但不单独构成数据新鲜度的因果证明。

## 12. 最终状态

- Day 7：**完成**。
- 新训练：**没有**。
- 权重修改：**没有**。
- 6 项检查：**通过**。
- 远端 GPU 实例：已关机，后续 GPU 计算计费停止。

## 13. Day7 补丁说明

Day7 补丁重新从原始文件计算并交叉核对 SHA-256（文件指纹）。早期报告中
留出令牌哈希曾少一个字符；Day6 checkpoint 哈希曾发生相邻字符转置。
本次仅修复指纹和文档引用，没有改动模型权重、训练历史、评估指标或评估输出。
权威值见 `report/CANONICAL_HASH_MANIFEST.json`。
