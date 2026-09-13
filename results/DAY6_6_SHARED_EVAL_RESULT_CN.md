# Day 6.6（第六天第六阶段）——Shared Held-out Evaluation（共享留出评估：让两个已经训练好的模型在同一套“期末试卷”上答题）

> 这是英文 PDF 的中文版本，可直接上传给 ChatGPT。代码文件名、路径和命令保留原样，方便复现；正文中的英文术语都紧跟中文含义和生活化解释。

## 1. 最终结论

本阶段已经完成，而且**没有重新训练模型**。

我们拿现成的 Day 5（第五天：重复语料训练得到的模型）和 Day 6（第六天：新分片语料训练得到的模型）两个 336M checkpoint（检查点：训练过程中保存下来的模型快照），放在同一份 Shared Held-out Set（共享留出集：训练时没有拿来学习、专门用来考试的数据）上评估。

Day 6 的结果明显更好：

- Day 5 的加权 NLL（Negative Log-Likelihood，负对数似然：模型给正确答案的概率越低，分数越高）为 **8.424211**。
- Day 6 的加权 NLL（Negative Log-Likelihood，负对数似然：模型给正确答案的概率越低，分数越高）为 **3.039400**。
- Day 5 的 Perplexity（困惑度：模型像在多少个候选答案之间犹豫，越低越好）为 **4556.0505**。
- Day 6 的 Perplexity（困惑度：模型像在多少个候选答案之间犹豫，越低越好）为 **20.89271**。

Day 5 减 Day 6 的 NLL（Negative Log-Likelihood，负对数似然：模型给正确答案的概率越低，分数越高）差值为 **5.384811**。10,000 次 Paired Bootstrap（配对自助法：反复抽取文档对来估计结果稳定性）得到的 95% 区间为 **[5.367100, 5.402861]**。

这说明在本次控制实验中，Day 6 模型在同一留出集上表现更好；但不能把它夸大成“只因为数据更新所以一定更好”，因为两个模型的训练历史也不同。

## 2. 实验控制条件

| 项目 | 固定设置 |
|---|---|
| 模型规模 | 336,118,784 个参数；约 3.36 亿参数 |
| 模型结构 | dim（隐藏维度）1024；24 层；16 个 Query Head（查询头：提出问题的注意力分支）；4 个 KV Head（键值头：保存被查询信息的注意力分支）；RMSNorm（均方根归一化：让数值保持稳定）；RoPE（旋转位置编码：告诉模型词语位置）；SwiGLU（门控前馈网络：像阀门一样筛选信息）；上下文长度 512 |
| 分词器 | TinyLlama/TinyLlama-1.1B-Chat-v1.0；Tokenizer（分词器：把文字切成模型能读的整数）；词表 32,000；EOS（End of Sequence，序列结束标记）编号 2 |
| 评估模式 | `model.eval()`（模型评估模式：关闭训练专用行为）；`torch.inference_mode()`（推理模式：只前向计算，不保存反向传播所需的中间结果）；BF16（Brain Float 16，16 位脑浮点数：省显存的数字格式） |
| 损失计算 | Token-weighted causal cross-entropy（按令牌数量加权的因果交叉熵：逐个预测下一个令牌，并按真实令牌数求平均） |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition；CUDA（英伟达 GPU 计算平台） |
| 优化器和反向传播 | 没有使用；本阶段只读 checkpoint，不修改权重 |

两个 checkpoint（检查点：训练过程中保存下来的模型快照）都是第 15,259 步，参数量和结构完全一致。唯一计划中的训练差异是：Day 5 使用重复的 API（Application Programming Interface，程序接口）语料，Day 6 使用新分片语料。

## 3. 共享留出集如何制作

数据集是 FineWeb-Edu（高质量网页教育语料集）的 `sample-10BT` 配置。候选文件是 `013_00000.parquet`：Parquet（列式数据文件：像把表格按列存放，读取大数据更快）。

处理过程：

1. 检查了 8,681 行原始记录。
2. 按同一套清洗和 Tokenizer（分词器）流程保留 8,666 篇完整文档。
3. 得到 10,000,094 个令牌；目标预算是 10,000,000 个令牌。
4. 每篇文档末尾追加 EOS（序列结束标记）令牌，保留文档边界。
5. 候选集内部 Exact Text Deduplication（精确文本去重：完全相同的文章只留一份）删除 0 行。

留出令牌文件的 SHA-256（安全哈希：给文件生成几乎唯一的指纹）是：

```text
3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f98509a21df9214a6cc750
```

为了筛查和 Day 5 的可重建令牌流重复，我们计算了 64-token window（64 令牌窗口：连续截取 64 个令牌）哈希，窗口 stride（步长：每次向前移动多少个令牌）为 16。这个筛查拒绝了 15 篇候选文档。

候选分片与 Day 6 训练分片 `000_00000.parquet` 不同，因此可以报告 Source-shard Disjointness（来源分片不相交：文件级别不是同一个分片）。

**重要边界：**Day 5 当时通过 dataset-server API（数据集服务器接口）读取，没有保存原始分片编号和 URL（网页地址）清单。因此无法证明 Day 5 与留出集在文档语义层面绝对零重叠，也不能声称 URL 层面完全不重叠。本报告只声称已经实际完成的令牌窗口筛查和来源分片不相交。

原始 516 MB 的 Parquet 文件在处理、哈希和令牌流验证完成后已删除；处理后的令牌流、文档索引和审计清单保留在远端数据盘。

## 4. 统一评估方法

两个 checkpoint（检查点：训练过程中保存下来的模型快照）使用完全相同的评估脚本：

```text
pretrainlab_x/evaluation/shared_heldout_eval.py
```

评估时：

- 调用 `model.eval()`（模型评估模式）。
- 使用 `torch.inference_mode()`（推理模式）。
- 不执行 backward（反向传播：根据错误修改模型）和 optimizer step（优化器更新：真正改动权重）。
- 使用 BF16（16 位脑浮点数）自动混合精度。
- 对每个文档计算 NLL（Negative Log-Likelihood，负对数似然：模型给正确答案的概率越低，分数越高），再按全部可评分令牌数做加权平均。
- 文档最后一个令牌没有“下一个真实令牌”可预测，所以每个文档边界会少计一个目标。这就是请求 10,000,000 个令牌、最终可评分 9,992,428 个令牌的原因。

## 5. 正式评估结果

| Checkpoint（检查点） | 可评分令牌 | 文档数 | 加权 NLL | Perplexity（困惑度） | 评估时间 | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| Day 5 repeated API（重复接口语料） | 9,992,428 | 8,666 | 8.4242114093 | 4556.0505025110 | 62.649 秒 | 159,497.94 token/s（每秒令牌数） |
| Day 6 fresh shard（新分片语料） | 9,992,428 | 8,666 | 3.0394002921 | 20.8927099547 | 62.418 秒 | 160,089.71 token/s（每秒令牌数） |

原始 NLL（Negative Log-Likelihood，负对数似然）总和：

- Day 5：84,178,325.96462083。
- Day 6：30,370,988.58282305。

两次评估的峰值 GPU memory（显存占用）均为 2,614.41 MiB（兆二进制字节）。

## 6. 配对 Bootstrap（自助法）

两个结果按照相同的 8,666 个文档编号配对，然后用 seed（随机种子）1337 重采样 10,000 次。每次重采样都保留文档的令牌数量，避免长文档被错误地当成短文档处理。

结果：

```text
Day5 NLL - Day6 NLL = 5.3848111722
95% interval = [5.3670995529, 5.4028614536]
Perplexity difference = 4535.1577952563
Day6 relative perplexity reduction = 99.5414%
```

这里的正差值表示：在这份共享留出集上，Day 6 checkpoint（检查点：训练过程中保存下来的模型快照）的加权 NLL（Negative Log-Likelihood，负对数似然）更低。

## 7. 可复现代码和结果文件

本地代码：

- `evaluation/build_heldout.py`：构建共享留出流，并记录重叠审计。
- `evaluation/shared_heldout_eval.py`：用相同协议评估两个 checkpoint。
- `evaluation/compare_bootstrap.py`：执行配对文档 Bootstrap（自助法）。
- `evaluation/build_audits.py`：生成 checkpoint、可比性和重叠清单。
- `report/day6_6_summary.json`：机器可读取的结果摘要。

远端 AutoDL（GPU 租用平台）结果文件：

```text
/root/autodl-tmp/day6_6/heldout_processed/heldout_tokens.int32
/root/autodl-tmp/day6_6/heldout_processed/heldout_doc_index.jsonl
/root/autodl-tmp/day6_6/heldout_processed/heldout_manifest.json
/root/autodl-tmp/day6_6/reports/day5_formal_10m.json
/root/autodl-tmp/day6_6/reports/day6_formal_10m.json
/root/autodl-tmp/day6_6/reports/paired_bootstrap_10k.json
/root/autodl-tmp/day6_6/reports/checkpoint_manifest.json
/root/autodl-tmp/day6_6/reports/comparability_audit.json
/root/autodl-tmp/day6_6/reports/overlap_audit.json
```

本地两个 checkpoint 的 SHA-256 指纹：

```text
Day5: 2c49925b61e46c533026c201dba02bfe845071e7c453964fcc33f508a24c36f3
Day6: 3d1250567b0d3f311584b15a3ac1d73539cdd52b0baebfd7b290e1f7fdf21aca
```

本阶段不需要在线上传到 W&B（Weights & Biases，实验记录网站：把损失曲线和硬件指标放到网页上）；所有证据都保存在 JSON（结构化文本格式）和 JSONL（每行一个 JSON 对象）文件中。

## 8. 面试时应该怎样准确描述

可以说：

> 我实现了一个共享留出评估协议，固定了模型结构、Tokenizer、上下文长度和评估代码，对 Day 5 的重复语料 checkpoint 与 Day 6 的新分片 checkpoint 做了相同的 next-token（下一个令牌预测）评估。我们在 8,666 个配对文档、约 1,000 万可评分令牌上比较加权 NLL，并用 10,000 次配对 Bootstrap 给出置信区间。Day 6 的 NLL 从 8.424 降到 3.039；同时我保留了数据来源、哈希和重叠审计的边界，没有声称无法证明的语义零重叠。

不要说：

> 我证明了新数据必然导致模型提升，或者证明了两个语料语义上完全不重叠。

## 9. 最终状态

- Day 6.6：已完成。
- 模型：未重新训练，权重未修改。
- 评估：100K smoke test（冒烟测试：先用小样本验证流程）和约 1,000 万令牌正式评估均完成。
- 审计：checkpoint（检查点）哈希、模型可比性、令牌窗口筛查和限制说明均已保存。
- GPU：AutoDL B46 实例已关机，计算计费已停止；数据盘是否继续收费取决于平台存储规则。

## 13. Day7 补丁说明

最终 Day7 哈希审计直接从三个原始文件计算 SHA-256，并用 `sha256sum`、
OpenSSL 和 Python 交叉核对。审计发现早期报告里留出令牌哈希少了一个字符，
Day6 checkpoint 哈希有相邻字符转置。本补丁只修复文件指纹和审计文档，
没有修改模型权重、训练历史、评估指标或评估输出；权威值见
`report/CANONICAL_HASH_MANIFEST.json`。
