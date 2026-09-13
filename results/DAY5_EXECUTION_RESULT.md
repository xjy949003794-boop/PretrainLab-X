# Day 5 执行结果（严格对账）

执行日期：2026-09-13  
远端实例：AutoDL 西北 B 区 B46，1× RTX PRO 6000 96GB。服务器端训练已完成并已关机；AutoDL 页面核对为运行中 0，当前可用余额 ¥53.73、冻结 ¥0.00。

## 1. 本次真正完成的任务

本次只运行了一个严格目标实验，避免重复消耗 GPU：

- 模型：336,118,784 参数的 LLaMA-style decoder-only 模型。
- 数据：FineWeb-Edu `sample-10BT` 的 10M-token 本地切片；保留 9.9M train tokens 和 0.1M validation tokens。
- 数据处理：最短 200 字符、最长 200,000 字符、控制字符替换、精确文本去重；8,479 篇文档进入处理，8,478 篇保留，去重 1 篇。
- Tokenizer：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，词表 32,000，EOS id 2。
- 训练目标：15,259 个 optimizer steps（优化器更新步），每步实际 token 数为 `8 × 8 × 512 = 32,768`，因此总量为 `15,259 × 32,768 = 500,006,912` tokens。
- 配置：batch size 8、gradient accumulation 8、sequence length 512、BF16、gradient checkpointing 开启、warmup 1,000 步、cosine learning-rate schedule、最终检查点保存。
- 注意：500M tokens 大于本地 10M-token 切片，因此数据在训练过程中会被重复采样；这不是覆盖 500M 个独立网页 tokens 的大规模语料预训练。

## 2. 实测结果

| 指标 | 实测值 |
|---|---:|
| 完成步数 | 15,259 / 15,259 |
| 处理 tokens | 500,006,912 |
| 最终 train loss | 0.069339 |
| 最终 validation loss | 未在该长跑摘要中记录 |
| 总耗时 | 12,673.95 秒（约 3 小时 31 分） |
| 平均吞吐 | 39,424.75 tokens/s |
| 峰值显存 | 7,308.97 MB |
| 精度 | BF16 |
| 梯度累积 | 8 |
| 最终日志健康状态 | ok |
| 平均 GPU 利用率 | 77.24% |

训练健康监控在部分采样点报告过 `low_gpu_utilization`，但没有 NaN、进程退出或显存错误；最终进程正常退出。摘要中的总体 health status 为 `warning`，原因仅为这些低利用率采样点，不能把它包装成“全程无告警”。

## 3. 产物位置

- 逐步 JSONL 指标：`report/day5_336m_500m_metrics.jsonl`（15,259 行）。
- 训练摘要：`report/day5_336m_500m_summary.json`。
- PDF 总结：`report/PretrainLab-X_Final_Report.pdf`。
- W&B 离线运行：`wandb/offline-run-20260913_010008-c7kbwgbc`，run name `day5-336m-500m`，run id `c7kbwgbc`。
- 远端最终 checkpoint：`/root/day5_checkpoint_store/day5_336m_500m_step_15259.pt`，约 3.8GB；为节省下载时间和本地空间，本地只保留其远端路径记录，没有下载该大文件。

W&B 本次是 offline 模式；JSONL 是同一批指标的可审计副本。没有凭离线文件推断在线 URL，也没有声称已经在线同步。

## 4. 费用与资源边界

按 B46 页面显示的 ¥6.98/小时和训练耗时估算，GPU 训练时长成本约为 `3.5205 × 6.98 = ¥24.57`；平台最终账单可能按平台计费粒度略有差异，应以 AutoDL 账单为准。只使用了 1 张卡，没有重新启动第二张卡，没有重跑已有的 300M/653M/吞吐实验。

## 5. 必须诚实说明的范围

这次完成的是一个 **336M 参数、约 500M tokens 的单卡长程训练验证**，用于证明数据管线、训练引擎、监控、检查点和 W&B 离线记录能协同工作。它不是 3B/5B tokens 的完整 Day 4 大规模预训练，也不是在 500M 个独立文档 tokens 上训练，更不是达到生产模型能力的声明。此前的 336M 短跑、653M checkpoint on/off、吞吐扫描和健康监控结果仍属于受控小实验；它们与本次长跑一起构成可复现的工程验证套件。
