# Day 6 执行结果：Fresh-data 336M 预训练

执行日期：2026-09-13（Asia/Shanghai）

## 结论

Day 6 已按计划完成：先做 100 步 smoke test（冒烟测试），再做一次单卡 BF16 正式训练。正式训练只消费新语料中的唯一 token，未启用 replay（重复回放），未启用 adaptive monitor（自适应监控）。

这是一项真实语料、真实梯度更新的缩小版预训练工程验证，不应包装成生产规模预训练或完整论文复现。

## 语料与预处理证据

- 来源：`HuggingFaceFW/fineweb-edu`，配置 `sample-10BT`。
- 原始 Parquet 文件大小：`2,152,819,114` bytes；完成完整性校验后才开始预处理。
- 文档级切分：训练集与验证集在文档边界切分，避免同一文档跨集合泄漏。
- 训练 token：`505,007,770`。
- 验证 token：`5,000,809`。
- 总 token：`510,008,579`。
- 文档数：训练 `427,260`，验证 `4,243`。
- 过滤文档：`0`；去重删除：`395`。
- Tokenizer（分词器）：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，词表 `32,000`，EOS id `2`。
- 输出 `tokens.int32` 大小：`2,040,034,316` bytes（与总 token 数 × 4 一致）。
- token 文件 SHA-256：`7d64cc914967823d3f8cae142c76ef24c4472a344b4b89a70cbb59eca18483dc`。
- 原始 Parquet 在元信息和 token 文件校验完成后删除，避免占满 AutoDL 数据盘；处理后的 token 文件和元信息保留。

## Smoke test

- 步数：`100`。
- 每步 token：`32,768`。
- 唯一语料 token：`3,276,800`，与预期完全一致。
- 设备与精度：CUDA + BF16。
- 最佳验证损失：`7.6361942295977`（step 100）。
- 检查点：`/root/day6_checkpoint_store/day6_336m_500m_fresh_step_00100.pt`。
- 运行时间：约 `87.76 s`。

## 正式训练

- 模型参数量：`336,118,784`（约 336M）。
- 结构配置：`dim=1024`，`layers=24`，`heads=16`，`kv_heads=4`，`seq_len=512`。
- 训练步数：`15,259 / 15,259`。
- 每步 token：`32,768`。
- 目标与实际消费：`500,006,912 / 500,006,912` 个唯一语料 token，完全一致。
- 设备与精度：单卡 CUDA + BF16。
- 最佳验证损失：`3.1427599787712097`（step `15,000`）。
- 最终步：`15,259`。
- 训练耗时：`12,768.126777648926 s`（约 `3.55 h`）。
- 最终检查点：`/root/day6_checkpoint_store/day6_336m_500m_fresh_step_15259.pt`（远端约 3.8G）。
- 指标日志：`/root/autodl-tmp/day6src/pretrainlab_x/logs/day6_336m_500m_fresh_metrics.jsonl`（15,275 行，包含训练与验证记录）。
- 汇总文件：`/root/autodl-tmp/day6src/pretrainlab_x/day6_336m_500m_fresh_summary.json`。

## W&B 与监控边界

W&B 使用 offline（离线）模式写入远端磁盘，避免训练依赖公网。已生成两个 run 目录：

- `offline-run-20260913_121732-j6b9454`（smoke test）
- `offline-run-20260913_121938-ok1gtjf8`（正式训练）

当前没有在线 W&B URL；这不代表没有记录。联网后可将对应目录用 `wandb sync` 上传。训练过程中保留 loss、学习率、梯度和 GPU 监控日志；adaptive monitor 保持关闭，避免未经授权改变训练行为。

## 本地代码入口

- [train_day6.py](./train_day6.py)
- [fresh_preprocessing.py](./data/fresh_preprocessing.py)
- [day6_336m_500m_fresh.yaml](./experiments/day6_336m_500m_fresh.yaml)

## 诚实的范围说明

500M token、336M 参数的单卡运行是面向实习作品集的可审计缩小版预训练实验：它证明了真实语料接入、文档级切分、无回放 token 游标、BF16 训练、验证、检查点和离线实验记录能够闭环运行；它不等同于数十亿参数、万亿 token 的工业预训练，也不提供生产模型质量保证。

## 资源回收

正式训练进程已结束，证据核验完成后已将 B46 AutoDL 实例关机；控制台已确认显示“已关机”，GPU 资源已释放。实际扣费以 AutoDL 账单为准，本文件不臆测金额。
