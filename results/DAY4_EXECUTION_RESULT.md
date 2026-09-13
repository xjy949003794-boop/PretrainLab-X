# Day4 执行结果与边界说明

执行日期：2026-09-12  
项目：PretrainLab-X  
远端设备：AutoDL，1 × NVIDIA RTX PRO 6000 96GB（页面标价 ¥6.98/小时）

## 先给结论

本次已经完成了 Day4 的**工程闭环和可复现实验**，并在实验结束后关闭了 AutoDL 实例。完成内容包括：

- 从官方 FineWeb-Edu 数据源准备了 10,000,000 个真实 token（不是内置 fallback 文本）；
- 使用 TinyLlama tokenizer，执行长度过滤、控制字符清理、精确文本去重、EOS 拼接以及 train/validation 切分；
- 在真实 GPU 上验证 336M 参数和 653M 参数的 LLaMA-style 模型前向、反向和短程训练；
- 在单进程 `world_size=1` 上验证 FSDP（Fully Sharded Data Parallel，完全分片数据并行）代码路径；
- 运行梯度异常检测演示，主动制造梯度爆炸风险并被监控器识别；
- 生成 scaling report（规模实验报告）PDF 和对应 JSON 指标。

但必须如实标注：**没有启动 3B-token 的 300M 正式长训、5B-token 的 700M 正式长训，也没有启动 4-GPU FSDP。**这不是把正式训练伪装成完成，而是根据实际速度、余额和资源情况停止在可验证的 smoke test（冒烟验证）范围内。

## 与 Day4 计划逐项对照

| 计划项 | 本次状态 | 证据/说明 |
|---|---|---|
| 真实预训练语料 | 已完成（缩小数据量） | `data/fineweb_10m/metadata.json`；10M tokens，`real_data=true` |
| 300M 模型 | 已完成短程基准 | 336,118,784 参数，8 steps，BF16 |
| 700M 模型 | 已完成短程基准 | 653,477,120 参数，4 steps，BF16 |
| FSDP | 已完成代码路径验证 | `world_size=1`；没有声称完成多卡分片训练 |
| 4-GPU FSDP | 未启动 | 当时没有可用的 4 卡主机，且会显著增加费用 |
| 3B-token / 5B-token 长训 | 未启动 | 当前单卡预算不足，且 10M 独立 token 不能被描述为 3B/5B 独立语料 |
| scaling report | 已完成 | `report/scaling_analysis.pdf` 与 `scaling_metrics.json` |
| 训练异常诊断 | 已完成 | `monitor/anomaly_demo.py` 检测到 `gradient_explosion_risk` |
| W&B 在线记录 | 未完成 | 容器没有 `wandb` 包，且远端出网受限；保留本地 JSONL 作为审计记录 |

## 数据处理结果

数据目录：`data/fineweb_10m/`

- 数据集：`HuggingFaceFW/fineweb-edu`
- 配置：`sample-10BT`
- 来源模式：官方 dataset-server API（在本地获取后通过校验和传到 AutoDL）
- 看到的文档数：8,479
- 保留文档数：8,478
- 删除的精确重复文档：1
- 清理后字符数：39,822,471
- 总 token 数：10,000,000
- train split：9,900,000
- validation split：100,000
- tokenizer：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 词表大小：32,000
- EOS id：2
- token 数组原始字节（`tokens.npy` 加载后的 `int32` 数组）SHA-256：`53214170c405d3accd4b643950a82d0e71909430e9e8f4aef3029fd610ce6b13`；这是数据内容校验，不是 `.npy` 文件封装头的校验
- 过滤规则：最短 200 字符、最长 200,000 字符、控制字符替换、精确文本去重

数据集说明见官方 [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。

## 远端实验实测结果

### 336M（计划中的 300M 配置）

- 参数量：336,118,784
- 训练步数：8
- 处理 token：16,384
- 最终 train loss：8.98883056640625
- 最终 validation loss：8.708099365234375
- 用时：5.797656 秒
- 精度：BF16
- 注意力后端：PyTorch SDPA 自动后端（不是声称安装了第三方 `flash-attn`）
- 指标日志：`report/day4_300m_benchmark_metrics.jsonl`

### 653M（计划中的 700M 配置）

- 参数量：653,477,120
- 训练步数：4
- 处理 token：8,192
- 最终 train loss：9.158004760742188
- 最终 validation loss：9.078956604003906
- 用时：6.640977 秒
- 精度：BF16
- 注意力后端：PyTorch SDPA 自动后端
- 指标日志：`report/day4_700m_benchmark_metrics.jsonl`

### FSDP 路径

- 参数量：336,118,784
- `world_size`：1
- 训练步数：16
- 首步 loss：10.5712890625
- 末步 loss：7.652740478515625
- 用时：5.268668 秒
- 精度：BF16
- 日志：`report/fsdp_metrics.jsonl`

单进程运行时 PyTorch 会使用 `NO_SHARD`，因此这一项证明的是 FSDP 初始化、包装、前向/反向和优化器路径可运行，**不等于 4 卡通信与显存分片已经验证**。

## 为什么没有直接跑计划中的长训

本次单卡标价为 ¥6.98/小时，账户余额当时约 ¥95。用短程基准的总 token/总时间做保守估算：

| 目标 | 基准吞吐（含短跑开销） | 估算 GPU 时间 | 估算费用 |
|---|---:|---:|---:|
| 300M × 3B tokens | 约 2,826 tokens/s | 约 295 小时 | 约 ¥2,058 |
| 700M × 5B tokens | 约 1,234 tokens/s | 约 1,126 小时 | 约 ¥7,859 |

这些是量级估算，不是扣费账单；实际稳态吞吐、启动时间和平台价格都会影响结果。它们足以说明：用当时约 ¥95 的余额直接启动两个正式长训会在中途耗尽，而且会把 10M token 数据重复采样后误写成独立 3B/5B 语料。因此本次没有擅自充值、没有擅自扩大预算，也没有伪造完成状态。

## 可复现命令

在项目根目录执行：

```bash
cd pretrainlab_x
PYTHONPATH=. python3 -m compileall -q .
PYTHONPATH=. WANDB_MODE=offline python3 train_day4.py --config experiments/benchmark_300m.yaml
PYTHONPATH=. WANDB_MODE=offline python3 train_day4.py --config experiments/benchmark_700m.yaml
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
  distributed/run_fsdp.py --config experiments/fsdp.yaml
PYTHONPATH=monitor:. python3 monitor/anomaly_demo.py
python3 experiments/scaling_report.py \
  --run 300m=report/day4_300m_benchmark_summary.json \
  --run 700m=report/day4_700m_benchmark_summary.json \
  --run fsdp=report/fsdp_summary.json \
  --output report/scaling_analysis.pdf \
  --json report/scaling_metrics.json
```

## 产物清单

- `model/llama.py`：现代 LLaMA-style 模型配置、untied head 和 activation checkpointing（激活检查点）支持
- `data/preprocessing.py`：真实 FineWeb-Edu 流式读取、过滤、去重、切分
- `data/tokenizer.py`：tokenizer 适配器
- `distributed/fsdp.py`、`distributed/run_fsdp.py`：FSDP 初始化和运行入口
- `experiments/300m_train.yaml`、`experiments/700m_train.yaml`：计划目标配置（未启动长训）
- `experiments/benchmark_300m.yaml`、`experiments/benchmark_700m.yaml`：本次低成本基准配置
- `report/day4_300m_benchmark_summary.json`
- `report/day4_700m_benchmark_summary.json`
- `report/fsdp_summary.json`
- `report/scaling_metrics.json`
- `report/scaling_analysis.pdf`

AutoDL 实例已在全部实验完成后关机；没有留下运行中的 GPU 实例。准确扣费以 AutoDL 的账单页面为准，本报告不虚构一个无法读取的总金额。
