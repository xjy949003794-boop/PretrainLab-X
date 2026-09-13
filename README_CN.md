<p align="center">
  <a href="./README.md">English</a> | <b>简体中文</b>
</p>
# PretrainLab-X：可审计的迷你基础模型预训练栈

这是一个面向大模型预训练工程岗位的**缩小版、可检查、可复现**项目。它不是把开源训练脚本换个名字，而是把数据、模型、训练引擎、监控、评估和证据归档连成完整链路。

## 做了什么

- 现代 LLaMA 风格 decoder-only Transformer：RMSNorm、RoPE、SwiGLU、GQA。
- 使用 PyTorch 的 scaled-dot-product attention（SDPA），运行记录为 `sdpa_auto`；没有把第三方 `flash-attn` wheel 安装成功包装成已完成。
- BF16 自动混合精度、梯度累积、梯度裁剪、warmup + cosine 学习率、原子 checkpoint、优化器/调度器/RNG 恢复、可选激活检查点。
- FineWeb-Edu 预处理代码：过滤、去重、文档边界切分、tokenizer 溯源和 token 文件哈希。
- JSONL 主日志、可选 W&B 离线日志、训练健康诊断、共享留出评估、配对 bootstrap 置信区间、sanity check 和 SHA-256 审计。

## 关键证据

| 阶段 | 已核验结果 | 必须同时说明的边界 |
|---|---:|---|
| 严格 Day 2-3 | 30,419,456 参数，500 步 | 使用确定性的 byte-level fallback 语料 |
| Day 4 300M 基准 | 336,118,784 参数，8 步 | 短程基准，不是 3B token 长训 |
| Day 4 700M 基准 | 653,477,120 参数，4 步 | 短程基准，不是 5B token 长训 |
| Day 5 长跑 | 336M，15,259 步，处理 500,006,912 token，最终 loss 0.069339 | 9.9M token 切片被重复采样 |
| Day 6 新鲜数据 | 336M，15,259 步，500,006,912 个唯一 token | 单卡缩小版验证 |
| 共享留出评估 | Day5 NLL 8.424211；Day6 NLL 3.039400；差值 5.384811 | 不是新鲜度因果证明 |
| Day 7 | checkpoint、配置、随机基线、精度交叉核验通过 | 没有重新训练或修改权重 |

机器可读唯一数字源是 [`results/final_metrics.json`](results/final_metrics.json)。阶段报告位于 [`results/`](results/)。

## 目录说明

```text
model/          模型和注意力实现
engine/         训练器、调度器、checkpoint
data/           预处理和 tokenizer 适配（不含原始数组）
distributed/    FSDP 入口
evaluation/     留出评估、bootstrap、sanity、审计
monitor/        JSONL/W&B、GPU/梯度和诊断
experiments/    配置、基准和报告脚本
results/        人类可读报告和小型机器可读证据
figures/        发布图表
audit/          权威哈希清单与交叉核验
docs/           设计、数据、实验、评估说明
```

## 快速运行

公开包不包含原始 token 数组和模型 checkpoint。准备自己的小型 token 数组后，可执行：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py --config configs/debug.yaml
python train.py --config configs/debug.yaml
python verify_run.py --run-dir .
```

详细复现边界见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

## 诚实边界

这是作品集级的工程闭环，不是工业级预训练，也不是多机多卡性能报告，更不是严格控制所有变量的数据新鲜度因果实验。请先阅读 [`LIMITATIONS.md`](LIMITATIONS.md)。
