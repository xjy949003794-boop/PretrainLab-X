# Day 2–3 执行结果

执行日期：2026-09-12

## 运行环境

- AutoDL：西北 B 区 C79，`RTX PRO 6000` 1 卡。
- GPU 已在任务完成后关机；C79、原 C90、D04 当前均为“已关机”。
- 训练设备：`cuda`。
- 注意力实现：PyTorch `scaled_dot_product_attention` 的 `sdpa_auto` 后端；没有额外编译 `flash-attn`，以避免把时间花在环境安装上。

## Day 2：现代 LLaMA 风格基座模型

已实现并通过远端 CPU 前向检查：

- `RMSNorm`
- `RoPE`
- `SwiGLU`
- `GQA`（8 个 query heads、2 个 key/value heads，重复因子 4）
- 权重绑定的 decoder-only causal language model

模型配置为 4,361,472 个参数、6 层、隐藏维度 256、词表 512。

## Day 3：训练引擎与观测

已实现：

- BF16 自动混合精度（GPU 上实际生效）
- 梯度累积 4 步
- 梯度裁剪
- warmup + cosine 学习率调度
- 原子 checkpoint，包含模型、优化器、调度器、随机数状态和数据元数据
- 从 step 60 断点恢复到 step 120
- JSONL 训练日志、吞吐、梯度范数、峰值显存、验证 loss
- 离线异常诊断与确定性梯度爆炸演示

## 实际结果

| 指标 | 结果 |
|---|---:|
| 训练日志记录数 | 120 |
| 训练步数 | 1 → 120（含 60 → 120 续训） |
| token 数 | 491,520 |
| 首步 loss | 6.2887 |
| 末步 loss | 1.5323 |
| 最佳验证 loss | 1.5603 |
| 最大梯度范数 | 8.7660 |
| 平均吞吐 | 113,706 token/s |
| 峰值显存 | 205.5 MB |
| 观测报告 | `passed: true`，无异常 |

检查点已验证可读：

- `checkpoints/day2_day3_step_00060.pt`
- `checkpoints/day2_day3_step_00120.pt`

## 费用核对

AutoDL 收支明细显示：

- 本次新 C79 实例支出：`¥2.94`（备注为 `47034ea2ad-a188e164`）。
- 之前 C90 实例的 `¥3.73` 是另一笔历史支出，不计入本次 Day 2–3。
- 关机后余额：`¥100.91`。

这是一个**可复现的短程工程 smoke test（冒烟验证）**，不是大规模语料上的完整基座模型预训练；它验证的是现代架构、训练控制、断点恢复和观测链路已经真正跑通。
