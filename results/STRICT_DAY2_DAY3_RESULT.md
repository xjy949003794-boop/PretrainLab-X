# PretrainLab-X：Day 2 / Day 3 严格执行记录

执行日期：2026-09-12  
远程实例：AutoDL 西北 B 区 A09，实例 ID `505f459471-58905153`  
状态：已完成并已关机；C79、C90、D04 也保持关机。

## 1. 本次到底跑了什么

这次没有使用之前的 4.36M 参数、120 步 smoke test（冒烟测试）结果，而是按严格配置完成了：

- 30.42M 参数的现代 LLaMA-style decoder-only 模型；
- RMSNorm、RoPE、SwiGLU、GQA（16 个 query heads / 4 个 KV heads，repeat=4）；
- BF16 autocast、梯度累积 8、梯度裁剪 1.0；
- 25 步 warmup + cosine learning-rate schedule；
- 每 100 步保存 checkpoint，并保存 RNG、optimizer、scheduler 状态；
- W&B 离线记录 + 本地 JSONL；
- 高学习率稳定性实验；
- 从 step 400 checkpoint 恢复到 step 500 的实测验证；
- 三份 observatory 报告和确定性的异常诊断 demo。

模型参数量：`30,419,456`（30.42M）。

## 2. 远程环境与数据

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，约 97,887 MiB；
- PyTorch：`2.8.0+cu128`，CUDA 12.8；
- BF16：可用；
- 数据：2,000,000 tokens，train 1,800,000 / validation 200,000，vocab 512；
- `dataset_source`：`builtin-fallback`，6 条内置文本记录。

数据端点在该实例上不可用，因此严格训练使用了确定性的 byte-level fallback（字节级回退语料）来保证模型、训练引擎和监控链路完整可复现。它是工程验收语料，不是 FineWeb-Edu 或大规模互联网语料，不能把本次结果表述成大规模语料预训练。

## 3. 实验结果

| 实验 | 记录/步数 | 起止 step | 首个 loss → 最后 loss | 最佳 loss | 最大（裁剪前）grad norm | 检查点 |
|---|---:|---:|---:|---:|---:|---:|
| 30M 正式组 | 500 / 500 | 0 → 500 | 6.247265 → 0.015530 | 0.014527 | 15.01231 | 100/200/300/400/500 |
| 高学习率组（LR=3e-3） | 500 / 500 | 0 → 500 | 6.247265 → 0.044620 | 0.042047 | 19.08514 | 100/200/300/400/500 |
| 正常 LR 恢复组 | 500 / 500 | 0 → 500 | 6.247265 → 0.015530 | 0.014527 | 15.01231 | 100/200/300/400/500 |
| checkpoint 恢复验证 | 100 / 100 | 400 → 500 | 0.016214 → 0.015530 | 0.014527 | 0.07782 | step 500 |

正式组最终 validation loss 为 `0.0156888`；高 LR 组为 `0.0462291`；正常 LR 恢复组为 `0.0156888`。四组均保持有限数值，没有 NaN/Inf；高 LR 组的 loss 仍下降，但明显比正常学习率组更差。

## 4. 监控与异常结论

生成的文件：

- `logs/day2_day3_30m_metrics.jsonl`
- `logs/day3_high_lr_metrics.jsonl`
- `logs/day3_recovery_metrics.jsonl`
- `logs/day3_resume_metrics.jsonl`
- `day2_day3_30m_observatory.json`
- `day3_high_lr_observatory.json`
- `day3_recovery_observatory.json`
- `logs/anomaly_demo.jsonl`
- `anomaly_demo_report.json`

Observatory（观测器）把三次正式日志标记为 `gradient_explosion_risk`，原因是它记录的是**裁剪前**全局梯度范数，阈值为 10；实际训练随后按 `grad_clip=1.0` 进行了裁剪，且没有发生非有限值。这个标记是诊断提醒，不等于训练崩溃。异常 demo 能稳定复现 loss spike（损失突增）和梯度风险的诊断路径。

## 5. FlashAttention 与 W&B 的边界

- 已按方案尝试 `pip install flash-attn --no-build-isolation`；补装 `ninja` 后又重试。远程环境的 wheel 编译一直停在 `Building wheel for flash-attn (setup.py): started`，`flash_attn` import spec 最终仍为 `False`，为避免继续空耗 GPU 费用已在关机前终止编译进程。
- 实际训练使用 PyTorch `scaled_dot_product_attention`，运行记录为 `attention_backend=sdpa_auto`。这允许 PyTorch 在硬件支持时自行选择融合 kernel，但不能声称第三方 `flash_attn` wheel 已安装。
- W&B 已安装并在四次运行中以 `WANDB_MODE=offline` 写入本地离线 run；实例没有登录 W&B 账号，因此没有伪造在线 dashboard 链接。本地 JSONL 是完整主记录。

## 6. 费用与关机

账单页面显示本次 A09 订单 `7892008130640613697` 的两笔扣费为 `￥5.40 + ￥0.21 = ￥5.61`，最后一段计费周期为 17:00:00–17:01:46；关机后状态已确认是“已关机”。充值后的余额为 `￥107.58`，扣除此前 C90 `￥3.73`、C79 `￥2.94` 以及本次 A09 `￥5.61` 后页面余额为 `￥95.30`。后续不会继续产生 GPU 运行费。

## 7. 结论边界

这次是一个**完整的 Day 2 / Day 3、30M-class、单 GPU 预训练工程闭环**：模型结构、训练引擎、稳定性实验、checkpoint 恢复和监控都实际跑通了 500 步。它仍然不是 Day 4 的 FSDP 多卡训练，也不是大规模 FineWeb/网页语料预训练；数据回退语料和单卡规模是已明确记录的限制。

源码与配置在本地目录 `pretrainlab_x/`；严格配置包括 `configs/day2_day3_strict.yaml`、`configs/day3_high_lr_strict.yaml`、`configs/day3_recovery_strict.yaml` 和 `configs/day3_resume_strict.yaml`。
