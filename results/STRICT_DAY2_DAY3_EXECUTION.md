# 严格版 Day 2/Day 3 执行清单

该清单对应 `/Users/xiongjunyi/Downloads/研究方案day2day3`，不使用此前的 4.36M/120 步压缩配置。

## Day 2

1. 分模块运行 `test_rmsnorm.py`、`test_rope.py`、`test_swiglu.py`、`test_attention.py`、`test_model.py`。
2. 使用 16 个 query heads、4 个 KV heads 的 30M-class LLaMA 配置。
3. 尝试安装独立 `flash-attn` 包；若上游包与当前 CUDA/Blackwell 环境确实不兼容，保留安装日志并明确记录实际可用后端，不把失败伪装成成功。
4. 完整前向测试后，运行 30M-class 模型的 500 步 BF16 小训练。

## Day 3

1. 运行 BF16、8 步梯度累积、梯度裁剪、warmup+cosine scheduler。
2. 保存并读取 step 100/200/300/400/500 checkpoint。
3. 记录 loss、learning rate、gradient norm、GPU 显存、GPU 利用率和 throughput。
4. 安装并接入 W&B；需要用户提供 W&B API key 时暂停在登录步骤，不伪造线上曲线。
5. 用学习率 `3e-3` 运行独立稳定性实验，记录 loss spike/梯度风险。
6. 用 `3e-4 + warmup + clipping` 重新训练恢复，并生成对照报告。

## 明确边界

这仍然是原方案 Day 2/3 的 30M-class 小规模预训练阶段，不是 Day 4 的 FSDP、300M/700M 多卡规模实验；但本次不再把 Day 2/3 自行缩小为 4.36M/120 步 smoke test。
