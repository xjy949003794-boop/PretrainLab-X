# PretrainLab-X Day 1 实际运行记录

执行日期：2026-09-12

## 资源与费用

- AutoDL 西北 B 区 C90 机
- NVIDIA RTX PRO 6000 Blackwell Server Edition，1 卡，96 GB
- 按量计费；实例已关机，已停止 GPU 计费
- 本次新增流水：`-¥3.73`；关机后账户余额：`¥103.85`

## 训练闭环

- 结构：decoder-only LLaMA-style
- 模块：RMSNorm、RoPE、causal scaled dot-product attention、SwiGLU、残差连接、权重绑定
- 参数量：4,951,296
- Tokenizer：本地 ByteLevel-BPE，词表 512
- token 数：2,000,000；上下文长度 256
- 优化：AdamW、BF16 autocast、cosine learning-rate schedule、warmup、gradient clipping
- 训练：500 步；GPU 训练墙钟时间约 8.8 秒
- loss：`6.302780 -> 0.0256798`
- 最后一次 validation loss：`0.0255304`

## 产物（远程实例）

远程目录：`/root/PretrainLab-X`

- `prepare_data.py`
- `train.py`
- `verify_run.py`
- `configs/debug.yaml`
- `data/raw/sample.jsonl`
- `data/tokenized/train.npy`
- `data/tokenized/tokenizer.json`
- `data/tokenized/metadata.json`
- `logs/train.jsonl`
- `checkpoints/step_00100.pt` … `step_00500.pt`
- `run_summary.json`
- `run_audit.json`
- `RUN_REPORT.md`

## 数据边界

本次首先尝试了方案要求的 `HuggingFaceFW/fineweb-edu`，但实例访问 Hugging Face endpoint 不可达。为避免租卡空转，Day 1 改用有明确记录的受限本地技术语料完成工程验证；`metadata.json` 保存了请求数据集、实际来源和网络限制。FineWeb-Edu 接入应在后续拿到可达镜像或预加载副本后再做，不能把这次 smoke run 宣称成 FineWeb-Edu 训练。

## 审计

`verify_run.py` 已通过：tokenized data、metadata、训练日志、checkpoint、summary 均存在，500 条训练记录完整，loss 下降，最终 checkpoint 已在 CPU 上重新加载成功。

