# Day8 执行结果：PretrainLab-X 发布归档

执行日期：2026-09-14  
执行范围：代码、实验证据、报告和发布自检；不重新训练模型，不重新评估，不上传 W&B。

## 结论

Day8 的本地发布包已完成，目录为：

```text
day8_release/PretrainLab-X/
```

它是一个**公开安全的源码与证据发布包**，不是包含模型权重和原始语料的完整训练工作区。

## 已完成

1. 收拢 Day2-Day7 的模型、训练引擎、评估、分布式路径、配置、测试和脚本。
2. 从 B46 取回缺失的 Day6 原始指标日志、Day6 汇总，以及 Day7 合理性检查报告和机器可读结果。
3. 生成 `results/final_metrics.json`，所有数字均来自已收录的报告或机器可读证据。
4. 生成 5 张发布图：学习率对比、梯度风险、共享留出 NLL、记忆化/泛化、Day6 训练曲线。
5. 生成 `results/pdf/PretrainLab-X_Day8_Technical_Report.pdf`，并完成 4 页渲染检查。
6. 写入中英文 README、设计说明、数据管线、实验说明、评估说明、复现说明和限制说明。
7. 加入 `.gitignore`，防止误提交权重、原始语料、缓存和离线 W&B 文件。
8. 在 B46 上以 `PYTHONPATH=.` 运行 6 个组件自检：RMSNorm、RoPE、SwiGLU、GQA、完整模型前向和训练观测逻辑，全部通过。
9. 运行发布审计脚本，检查必需文件、哈希格式、JSON、敏感标记、文件大小和禁止后缀。

## 证据边界

- Day6 原始 JSONL 已纳入发布包，用于绘制训练曲线。
- 严格 Day2-Day3 的原始 JSONL 不在现有工作区；发布图使用严格报告中的端点表，并在文档中明确标注为 endpoint view（端点视图）。
- Day4 的 FSDP 证据是 `world_size=1` 路径验证，不应表述为多 GPU 扩展训练。
- Day5/Day6 的大模型权重、原始语料、token 数组和 516MB 级数据文件未复制到公开包；它们保留在私有工作区或远端磁盘。
- W&B 离线目录没有上传到线上；JSONL 和 Markdown 证据已经足以复现主要图表，避免上传重复二进制和隐私风险。

## 主要产物

```text
README.md
README_CN.md
LIMITATIONS.md
REPRODUCIBILITY.md
results/final_metrics.json
results/pdf/PretrainLab-X_Day8_Technical_Report.pdf
figures/*.png
audit/CANONICAL_HASH_MANIFEST.json
audit/HASH_CROSSCHECK.json
scripts/build_release_artifacts.py
scripts/verify_release.py
```

## GitHub 状态

GitHub 私有仓库已创建：

```text
https://github.com/xjy949003794-boop/PretrainLab-X
```

当前已核对仓库可见性为 **Private（私有）**。代码推送尚未完成：本机没有可用的 GitHub 命令行凭据，且浏览器上传控件拒绝自动注入本地目录，因此不能把“代码已上传”写成既成事实。发布包本身已完整保存在本机；待用户在已登录环境中执行一次 `git push -u origin main` 或手动上传后，仓库即可获得完整源码。

## 计费与远端清理

B46 仅用于收拢缺失证据，没有进行 GPU 训练；本次已确认状态为“已关机”。远端归档压缩包可以留在数据盘，不会继续产生 GPU 开机计费，但会占用少量磁盘空间。
