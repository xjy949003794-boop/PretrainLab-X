# Day7 补丁：Hash Audit & Documentation Repair（哈希审计与文档修复）

## 结论

Day7 补丁已完成。此次只读审计直接从 B46 上的三个原始文件计算 SHA-256（文件指纹），
并用 `sha256sum`、OpenSSL 和 Python `hashlib` 交叉核对。三份文件的哈希长度均为 64，
三种方法全部一致。

本次没有重新训练、没有重新评估，也没有修改模型权重、训练历史、指标或评估输出。

## 权威文件指纹

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| Day5 checkpoint | 4,033,742,349 | `2c49925b61e46c533026c201dba02bfe845071e7c453964fcc33f508a24c36f3` |
| Day6 checkpoint | 4,033,737,695 | `3d1250567b0d3f311584b15a3ac1d73539cdd52b0baebfd7b290e1f7fdf21aca` |
| shared held-out tokens | 40,004,376 | `3f0d22ee8933fc828a8876bf044ef7b40f96bf8cb0f98509a21df9214a6cc750` |

完整机器可读清单见 `CANONICAL_HASH_MANIFEST.json`；交叉核对见 `HASH_CROSSCHECK.json`。

## 修复内容

早期 Day6.6/Day7 文档存在两处人工转录错误：

1. 留出令牌哈希少了一个字符，原记录为非法的 63 字符字符串；已统一为现场计算的 64 字符值。
2. Day6 checkpoint 哈希有相邻字符转置；已统一为现场计算值。

原始错误版本没有删除，保存在带 `.pre_day7_patch` 后缀的备份文件中，便于审计和对照。

## 远端产物

审计原始输出保留在 B46 数据盘：

```text
/root/autodl-tmp/day7_patch/raw/
/root/autodl-tmp/day7_patch/reports/
```

其中包含 `CANONICAL_HASH_MANIFEST.json`、`HASH_CROSSCHECK.json`、
`HASH_VERIFICATION_FINAL.txt` 和本结果说明。实例完成后应关机以停止 GPU 计费。
