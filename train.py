"""A compact LLaMA-style decoder-only pretraining loop for Day 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def cfg_load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [1, 1, T, D/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(t, self.head_dim, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = int(cfg["dim"])
        self.attn_norm = RMSNorm(d)
        self.attn = CausalSelfAttention(d, int(cfg["n_heads"]), int(cfg["max_seq_len"]), float(cfg.get("dropout", 0.0)))
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, int(cfg.get("multiple_of", 64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class MiniLlama(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        d = int(cfg["dim"])
        self.tok_embeddings = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(int(cfg["n_layers"]))])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.max_seq_len = int(cfg["max_seq_len"])
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_embeddings(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64, copy=False)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64, copy=False)) for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: nn.Module, data: np.ndarray, cfg: dict, device: torch.device, generator: torch.Generator) -> float:
    model.eval()
    vals = []
    for _ in range(int(cfg.get("eval_batches", 10))):
        x, y = get_batch(data, int(cfg["batch_size"]), int(cfg["max_seq_len"]), device, generator)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        vals.append(float(loss))
    model.train()
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()
    cfg = cfg_load(args.config)
    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = Path(cfg.get("output_dir", "."))
    data_dir = root / cfg.get("data_dir", "data/tokenized")
    train_data = np.load(data_dir / "train_split.npy", mmap_mode="r")
    val_data = np.load(data_dir / "val_split.npy", mmap_mode="r")
    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        data_meta = json.load(f)
    vocab_size = int(data_meta["vocab_size"])
    model = MiniLlama(vocab_size, cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    max_steps = int(cfg["max_steps"])
    warmup = int(cfg.get("warmup_steps", 0))
    base_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_learning_rate", base_lr * 0.1))
    scaler_enabled = device.type == "cuda"
    log_dir = root / "logs"
    ckpt_dir = root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    rng = torch.Generator().manual_seed(seed + 1)
    eval_rng = torch.Generator().manual_seed(seed + 2)
    start = time.time()

    model.train()
    last_loss = None
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, max_steps + 1):
            x, y = get_batch(train_data, int(cfg["batch_size"]), int(cfg["max_seq_len"]), device, rng)
            progress = step / max_steps
            if step <= warmup and warmup > 0:
                lr = base_lr * step / warmup
            else:
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                lr = min_lr + (base_lr - min_lr) * cosine
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
                _, loss = model(x, y)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0))))
            optimizer.step()
            last_loss = float(loss.detach())
            record = {"step": step, "loss": last_loss, "lr": lr, "grad_norm": grad_norm, "tokens": step * int(cfg["batch_size"]) * int(cfg["max_seq_len"]), "elapsed_s": time.time() - start}
            if step % int(cfg.get("eval_interval", 50)) == 0 or step == 1:
                record["val_loss"] = evaluate(model, val_data, cfg, device, eval_rng)
            if step % int(cfg.get("log_interval", 10)) == 0 or step == 1:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            if step % int(cfg.get("save_interval", 100)) == 0 or step == max_steps:
                torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "data_meta": data_meta, "param_count": param_count, "last_record": record}, ckpt_dir / f"step_{step:05d}.pt")

    summary = {"device": str(device), "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "param_count": param_count, "max_steps": max_steps, "final_loss": last_loss, "elapsed_s": time.time() - start, "data": data_meta}
    with (root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

