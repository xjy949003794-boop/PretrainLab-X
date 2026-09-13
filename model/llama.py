"""Small, inspectable LLaMA-style decoder-only language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .attention import GQAAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


@dataclass
class LlamaConfig:
    vocab_size: int
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 256
    multiple_of: int = 64
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, vocab_size: int, values: dict) -> "LlamaConfig":
        return cls(
            vocab_size=vocab_size,
            dim=int(values.get("dim", 256)),
            n_layers=int(values.get("n_layers", 6)),
            n_heads=int(values.get("n_heads", 8)),
            n_kv_heads=int(values.get("n_kv_heads", values.get("n_heads", 8))),
            max_seq_len=int(values.get("max_seq_len", 256)),
            multiple_of=int(values.get("multiple_of", 64)),
            dropout=float(values.get("dropout", 0.0)),
            rope_base=float(values.get("rope_base", 10000.0)),
            tie_embeddings=bool(values.get("tie_embeddings", True)),
            gradient_checkpointing=bool(values.get("gradient_checkpointing", False)),
        )


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim)
        self.attention = GQAAttention(
            cfg.dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.max_seq_len,
            cfg.dropout,
            cfg.rope_base,
        )
        self.ffn_norm = RMSNorm(cfg.dim)
        self.feed_forward = SwiGLU(cfg.dim, cfg.multiple_of)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.config = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def attention_backend(self) -> str:
        return self.layers[0].attention.backend_name

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x = activation_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss
