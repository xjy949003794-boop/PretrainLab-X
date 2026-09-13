"""Small adapter around an openly downloadable LLaMA-family tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenizerAdapter:
    tokenizer: Any
    name: str

    @property
    def vocab_size(self) -> int:
        value = getattr(self.tokenizer, "vocab_size", None)
        if value is None:
            value = len(self.tokenizer)
        return int(value)

    @property
    def eos_id(self) -> int:
        token_id = getattr(self.tokenizer, "eos_token_id", None)
        if token_id is None:
            raise RuntimeError("the selected tokenizer has no EOS token")
        return int(token_id)

    @property
    def pad_id(self) -> int | None:
        token_id = getattr(self.tokenizer, "pad_token_id", None)
        return None if token_id is None else int(token_id)

    def encode(self, text: str) -> list[int]:
        return [int(x) for x in self.tokenizer.encode(text, add_special_tokens=False)]


def build_tokenizer(name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir: str | None = None) -> TokenizerAdapter:
    """Load a public LLaMA-family tokenizer through ``transformers``.

    We intentionally fail loudly instead of silently replacing a real tokenizer
    with the Day 1 byte fallback.  A Day 4 run must record the actual tokenizer
    used in its metadata.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("transformers is required for the Day 4 tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"tokenizer {name!r} does not expose eos_token_id")
    return TokenizerAdapter(tokenizer=tokenizer, name=name)
