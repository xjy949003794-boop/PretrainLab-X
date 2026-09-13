"""Warmup followed by cosine learning-rate decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineScheduler:
    """A tiny explicit scheduler whose step number is checkpoint-friendly."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.last_step = 0

    def lr_at(self, step: int) -> float:
        step = max(1, int(step))
        if self.warmup_steps and step <= self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, step: int) -> float:
        self.last_step = int(step)
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"last_step": self.last_step}

    def load_state_dict(self, state: dict) -> None:
        self.last_step = int(state.get("last_step", 0))
