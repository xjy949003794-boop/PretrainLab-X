"""Gradient measurements used by the training observatory."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def global_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return the L2 norm of all finite gradients before clipping."""
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum().double().cpu()
    value = float(torch.sqrt(squared))
    return value if math.isfinite(value) else float("nan")


def clip(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """Clip gradients and return the pre-clipping norm."""
    value = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)
    return float(value.detach().cpu())
