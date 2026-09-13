"""Atomic checkpoint save/load, including optimizer and RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    config: dict,
    data_meta: dict,
    last_record: dict,
    extra_state: dict | None = None,
) -> None:
    """Write a complete checkpoint, then atomically replace the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "config": config,
        "data_meta": data_meta,
        "last_record": last_record,
        "rng_state": _rng_state(),
        "extra_state": extra_state or {},
    }
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Restore model/optimizer/scheduler and return the saved metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload.get("rng_state", {}))
    return payload
