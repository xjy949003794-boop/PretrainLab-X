"""Lightweight GPU memory and utilization sampling."""

from __future__ import annotations

import subprocess

import torch


def snapshot(device: torch.device) -> dict[str, float | None]:
    """Return memory metrics plus nvidia-smi utilization when available."""
    if device.type != "cuda":
        return {
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "peak_memory_mb": None,
            "gpu_utilization_pct": None,
        }
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    utilization: float | None = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        utilization = float(output.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        pass
    return {
        "memory_allocated_mb": allocated,
        "memory_reserved_mb": reserved,
        "peak_memory_mb": peak,
        "gpu_utilization_pct": utilization,
    }
