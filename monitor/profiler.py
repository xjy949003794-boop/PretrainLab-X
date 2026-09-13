"""Small timing helpers for throughput accounting."""

from __future__ import annotations

import time


class StepTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started
