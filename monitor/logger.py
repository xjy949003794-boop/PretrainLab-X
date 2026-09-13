"""JSONL and optional Weights & Biases metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricLogger:
    """Write every metric locally and mirror it to W&B when enabled."""

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        wandb_enabled: bool = False,
        wandb_project: str = "PretrainLab-X",
        wandb_run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a" if append else "w", encoding="utf-8")
        self._wandb = None
        self.run_info: dict[str, Any] = {}
        if wandb_enabled:
            try:
                import wandb
            except ImportError as exc:  # pragma: no cover - depends on runtime image
                self.close()
                raise RuntimeError("wandb is enabled but is not installed") from exc
            init_kwargs: dict[str, Any] = {
                "project": wandb_project,
                "name": wandb_run_name,
                "config": config or {},
                "resume": "allow",
            }
            mode = (config or {}).get("wandb_mode")
            entity = (config or {}).get("wandb_entity")
            tags = (config or {}).get("wandb_tags")
            group = (config or {}).get("wandb_group")
            job_type = (config or {}).get("wandb_job_type")
            if mode:
                init_kwargs["mode"] = str(mode)
            if entity:
                init_kwargs["entity"] = str(entity)
            if tags:
                init_kwargs["tags"] = list(tags)
            if group:
                init_kwargs["group"] = str(group)
            if job_type:
                init_kwargs["job_type"] = str(job_type)
            self._wandb = wandb.init(**init_kwargs)
            self.run_info = {
                "id": getattr(self._wandb, "id", None),
                "name": getattr(self._wandb, "name", None),
                "url": getattr(self._wandb, "url", None),
                "dir": getattr(self._wandb, "dir", None),
                "mode": mode or "online",
                "project": wandb_project,
                "entity": entity,
            }

    def log(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(record, step=int(record["step"]))

    def close(self) -> None:
        if getattr(self, "_file", None) is not None and not self._file.closed:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
