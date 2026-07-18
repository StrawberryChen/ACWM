"""Small W&B boundary so training also works with logging disabled."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, config: dict[str, Any], full_config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.run = None
        if self.enabled:
            try:
                import wandb
            except ImportError as error:
                raise ImportError("W&B logging is enabled; install the 'colab' extra or wandb") from error
            self.wandb = wandb
            self.run = wandb.init(project=config.get("project", "acwm"), entity=config.get("entity"),
                                  name=config.get("name"), mode=config.get("mode", "online"),
                                  config=full_config)

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def video(self, path: str | Path, fps: int = 20):
        if self.run is None:
            return str(path)
        try:
            return self.wandb.Video(str(path), fps=fps, format=Path(path).suffix.lstrip("."))
        except Exception as error:
            print(f"Warning: failed to prepare W&B video {path}: {error}")
            return str(path)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def append_metrics(path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
