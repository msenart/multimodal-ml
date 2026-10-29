from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metric_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "epoch": epoch, "metric_value": metric_value},
        path,
    )


def load_checkpoint(path: Path, model: nn.Module) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return {"epoch": checkpoint["epoch"], "metric_value": checkpoint["metric_value"]}


class EarlyStopper:
    """Call `stopper.step(metric_value)` once per epoch from your own training loop.

    Returns True once `metric_value` hasn't improved for `patience` consecutive
    calls (stop training). Check `stopper.is_best` right after `step()` to know
    whether this epoch's value is the best seen so far (i.e. whether to checkpoint).
    """

    def __init__(self, patience: int = 5, mode: Literal["min", "max"] = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.num_bad_epochs = 0
        self.is_best = False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def step(self, metric_value: float) -> bool:
        if self._improved(metric_value):
            self.best = metric_value
            self.num_bad_epochs = 0
            self.is_best = True
        else:
            self.num_bad_epochs += 1
            self.is_best = False
        return self.num_bad_epochs >= self.patience
