from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch

if TYPE_CHECKING:
    from multimodal_ml.training import Loaders


class MultimodalModel(nn.Module, ABC):
    """Contract for a model architecture that plugs into this project's pipeline.

    Subclass this, implement `__init__(self, config)` (read `config.text_dim`,
    `config.audio_dim`, `config.vision_dim` instead of hardcoding dimensions) and
    `forward(self, batch) -> Tensor` returning shape (B,) raw-scale predictions on
    the dataset's label scale.
    """

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: Batch) -> Tensor: ...

    def training_routine(
        self, loaders: "Loaders", device: torch.device, args: argparse.Namespace
    ) -> dict[str, float]:
        """Runs this model's full train/validate/test routine; returns test metrics.

        Default: multimodal_ml.training.default_routine (standard MSE regression
        with early stopping) — fine for most models. Override this method on YOUR
        model's own class (in your own models/your_model.py) if it needs a
        different loss / LR schedule / gradient accumulation / etc. Keeping the
        override on your own class means customizing it never requires editing
        training.py or __main__.py, which everyone else's models also rely on.

        By convention, overrides should still honor args.epochs/lr/patience/
        batch_size unless there's a specific reason not to, so `--epochs 1` etc.
        behaves predictably across every model.
        """
        from multimodal_ml.training import default_routine

        return default_routine(self, loaders, device, args)
