from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch


class MultimodalModel(nn.Module, ABC):
    """Contract for a model architecture that plugs into this project's pipeline.

    Subclass this, implement `__init__(self, config)` (read `config.text_dim`,
    `config.audio_dim`, `config.vision_dim` instead of hardcoding dimensions) and
    `forward(self, batch) -> Tensor` returning shape (B,) raw-scale predictions on
    the dataset's label scale. Loss computation is deliberately NOT part of this
    interface — that's your training routine's job, not the model's.
    """

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: Batch) -> Tensor: ...
