"""A simple audio-only comparison model; no claim of a published reproduction."""
from __future__ import annotations

import argparse

from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.baseline import _masked_mean


class AudioMLP(MultimodalModel):
    def __init__(self, config: DatasetConfig, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__(config)
        self.feature_dim = hidden_dim
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must be positive and 0 <= dropout < 1")
        self.net = nn.Sequential(
            nn.Linear(config.audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "AudioMLP":
        return cls(config, hidden_dim=args.hidden_dim, dropout=args.dropout)

    def encode(self, batch: Batch) -> Tensor:
        return self.net[:-1](_masked_mean(batch.audio, batch.audio_lengths))

    def forward(self, batch: Batch) -> Tensor:
        return self.net[-1](self.encode(batch)).squeeze(-1)
