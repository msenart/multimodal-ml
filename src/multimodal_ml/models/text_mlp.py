"""A simple text-only comparison model; no claim of a published reproduction."""
from __future__ import annotations

import argparse

from torch import Tensor

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.sequence_mlp import mlp_with_regressor


class TextMLP(MultimodalModel):
    """Text features -> mean over time -> MLP -> overall sentiment regression.

    Batches carry no text lengths, so the mean covers every padded position
    (same pooling as ConcatMLP). Only text enters the model.
    """

    def __init__(self, config: DatasetConfig, hidden_dim: int = 64, dropout: float = 0.3, num_layers: int = 1):
        super().__init__(config)
        if hidden_dim < 1 or not 0 <= dropout < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive and 0 <= dropout < 1")
        self.feature_dim = hidden_dim
        self.net = mlp_with_regressor(config.text_dim, hidden_dim, dropout, num_layers)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--num-layers", type=int, default=1, help="Hidden Linear/ReLU/Dropout blocks")

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "TextMLP":
        return cls(config, hidden_dim=args.hidden_dim, dropout=args.dropout, num_layers=args.num_layers)

    def encode(self, batch: Batch) -> Tensor:
        text = batch.text
        if text.ndim != 3 or text.size(-1) != self.config.text_dim or text.size(1) == 0:
            raise ValueError(f"Expected text (B,T,{self.config.text_dim}) with T > 0")
        return self.net[:-1](text.mean(dim=1))

    def forward(self, batch: Batch) -> Tensor:
        return self.net[-1](self.encode(batch)).squeeze(-1)
