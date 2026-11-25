"""Shared audio/vision encoder: pool the real frames over time, then a small MLP.

AudioMLP and VisionMLP are this same encoder applied to a different modality.
Inputs are precomputed per-frame features; padding past the declared length is
ignored. Normalisation is left to the shared train-fit Standardize transform.
"""
from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.baseline import _masked_mean


def _masked_stats(x: Tensor, lengths: Tensor) -> Tensor:
    """Mean, std and max over the real frames of (B, T, D) -> (B, 3*D).

    An empty clip gives zeros (std ~1e-4: the variance is floored at 1e-8 so the
    square root keeps a finite gradient for constant or single-frame clips).
    """
    mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    n = mask.sum(1).clamp(min=1)
    mean = (x * mask).sum(1) / n
    std = (((x - mean.unsqueeze(1)) * mask).square().sum(1) / n).clamp_min(1e-8).sqrt()
    peak = x.masked_fill(~mask, float("-inf")).amax(1).nan_to_num(neginf=0.0)
    return torch.cat([mean, std, peak], dim=-1)


def mlp_with_regressor(input_dim: int, hidden_dim: int, dropout: float, num_layers: int) -> nn.Sequential:
    """`num_layers` x (Linear -> ReLU -> Dropout), then Linear(hidden_dim, 1).

    `net[:-1]` is the encoder (output size hidden_dim), `net[-1]` the regressor.
    """
    layers = []
    for i in range(num_layers):
        layers += [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers, nn.Linear(hidden_dim, 1))


# name -> (pooling function, output size as a multiple of the feature dimension)
POOLINGS = {"mean": (_masked_mean, 1), "stats": (_masked_stats, 3)}


class SequenceMLP(MultimodalModel):
    """Features -> pooling over real frames -> Linear/ReLU/Dropout -> regression.

    Subclasses only set `modality` ("audio" or "vision"); the batch field, its
    lengths and the input size are read from it. `pooling`: "mean" (masked mean)
    or "stats" (masked mean + std + max, so the first layer sees 3x the features).
    `num_layers`: number of Linear/ReLU/Dropout blocks before the regressor
    (1 keeps the original layout, so earlier checkpoints still load).
    """

    modality: str

    def __init__(self, config: DatasetConfig, hidden_dim: int = 64, dropout: float = 0.3,
                 pooling: str = "mean", num_layers: int = 1):
        super().__init__(config)
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must be positive and 0 <= dropout < 1")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {list(POOLINGS)}")
        self.pooling = pooling
        self.input_dim = getattr(config, f"{self.modality}_dim")
        self.feature_dim = hidden_dim
        self.net = mlp_with_regressor(POOLINGS[pooling][1] * self.input_dim, hidden_dim, dropout, num_layers)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--pooling", choices=list(POOLINGS), default="mean",
                            help="How frames are pooled over time: mean, or mean+std+max")
        parser.add_argument("--num-layers", type=int, default=1, help="Hidden Linear/ReLU/Dropout blocks")

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "SequenceMLP":
        return cls(config, hidden_dim=args.hidden_dim, dropout=args.dropout, pooling=args.pooling,
                   num_layers=args.num_layers)

    def encode(self, batch: Batch) -> Tensor:
        x = getattr(batch, self.modality)
        if x.ndim != 3 or x.size(-1) != self.input_dim or x.size(1) == 0:
            raise ValueError(f"Expected {self.modality} (B,T,{self.input_dim}) with T > 0")
        lengths = getattr(batch, f"{self.modality}_lengths").to(x.device)
        return self.net[:-1](POOLINGS[self.pooling][0](x, lengths))

    def forward(self, batch: Batch) -> Tensor:
        return self.net[-1](self.encode(batch)).squeeze(-1)
