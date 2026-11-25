"""Trainable visual-feature encoder with a separate sentiment regression head.

This small feature-level baseline uses length-masked pooling and optional
train-only z-scoring. Raw image feature extractors are not trained here.
"""
from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel


class VisionMLPDeep(MultimodalModel):
    def __init__(self, config: DatasetConfig, hidden_dim: int = 64, dropout: float = 0.3,
                 normalization: str = "train_zscore"):
        super().__init__(config)
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must be positive and 0 <= dropout < 1")
        if normalization not in ("train_zscore", "none"):
            raise ValueError("normalization must be train_zscore or none")
        self.feature_dim = hidden_dim
        self.normalization = normalization
        self.register_buffer("feature_mean", torch.zeros(config.vision_dim))
        self.register_buffer("feature_scale", torch.ones(config.vision_dim))
        self.register_buffer("normalizer_fitted", torch.tensor(normalization == "none"))
        self.register_buffer("normalizer_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("constant_features", torch.zeros(config.vision_dim, dtype=torch.bool))
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.vision_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.regressor = nn.Linear(hidden_dim, 1)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--vision-normalization", choices=["train_zscore", "none"], default="train_zscore")

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "VisionMLPDeep":
        return cls(config, args.hidden_dim, args.dropout, args.vision_normalization)

    def pool(self, batch: Batch) -> tuple[Tensor, Tensor]:
        vision, lengths = batch.vision, batch.vision_lengths
        if vision.ndim != 3 or vision.size(-1) != self.config.vision_dim or vision.size(1) == 0:
            raise ValueError(f"Expected vision (B,T,{self.config.vision_dim}) with T > 0")
        lengths = lengths.to(device=vision.device)
        if lengths.shape != (vision.size(0),) or not torch.isfinite(lengths).all():
            raise ValueError("Expected one finite visual length per sample")
        if (lengths < 0).any() or (lengths > vision.size(1)).any():
            raise ValueError("Visual lengths must be within [0,T]")
        if lengths.is_floating_point() and (lengths != lengths.floor()).any():
            raise ValueError("Visual lengths must be integers")
        mask = torch.arange(vision.size(1), device=vision.device)[None, :] < lengths[:, None]
        clean = torch.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.where(mask.unsqueeze(-1), clean, 0.0).sum(1) / lengths.clamp_min(1).unsqueeze(-1)
        return pooled, lengths > 0

    @torch.no_grad()
    def fit_normalizer(self, train_loader) -> None:
        """Fit once from nonempty training examples; preserve RNG and saved stats."""
        if self.normalizer_fitted.item():
            return
        total = torch.zeros(self.config.vision_dim, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        with torch.random.fork_rng(devices=[]):
            for batch in train_loader:
                pooled, nonempty = self.pool(batch)
                values = pooled[nonempty].detach().cpu().double()
                total += values.sum(0)
                squares += values.square().sum(0)
                count += len(values)
        if not count:
            raise ValueError("Cannot fit visual normalization without nonempty training samples")
        mean = total / count
        scale = (squares / count - mean.square()).clamp_min(0).sqrt()
        constant = scale < 1e-6
        scale = torch.where(constant, 1.0, scale)
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_scale.copy_(scale.to(self.feature_scale))
        self.normalizer_samples.fill_(count)
        self.constant_features.copy_(constant.to(self.constant_features))
        self.normalizer_fitted.fill_(True)

    def encode(self, batch: Batch) -> Tensor:
        """Return (B, feature_dim) trainable features for a future fusion model."""
        if not self.normalizer_fitted.item():
            raise RuntimeError("Fit visual normalization on the training loader before prediction")
        pooled, nonempty = self.pool(batch)
        representation = self.encoder((pooled - self.feature_mean) / self.feature_scale)
        return torch.where(nonempty.unsqueeze(-1), representation, 0.0)

    def forward(self, batch: Batch) -> Tensor:
        return self.regressor(self.encode(batch)).squeeze(-1)

    def training_routine(self, loaders, device, args) -> dict[str, float]:
        self.fit_normalizer(loaders[0])
        return super().training_routine(loaders, device, args)
