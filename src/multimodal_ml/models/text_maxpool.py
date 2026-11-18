
from __future__ import annotations

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel


def masked_max_text(x: Tensor) -> Tensor:
    """Max-pool text features while ignoring zero-vector padding."""
    mask = x.abs().sum(dim=-1) > 1e-8

    # Exclude padding before computing the maximum.
    masked_x = x.masked_fill(
        ~mask.unsqueeze(-1),
        torch.finfo(x.dtype).min,
    )

    pooled = masked_x.max(dim=1).values

    # Handle samples containing only padding.
    has_valid = mask.any(dim=1, keepdim=True)
    pooled = torch.where(has_valid, pooled, torch.zeros_like(pooled))

    return pooled


class TextMaxPoolMLP(MultimodalModel):
    """Text-only sentiment regression using masked max pooling and MLP."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.net = nn.Sequential(
            nn.Linear(config.text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        text_repr = masked_max_text(batch.text)
        return self.net(text_repr).squeeze(-1)
