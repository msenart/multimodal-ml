from __future__ import annotations

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel


def _masked_mean(x: Tensor, lengths: Tensor) -> Tensor:
    """Mean over the time dimension of (B, T, D), ignoring padding past `lengths`."""
    timesteps = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    mask = (timesteps < lengths.unsqueeze(1)).unsqueeze(-1).float()
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class ConcatMLP(MultimodalModel):
    """Mean-pool each modality over time, concat, 2-layer MLP -> scalar.

    Not a serious baseline — exists purely so the full pipeline (data loading,
    batching, forward/loss, metrics, checkpointing) can be exercised end-to-end
    before any real architecture has been written. See scripts/train_baseline.py.
    """

    def __init__(self, config: DatasetConfig, hidden_dim: int = 128):
        super().__init__(config)
        input_dim = config.text_dim + config.audio_dim + config.vision_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        text_repr = batch.text.mean(dim=1)
        audio_repr = _masked_mean(batch.audio, batch.audio_lengths)
        vision_repr = _masked_mean(batch.vision, batch.vision_lengths)
        features = torch.cat([text_repr, audio_repr, vision_repr], dim=-1)
        return self.net(features).squeeze(-1)
