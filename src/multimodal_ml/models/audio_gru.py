from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel


class AudioGRU(MultimodalModel):
    """Audio features -> bidirectional GRU -> overall sentiment regression.

    Only audio and its lengths enter the model. Packed sequences exclude trailing
    padding; an empty sequence receives an all-zero representation. The shared
    training routine uses overall regression_labels, including on CH-SIMS.
    """

    def __init__(
        self, config: DatasetConfig, hidden_dim: int = 64, dropout: float = 0.3
    ):
        super().__init__(config)
        self.feature_dim = 2 * hidden_dim
        if hidden_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim must be positive and 0 <= dropout < 1")
        self.encoder = nn.GRU(
            input_size=config.audio_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "AudioGRU":
        return cls(config, hidden_dim=args.hidden_dim, dropout=args.dropout)

    def encode(self, batch: Batch) -> Tensor:
        audio = batch.audio
        if audio.ndim != 3 or audio.size(-1) != self.config.audio_dim:
            raise ValueError(f"Expected audio (B,T,{self.config.audio_dim})")
        lengths = batch.audio_lengths.detach().to(device="cpu", dtype=torch.long)
        if lengths.shape != (audio.size(0),) or audio.size(1) == 0:
            raise ValueError("Expected one audio length per sample and T > 0")
        if (lengths < 0).any() or (lengths > audio.size(1)).any():
            raise ValueError("Audio lengths must be within [0, T]")
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        packed = pack_padded_sequence(
            audio, lengths.clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        features = features * (lengths > 0).to(features.device).unsqueeze(-1)
        return features

    def forward(self, batch: Batch) -> Tensor:
        return self.regressor(self.encode(batch)).squeeze(-1)
