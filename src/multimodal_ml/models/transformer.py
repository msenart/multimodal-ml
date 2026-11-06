from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import *

def _padding_mask(x: Tensor, lengths: Tensor | None) -> Tensor:
    """(B, T) bool mask, True where a timestep is padding (past `lengths`)."""
    timesteps = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    if lengths is None:
        return torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
    return timesteps >= lengths.unsqueeze(1)


class MMTransformer(MultimodalModel):
    """Per-modality projection + learned positional embedding (sized from
    config.*_len), all tokens concatenated along time into one transformer
    encoder with padding masks, then masked mean-pooled into a regression head."""

    def __init__(self, config: DatasetConfig, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__(config)
        dims = {"text": config.text_dim, "audio": config.audio_dim, "vision": config.vision_dim}
        lens = {"text": config.text_len, "audio": config.audio_len, "vision": config.vision_len}
        self.proj = nn.ModuleDict({m: nn.Linear(d, hidden_dim) for m, d in dims.items()})
        self.pos = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, lens[m], hidden_dim)) for m in dims}
        )
        for p in self.pos.values():
            nn.init.normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-heads", type=int, default=4)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "MMTransformer":
        return cls(config, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads)

    def forward(self, batch: Batch) -> Tensor:
        inputs = {
            "text": (batch.text, None),  # text has no true lengths: all timesteps are real
            "audio": (batch.audio, batch.audio_lengths),
            "vision": (batch.vision, batch.vision_lengths),
        }
        tokens, masks = [], []
        for m, (x, lengths) in inputs.items():
            tokens.append(self.proj[m](x) + self.pos[m][:, : x.size(1)])
            masks.append(_padding_mask(x, lengths))
        x = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)
    
    def training_routine(self, loaders: "Loaders", device: torch.device, args: argparse.Namespace) -> Tensor:
        from multimodal_ml.training import mmtransformer_routine
        
        return mmtransformer_routine(self, loaders, device, args)