from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import *
from multimodal_ml.seed import get_device

def _padding_mask(x: Tensor, lengths: Tensor | None) -> Tensor:
    """(B, T) bool mask, True where a timestep is padding (past `lengths`)."""
    timesteps = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    if lengths is None:
        return torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
    return timesteps >= lengths.unsqueeze(1)


class SeqEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2, stride=2)
        self.proj = nn.Linear(128 * 3, emb_dim)

    @staticmethod
    def _mask(lengths, T):
        return (torch.arange(T, device=lengths.device)[None] < lengths[:, None]).unsqueeze(1)  # (B,1,T)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        B, T, _ = x.shape
        if T == 0:
            return self.proj(x.new_zeros(B, 256))

        m = self._mask(lengths, T)
        h = x.transpose(1, 2) * m
        h = torch.relu(self.conv1(h)) * m
        h = torch.relu(self.conv2(h))

        # adaptive mask, since the output will be shrunk by 2
        l = (lengths + 1) // 2
        m2 = self._mask(l, h.size(-1))
        h = h * m2

        mean = h.sum(-1) / l.clamp(min=1)[:, None].to(h.dtype)
        n = l.clamp(min=1)[:, None].to(h.dtype)
        diff = (h - mean[..., None])*m2
        std = torch.sqrt(diff.pow(2).sum(-1) / n + 1e-5)
        mx = h.amax(-1)
        return self.proj(torch.cat([mean, mx, std], 1))

class MMTransformer(MultimodalModel):
    """Per-modality projection + learned positional embedding (sized from
    config.*_len), all tokens concatenated along time into one transformer
    encoder with padding masks, then masked mean-pooled into a regression head."""

    def __init__(self, config: DatasetConfig, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__(config)
        device = get_device()
        # dims = {"text": config.text_dim, "audio": config.audio_dim, "vision": config.vision_dim}
        # lens = {"text": config.text_len, "audio": config.audio_len, "vision": config.vision_len}
        # self.proj = nn.ModuleDict({m: nn.Linear(d, hidden_dim) for m, d in dims.items()})
        # self.pos = nn.ParameterDict(
        #     {m: nn.Parameter(torch.zeros(1, lens[m], hidden_dim)) for m in dims}
        # )
        # for p in self.pos.values():
        #     nn.init.normal_(p, std=0.02)
        # layer = nn.TransformerEncoderLayer(
        #     2*hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        # )
        self.modality_emb = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)  # « je suis l'audio » / « je suis la vision »
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)        # token de résumé
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=4 * hidden_dim,
            batch_first=True, norm_first=True,
        )       
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden_dim, 1)
        self.audio_encoder = SeqEncoder(config.audio_dim, emb_dim=hidden_dim).to(device)
        self.vision_encoder = SeqEncoder(config.vision_dim, emb_dim=hidden_dim).to(device)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-heads", type=int, default=4)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "MMTransformer":
        return cls(config, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads)

    def forward(self, batch: Batch) -> Tensor:
        # inputs = {
        #     "text": (batch.text, None),  # text has no true lengths: all timesteps are real
        #     "audio": (batch.audio, batch.audio_lengths),
        #     "vision": (batch.vision, batch.vision_lengths),
        # }
        # tokens, masks = [], []
        # for m, (x, lengths) in inputs.items():
        #     tokens.append(self.proj[m](x) + self.pos[m][:, : x.size(1)])
        #     masks.append(_padding_mask(x, lengths))
        # print(batch.vision.shape)
        # x = torch.cat(tokens, dim=1)
        # pad = torch.cat(masks, dim=1)
        # x = self.encoder(x, src_key_padding_mask=pad)
        # keep = (~pad).unsqueeze(-1).float()
        # pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        # return self.head(pooled).squeeze(-1)

        audio = self.audio_encoder(batch.audio, batch.audio_lengths)
        vision = self.vision_encoder(batch.vision, batch.vision_lengths)
        B = audio.size(0)
        
        tokens = torch.stack([audio, vision], dim=1) + self.modality_emb
        x = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        missing = torch.stack([batch.audio_lengths == 0,
                            batch.vision_lengths == 0], dim=1)
        pad = torch.cat([missing.new_zeros(B, 1), missing], dim=1)    

        out = self.transformer(x, src_key_padding_mask=pad)
        return self.head(out[:, 0]).squeeze(-1)

    
    def training_routine(self, loaders: "Loaders", device: torch.device, args: argparse.Namespace) -> Tensor:
        from multimodal_ml.training import mmtransformer_routine
        
        return mmtransformer_routine(self, loaders, device, args)