"""Concatenation fusion with a linear head; optional OGM-GE training (see multimodal_ml/ogm_ge.py)."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from multimodal_ml import ogm_ge
from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.metrics import compute_contribution_metrics
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.sequence_mlp import POOLINGS, SequenceMLP
from multimodal_ml.models.text_mlp import TextMLP
from multimodal_ml.models.vision_mlp import VisionMLP
from multimodal_ml.models.vision_mlp_deep import VisionMLPDeep


class ConcatRegressionFusion(nn.Module):
    """Concatenate heterogeneous encoder features and apply a shared linear head.

    This mirrors the decomposable concat+linear choice in the official OGM code,
    extended here to an arbitrary number of regression feature branches. Its
    partial outputs are additive contributions, not calibrated unimodal models
    or a justified regression replacement for OGM's classification confidence.
    """

    def __init__(self, feature_dims: dict[str, int]):
        super().__init__()
        if not feature_dims or any(d < 1 for d in feature_dims.values()):
            raise ValueError("Each modality requires a positive feature dimension")
        self.feature_dims = dict(feature_dims)
        self.head = nn.Linear(sum(feature_dims.values()), 1)

    def _validate(self, features: dict[str, Tensor]) -> None:
        if features.keys() != self.feature_dims.keys():
            raise ValueError("Feature keys must match the configured modalities")
        batch_sizes = set()
        for name, dim in self.feature_dims.items():
            if features[name].ndim != 2 or features[name].size(1) != dim:
                raise ValueError(f"{name}: expected (B,{dim}) features")
            batch_sizes.add(features[name].size(0))
        if len(batch_sizes) != 1:
            raise ValueError("Modalities must contain the same aligned samples")

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        self._validate(features)
        return self.head(torch.cat([features[name] for name in self.feature_dims], dim=-1)).squeeze(-1)

    def contributions(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        self._validate(features)
        outputs, start = {}, 0
        for name, dim in self.feature_dims.items():
            outputs[name] = F.linear(features[name], self.head.weight[:, start:start + dim],
                                     self.head.bias / len(self.feature_dims)).squeeze(-1)
            start += dim
        return outputs


TEXT_ENCODERS = {"text-mlp": TextMLP}
AUDIO_ENCODERS = {"audio-mlp": AudioMLP, "audio-gru": AudioGRU}
# "vision-mlp" matches the text/audio MLPs; "vision-mlp-deep" is the 3-layer
# VisionMLPDeep with its own normaliser, as used for the results in docs/vision.
VISION_ENCODERS = {"vision-mlp": VisionMLP, "vision-mlp-deep": VisionMLPDeep}


class ConcatFusion(MultimodalModel):
    """Per-modality encoders fused by a shared ConcatRegressionFusion head.

    Takes a Batch like every other pipeline model. The encoder for each modality
    is chosen by name (see TEXT_ENCODERS / AUDIO_ENCODERS / VISION_ENCODERS);
    "none" leaves that modality out of the model entirely. Each encoder's own
    regressor is unused; only its `encode` output enters the head.
    """

    def __init__(self, config: DatasetConfig, text: str = "text-mlp", audio: str = "audio-mlp",
                 vision: str = "vision-mlp", hidden_dim: int = 64, dropout: float = 0.3,
                 vision_normalization: str = "train_zscore", pooling: str = "mean", num_layers: int = 1):
        super().__init__(config)
        if text != "none" and text not in TEXT_ENCODERS:
            raise ValueError(f"text must be one of {[*TEXT_ENCODERS, 'none']}")
        if audio != "none" and audio not in AUDIO_ENCODERS:
            raise ValueError(f"audio must be one of {[*AUDIO_ENCODERS, 'none']}")
        if vision != "none" and vision not in VISION_ENCODERS:
            raise ValueError(f"vision must be one of {[*VISION_ENCODERS, 'none']}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {list(POOLINGS)}")

        def build(cls):
            if issubclass(cls, SequenceMLP):  # audio-mlp / vision-mlp share the pooled encoder
                return cls(config, hidden_dim, dropout, pooling, num_layers)
            if cls is TextMLP:
                return cls(config, hidden_dim, dropout, num_layers)
            if cls is VisionMLPDeep:
                return cls(config, hidden_dim, dropout, vision_normalization)
            return cls(config, hidden_dim, dropout)

        encoders = {}
        if text != "none":
            encoders["text"] = build(TEXT_ENCODERS[text])
        if audio != "none":
            encoders["audio"] = build(AUDIO_ENCODERS[audio])
        if vision != "none":
            encoders["vision"] = build(VISION_ENCODERS[vision])
        if not encoders:
            raise ValueError("ConcatFusion needs at least one modality (text, audio and vision are all 'none')")
        self.encoders = nn.ModuleDict(encoders)
        self.head = ConcatRegressionFusion({name: enc.feature_dim for name, enc in self.encoders.items()})

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--text", choices=[*TEXT_ENCODERS, "none"], default="text-mlp",
                            help="Text encoder, or 'none' to leave text out")
        parser.add_argument("--audio", choices=[*AUDIO_ENCODERS, "none"], default="audio-mlp",
                            help="Audio encoder, or 'none' to leave audio out")
        parser.add_argument("--vision", choices=[*VISION_ENCODERS, "none"], default="vision-mlp",
                            help="Vision encoder, or 'none' to leave vision out")
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--vision-normalization", choices=["train_zscore", "none"], default="train_zscore",
                            help="Extra vision normaliser; only used by --vision vision-mlp-deep")
        parser.add_argument("--pooling", choices=list(POOLINGS), default="mean",
                            help="Pooling over time for audio-mlp and vision-mlp: mean, or mean+std+max")
        parser.add_argument("--num-layers", type=int, default=1,
                            help="Hidden blocks of text-mlp, audio-mlp and vision-mlp")
        ogm_ge.add_cli_args(parser)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "ConcatFusion":
        return cls(config, args.text, args.audio, args.vision, args.hidden_dim, args.dropout,
                   args.vision_normalization, args.pooling, args.num_layers)

    def encode(self, batch: Batch) -> dict[str, Tensor]:
        return {name: enc.encode(batch) for name, enc in self.encoders.items()}

    def forward(self, batch: Batch) -> Tensor:
        return self.head(self.encode(batch))

    def contributions(self, batch: Batch) -> dict[str, Tensor]:
        """Per-modality additive parts of `forward` (e.g. for OGM-style modulation)."""
        return self.head.contributions(self.encode(batch))

    @torch.no_grad()
    def diagnostic_metrics(self, loader, device) -> dict[str, float]:
        """Which modality dominates: see metrics.compute_contribution_metrics."""
        parts = {name: [] for name in self.encoders}
        labels = []
        for batch in loader:
            batch = batch.to(device)
            for name, c in self.contributions(batch).items():
                parts[name].append(c.cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
        return compute_contribution_metrics({n: np.concatenate(c) for n, c in parts.items()},
                                            np.concatenate(labels))

    def training_routine(self, loaders, device, args) -> dict[str, float]:
        for encoder in self.encoders.values():
            if hasattr(encoder, "fit_normalizer"):
                encoder.fit_normalizer(loaders[0])
        # --ogm none + --optimizer adam is the same computation as training.default_routine.
        metrics, self.training_history = ogm_ge.train(self, loaders, device, args)
        return metrics
