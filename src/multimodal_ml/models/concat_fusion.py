"""Concatenation fusion of per-modality encoders, with optional OGM / OGM-GE.

OGM-GE (Peng et al., CVPR 2022) slows down the encoder of the dominant modality
during joint training. See `ConcatFusion` for how it is adapted to regression
and to more than two modalities.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
OGM_MODES = ("none", "ogm", "ogm-ge")
OPTIMIZERS = ("adam", "sgd")


class ConcatFusion(MultimodalModel):
    """Per-modality encoders fused by a shared ConcatRegressionFusion head.

    Takes a Batch like every other pipeline model. The encoder for each modality
    is chosen by name (see TEXT_ENCODERS / AUDIO_ENCODERS / VISION_ENCODERS);
    "none" leaves that modality out of the model entirely. Each encoder's own
    regressor is unused; only its `encode` output enters the head.

    On-the-fly gradient modulation (`ogm`): "none" trains normally; "ogm" scales
    the gradient of each encoder by k_m before the optimizer step; "ogm-ge" also
    adds Gaussian noise to it (generalization enhancement). At each training step:

    - the head's additive part c_m = W_m phi_m + b/M (see `contributions`) is the
      uni-modal prediction of modality m, as in the paper (b/2 for M=2);
    - the score of m is s_m = sum_i exp(-(c_m,i - y_i)^2 / (2 sigma^2)), the
      Gaussian likelihood of the true label: the regression counterpart of the
      paper's softmax probability of the true class. sigma = (label range) / 6,
      i.e. 1 on CMU-MOSI;
    - rho_m = s_m / mean of the other modalities' scores (for M=2 this is the
      paper's ratio), and k_m = 1 - tanh(alpha * rho_m) if rho_m > 1, else 1;
    - every encoder parameter gets grad <- k_m * grad, and with "ogm-ge" also
      + N(0, std(grad)^2), the std taken over the tensor's unmodulated gradient
      as in the official code.

    Modulation is active for 0-indexed epochs in [ogm_start, ogm_end) (ogm_end
    None: until the end). The head is never modulated. Per-step rho/k values are
    appended to `ogm_log`.
    """

    def __init__(self, config: DatasetConfig, text: str = "text-mlp", audio: str = "audio-mlp",
                 vision: str = "vision-mlp", hidden_dim: int = 64, dropout: float = 0.3,
                 vision_normalization: str = "train_zscore", pooling: str = "mean", num_layers: int = 1,
                 ogm: str = "none", ogm_alpha: float = 0.5, ogm_start: int = 0, ogm_end: int | None = None):
        super().__init__(config)
        if ogm not in OGM_MODES:
            raise ValueError(f"ogm must be one of {list(OGM_MODES)}")
        if ogm_alpha < 0:
            raise ValueError("ogm_alpha must be >= 0")
        self.ogm, self.ogm_alpha, self.ogm_start, self.ogm_end = ogm, ogm_alpha, ogm_start, ogm_end
        self.ogm_sigma = (config.label_high - config.label_low) / 6
        self.ogm_log: list[dict] = []
        self._train_features: dict[str, Tensor] | None = None
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
        parser.add_argument("--ogm", choices=OGM_MODES, default="none",
                            help="Gradient modulation: none, ogm (modulation only) or ogm-ge (+ noise)")
        parser.add_argument("--ogm-alpha", type=float, default=0.5, help="OGM strength alpha")
        parser.add_argument("--ogm-start", type=int, default=0, help="First modulated epoch (0-indexed)")
        parser.add_argument("--ogm-end", type=int, default=None,
                            help="Modulation stops before this epoch (default: never)")
        parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam",
                            help="adam (project default) or sgd (the OGM-GE paper)")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument("--weight-decay", type=float, default=0.0)

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "ConcatFusion":
        return cls(config, args.text, args.audio, args.vision, args.hidden_dim, args.dropout,
                   args.vision_normalization, args.pooling, args.num_layers,
                   args.ogm, args.ogm_alpha, args.ogm_start, args.ogm_end)

    def encode(self, batch: Batch) -> dict[str, Tensor]:
        return {name: enc.encode(batch) for name, enc in self.encoders.items()}

    def forward(self, batch: Batch) -> Tensor:
        features = self.encode(batch)
        if self.ogm != "none" and self.training:
            self._train_features = {name: f.detach() for name, f in features.items()}
        return self.head(features)

    def configure_optimizer(self, args: argparse.Namespace) -> torch.optim.Optimizer:
        name = getattr(args, "optimizer", "adam")
        weight_decay = getattr(args, "weight_decay", 0.0)
        if name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=args.lr, momentum=getattr(args, "momentum", 0.9),
                                   weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=weight_decay)
        raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}")

    @torch.no_grad()
    def ogm_coefficients(self, features: dict[str, Tensor], labels: Tensor) -> tuple[dict, dict]:
        """(rho_m, k_m) for each modality from one batch; see the class docstring."""
        parts = self.head.contributions(features)
        scores = {m: torch.exp(-(c - labels).square() / (2 * self.ogm_sigma ** 2)).sum().item()
                  for m, c in parts.items()}
        ratios, coeffs = {}, {}
        for m, s in scores.items():
            others = [v for n, v in scores.items() if n != m]
            ratios[m] = s / max(sum(others) / len(others), 1e-8)
            coeffs[m] = 1 - float(np.tanh(self.ogm_alpha * ratios[m])) if ratios[m] > 1 else 1.0
        return ratios, coeffs

    def ogm_active(self, epoch: int) -> bool:
        return (self.ogm != "none" and len(self.encoders) > 1 and epoch >= self.ogm_start
                and (self.ogm_end is None or epoch < self.ogm_end))

    @torch.no_grad()
    def on_after_backward(self, batch: Batch, epoch: int) -> None:
        features, self._train_features = self._train_features, None
        if not self.ogm_active(epoch) or features is None:
            return
        ratios, coeffs = self.ogm_coefficients(features, batch.labels)
        self.ogm_log.append({"epoch": epoch, "ratio": ratios, "coeff": coeffs})
        for m, encoder in self.encoders.items():
            for p in encoder.parameters():
                if p.grad is None:
                    continue
                noise_std = p.grad.std().item() if p.grad.numel() > 1 else 0.0
                p.grad.mul_(coeffs[m])
                if self.ogm == "ogm-ge":
                    p.grad.add_(torch.randn_like(p.grad) * (noise_std + 1e-8))

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
        return super().training_routine(loaders, device, args)
