"""A minimal differentiable fusion head, not an OGM implementation."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
