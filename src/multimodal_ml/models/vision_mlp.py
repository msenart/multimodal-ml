"""A basic vision-only comparison model: the same encoder as AudioMLP.

Inputs are precomputed per-frame facial features, not images. For the deeper
model with its own normaliser used in docs/vision, see VisionMLPDeep.
"""
from __future__ import annotations

from multimodal_ml.models.sequence_mlp import SequenceMLP


class VisionMLP(SequenceMLP):
    """Vision features -> pooling over real frames (--pooling) -> MLP -> regression.

    Same encoder as AudioMLP (see SequenceMLP), applied to vision. An empty clip
    (vision_length 0) pools to a zero vector.
    """

    modality = "vision"
