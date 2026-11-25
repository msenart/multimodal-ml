"""A simple audio-only comparison model; no claim of a published reproduction."""
from __future__ import annotations

from multimodal_ml.models.sequence_mlp import SequenceMLP


class AudioMLP(SequenceMLP):
    """Audio features -> pooling over real frames (--pooling) -> MLP -> regression.

    Same encoder as VisionMLP (see SequenceMLP), applied to audio.
    """

    modality = "audio"
