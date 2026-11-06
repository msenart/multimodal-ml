"""Feature transforms applied to a dataset's arrays, owned by the model.

A model declares which transform its inputs need via `MultimodalModel.transform`
(default: `Standardize`); `__main__` passes it to the dataloaders, which fit it on
the train split and hand it to each `MultimodalDataset`. Subclass `FeatureTransform`
and assign it in your model's `__init__` to customise (`self.transform = None`
disables it).
"""

from __future__ import annotations

import numpy as np

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]  # (text, audio, vision), each (N, T, D)


def padding_mask(lengths: np.ndarray, time_steps: int) -> np.ndarray:
    """(N, T) bool, True where a timestep is padding."""
    return np.arange(time_steps)[None, :] >= lengths[:, None]


class FeatureTransform:
    """Contract: `fit` once on the train split's raw arrays, then call on any split.
    `__call__` must return new/updated arrays and keep padded timesteps at zero.
    Text has no true lengths (all timesteps are real)."""

    def fit(
        self, text: np.ndarray, audio: np.ndarray, vision: np.ndarray,
        audio_lengths: np.ndarray, vision_lengths: np.ndarray,
    ) -> "FeatureTransform":
        return self

    def __call__(
        self, text: np.ndarray, audio: np.ndarray, vision: np.ndarray,
        audio_lengths: np.ndarray, vision_lengths: np.ndarray,
    ) -> Arrays:
        return text, audio, vision


class Standardize(FeatureTransform):
    """Per-modality, per-feature standardisation with train-split statistics,
    clipped to +-clip_sigma.

    Some datasets ship raw features with extreme outliers (SIMS vision reaches
    ~2.8e7 while typical values are ~1), which overflow the backward pass and turn
    the weights into NaN. Stats use only real (unpadded) timesteps, on a row
    subsample, winsorised first so a few outliers can't inflate the std.
    """

    def __init__(self, clip_sigma: float = 5.0, max_rows: int = 50_000, winsor_pct=(0.1, 99.9)):
        self.clip_sigma = clip_sigma
        self.max_rows = max_rows
        self.winsor_pct = winsor_pct
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def fit(self, text, audio, vision, audio_lengths, vision_lengths) -> "Standardize":
        rng = np.random.default_rng(0)
        self.stats = {}
        for name, x, lengths in (
            ("text", text, None), ("audio", audio, audio_lengths), ("vision", vision, vision_lengths)
        ):
            rows = x.reshape(-1, x.shape[-1]) if lengths is None else x[~padding_mask(lengths, x.shape[1])]
            if len(rows) > self.max_rows:
                rows = rows[rng.choice(len(rows), self.max_rows, replace=False)]
            low, high = np.percentile(rows, self.winsor_pct, axis=0)
            rows = np.clip(rows, low, high)
            std = np.maximum(rows.std(axis=0), 1e-6)  # constant features: avoid /0
            self.stats[name] = (rows.mean(axis=0).astype(np.float32), std.astype(np.float32))
        return self

    def __call__(self, text, audio, vision, audio_lengths, vision_lengths) -> Arrays:
        if self.stats is None:
            raise RuntimeError("Standardize must be fit() on the train split before use.")
        out = []
        for name, x, lengths in (
            ("text", text, None), ("audio", audio, audio_lengths), ("vision", vision, vision_lengths)
        ):
            mean, std = self.stats[name]
            x = np.clip((x - mean) / std, -self.clip_sigma, self.clip_sigma).astype(np.float32)
            if lengths is not None:  # keep padding exactly zero
                x[padding_mask(lengths, x.shape[1])] = 0.0
            out.append(x)
        return tuple(out)
