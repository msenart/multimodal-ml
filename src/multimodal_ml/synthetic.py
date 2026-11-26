"""Synthetic data for the OGM-GE control: one easy modality, one slow modality, one useless modality.

Two scenarios, fixed before the OGM-GE runs (see report/ogm_ge.typ). In both,
"text" is the easy (linear) modality and has many pure-noise dimensions, so that
a large encoder can memorise the small training set; "vision" carries its
information through a non-linear code (slower to learn); "audio" is pure noise.

- "independent": y = s_T + 0.7 s_V + noise. Text sees s_T, vision sees s_V:
  the fusion needs both.
- "redundant": y = s + noise. Text sees a noisy copy of s, vision a clean
  non-linear code of s, so vision is the better modality: the question is
  whether the easy text slows its learning down.

Features are already standardised: no train-fitted transform is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from torch.utils.data import DataLoader, Dataset

from multimodal_ml.config import DatasetConfig, get_dataset_config
from multimodal_ml.data import collate_fn

AUDIO_DIM, VISION_DIM = 4, 8


@dataclass(frozen=True)
class Scenario:
    text_dim: int      # 1 informative dimension + (text_dim - 1) noise dimensions
    text_noise: float  # noise on the informative text dimension
    vision_freq: float  # base frequency of the sin/cos code of the vision signal
    vision_weight: float  # weight of s_V in the label ("independent" only)
    redundant: bool
    n_train: int


SCENARIOS = {
    "independent": Scenario(text_dim=512, text_noise=0.3, vision_freq=1.0, vision_weight=0.7,
                            redundant=False, n_train=300),
    "redundant": Scenario(text_dim=4096, text_noise=0.4, vision_freq=1.5, vision_weight=0.0,
                          redundant=True, n_train=200),
}
N_VALID, N_TEST, LABEL_NOISE = 500, 2000, 0.3


def synthetic_config(scenario: str) -> DatasetConfig:
    return replace(get_dataset_config("mosi"), name=f"synthetic-{scenario}", pkl_path=None,
                   text_dim=SCENARIOS[scenario].text_dim, audio_dim=AUDIO_DIM, vision_dim=VISION_DIM,
                   text_len=1, audio_len=1, vision_len=1)


def _arrays(sc: Scenario, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    s_t = rng.standard_normal(n)
    s_v = s_t if sc.redundant else rng.standard_normal(n)
    text = rng.standard_normal((n, sc.text_dim))
    text[:, 0] = s_t + sc.text_noise * rng.standard_normal(n)
    freqs = sc.vision_freq * np.arange(1, VISION_DIM // 2 + 1)
    code = np.concatenate([np.sin(freqs * s_v[:, None]), np.cos(freqs * s_v[:, None])], axis=1)
    rotation = np.linalg.qr(np.random.default_rng(0).standard_normal((VISION_DIM, VISION_DIM)))[0]
    signal = s_t if sc.redundant else s_t + sc.vision_weight * s_v
    return {"text": text, "audio": rng.standard_normal((n, AUDIO_DIM)), "vision": code @ rotation * np.sqrt(2),
            "labels": signal + LABEL_NOISE * rng.standard_normal(n)}


class _Split(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.a = {k: v.astype(np.float32) for k, v in arrays.items()}

    def __len__(self) -> int:
        return len(self.a["labels"])

    def __getitem__(self, i: int) -> dict:
        a = self.a
        return {"text": a["text"][i][None], "audio": a["audio"][i][None], "vision": a["vision"][i][None],
                "audio_length": 1, "vision_length": 1, "label": a["labels"][i], "id": str(i)}


def make_loaders(scenario: str, seed: int, batch_size: int = 16):
    """(train, valid, test) loaders; the data depend on `seed`, the shuffling on torch's global RNG."""
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    splits = [_Split(_arrays(sc, n, rng)) for n in (sc.n_train, N_VALID, N_TEST)]
    return tuple(DataLoader(s, batch_size=batch_size, shuffle=i == 0, collate_fn=collate_fn)
                 for i, s in enumerate(splits))
