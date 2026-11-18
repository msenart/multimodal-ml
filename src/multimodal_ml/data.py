from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from multimodal_ml.config import DatasetConfig

Split = Literal["train", "valid", "test"]


class MissingDataFileError(FileNotFoundError):
    def __init__(self, pkl_path: Path):
        readme = Path(__file__).resolve().parents[2] / "data" / "README.md"
        super().__init__(
            f"Expected dataset features at {pkl_path}, but that file doesn't exist.\n"
            f"See {readme} for download instructions."
        )


@lru_cache(maxsize=4)
def _load_pickle(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_split(config: DatasetConfig, split: Split) -> dict:
    if not config.pkl_path.exists():
        raise MissingDataFileError(config.pkl_path)
    data = _load_pickle(config.pkl_path)
    return data[split]


@dataclass
class Batch:
    text: Tensor
    audio: Tensor
    vision: Tensor
    audio_lengths: Tensor
    vision_lengths: Tensor
    labels: Tensor
    ids: list[str]

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            text=self.text.to(device),
            audio=self.audio.to(device),
            vision=self.vision.to(device),
            audio_lengths=self.audio_lengths.to(device),
            vision_lengths=self.vision_lengths.to(device),
            labels=self.labels.to(device),
            ids=self.ids,
        )


class MultimodalDataset(Dataset):
    """One split ("train"/"valid"/"test") of an MMSA-format feature pickle.

    MMSA's preprocessing already pads/truncates text/audio/vision to a fixed length
    per dataset, so samples need no further padding to be batched. Supplied audio/
    vision lengths let a model ignore trailing padding (e.g. when mean-pooling).

    The default preserves supplied lengths. The opt-in 'observed' policy extends
    them to retain nonzero frames beyond a supplied length; it is custom to this
    project, not a reproduction of MMSA's loader.
    """

    def __init__(self, config: DatasetConfig, split: Split):
        split_data = _load_split(config, split)
        self.text = np.asarray(split_data["text"], dtype=np.float32)
        self.audio = np.asarray(split_data["audio"], dtype=np.float32)
        self.vision = np.asarray(split_data["vision"], dtype=np.float32)
        self.labels = np.asarray(split_data["regression_labels"], dtype=np.float32)
        # MMSA files may store scalar targets as (N, 1). Keeping that shape
        # would broadcast MSE against (N,) predictions into an (N, N) loss.
        if self.labels.ndim == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels[:, 0]
        if self.labels.ndim != 1 or not np.isfinite(self.labels).all():
            raise ValueError(f"{split}: regression_labels must be finite scalar targets")
        self.ids = [str(i) for i in split_data["id"]]

        n = len(self.labels)
        if (
            self.audio.ndim != 3
            or self.audio.shape[0] != n
            or self.audio.shape[1] == 0
            or self.audio.shape[2] != config.audio_dim
        ):
            raise ValueError(f"{split}: expected audio (N,T,{config.audio_dim}), got {self.audio.shape}")
        if len(self.text) != n or len(self.vision) != n or len(self.ids) != n:
            raise ValueError(f"{split}: modalities, ids and labels have different sample counts")
        invalid = ~np.isfinite(self.audio)
        if invalid.any():
            warnings.warn(
                f"{split}: replacing {int(invalid.sum())} non-finite audio values with zero",
                RuntimeWarning,
                stacklevel=2,
            )
            self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        raw_lengths = np.asarray(
            split_data.get("audio_lengths", [self.audio.shape[1]] * n)
        ).reshape(-1)
        if (
            raw_lengths.shape != (n,)
            or not np.isfinite(raw_lengths).all()
            or (raw_lengths < 0).any()
            or (raw_lengths > self.audio.shape[1]).any()
            or (raw_lengths != np.floor(raw_lengths)).any()
        ):
            raise ValueError(f"{split}: audio_lengths must contain N integers within [0,T]")
        self.audio_lengths = np.asarray(
            raw_lengths, dtype=np.int64
        )
        # Some official MOSI samples contain signal after audio_lengths. Keep
        # the supplied length as a minimum, and include every observed frame.
        # Interior zero frames are preserved; only trailing padding is ignored.
        if config.audio_length_policy not in ("declared", "observed"):
            raise ValueError("audio_length_policy must be 'declared' or 'observed'")
        if config.audio_length_policy == "observed":
            nonzero_frames = np.any(self.audio != 0, axis=-1)
            observed_ends = np.where(
                nonzero_frames, np.arange(self.audio.shape[1])[None, :] + 1, 0
            ).max(axis=1)
            extended = observed_ends > self.audio_lengths
            if extended.any():
                warnings.warn(
                    f"{split}: audio lengths extended for {int(extended.sum())} samples "
                    "to retain nonzero frames beyond the supplied lengths",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.audio_lengths = np.maximum(self.audio_lengths, observed_ends)
        self.vision_lengths = np.asarray(
            split_data.get("vision_lengths", [self.vision.shape[1]] * n), dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "audio_length": self.audio_lengths[idx],
            "vision_length": self.vision_lengths[idx],
            "label": self.labels[idx],
            "id": self.ids[idx],
        }


def collate_fn(samples: list[dict]) -> Batch:
    return Batch(
        text=torch.from_numpy(np.stack([s["text"] for s in samples])),
        audio=torch.from_numpy(np.stack([s["audio"] for s in samples])),
        vision=torch.from_numpy(np.stack([s["vision"] for s in samples])),
        audio_lengths=torch.tensor([s["audio_length"] for s in samples], dtype=torch.long),
        vision_lengths=torch.tensor([s["vision_length"] for s in samples], dtype=torch.long),
        labels=torch.from_numpy(np.asarray([s["label"] for s in samples], dtype=np.float32)),
        ids=[s["id"] for s in samples],
    )


def make_dataloaders(
    config: DatasetConfig, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split, shuffle in (("train", True), ("valid", False), ("test", False)):
        dataset = MultimodalDataset(config, split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        )
    train_loader, valid_loader, test_loader = loaders
    return train_loader, valid_loader, test_loader
