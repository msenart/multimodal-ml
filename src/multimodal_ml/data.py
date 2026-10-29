from __future__ import annotations

import pickle
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
    per dataset, so samples need no further padding to be batched. audio_lengths/
    vision_lengths record each sample's true (unpadded) length, so a model can mask
    out the padding itself (e.g. when mean-pooling over time).
    """

    def __init__(self, config: DatasetConfig, split: Split):
        split_data = _load_split(config, split)
        self.text = np.asarray(split_data["text"], dtype=np.float32)
        self.audio = np.asarray(split_data["audio"], dtype=np.float32)
        self.vision = np.asarray(split_data["vision"], dtype=np.float32)
        self.labels = np.asarray(split_data["regression_labels"], dtype=np.float32)
        self.ids = [str(i) for i in split_data["id"]]

        n = len(self.labels)
        self.audio_lengths = np.asarray(
            split_data.get("audio_lengths", [self.audio.shape[1]] * n), dtype=np.int64
        )
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
        labels=torch.tensor([s["label"] for s in samples], dtype=torch.float32),
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
