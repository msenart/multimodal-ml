from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    pkl_path: Path
    text_dim: int
    audio_dim: int
    vision_dim: int
    label_low: float
    label_high: float
    has_acc7: bool
    has_unimodal_labels: bool


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "mosi": DatasetConfig(
        name="mosi",
        pkl_path=PROJECT_ROOT / "data" / "MOSI" / "Processed" / "unaligned_50.pkl",
        text_dim=768,
        audio_dim=5,
        vision_dim=20,
        label_low=-3.0,
        label_high=3.0,
        has_acc7=True,
        has_unimodal_labels=False,
    ),
    "sims": DatasetConfig(
        name="sims",
        pkl_path=PROJECT_ROOT / "data" / "SIMS" / "Processed" / "unaligned_39.pkl",
        text_dim=768,
        audio_dim=33,
        vision_dim=709,
        label_low=-1.0,
        label_high=1.0,
        has_acc7=False,
        has_unimodal_labels=True,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Valid options: {valid}") from None
