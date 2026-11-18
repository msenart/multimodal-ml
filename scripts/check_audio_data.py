"""Audit MMSA audio features before training; never modifies the dataset files."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import MultimodalDataset, _load_pickle

OFFICIAL_SHA256 = {
    "mosi": "78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524",
    "sims": "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=["mosi", "sims"], default=["mosi", "sims"])
    parser.add_argument("--verify-hash", action="store_true", help="Require the exact MMSA reference files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio-length-policy", choices=["declared", "observed"], default="declared")
    args = parser.parse_args()
    configs = [replace(get_dataset_config(name), audio_length_policy=args.audio_length_policy) for name in args.dataset]
    missing = [config.pkl_path for config in configs if not config.pkl_path.is_file()]
    if missing:
        print("Missing real data files. Download these before training:")
        for path in missing:
            print(f"  {path}")
        print("Instructions: docs/audio/README.md")
        return 1

    report = {}
    for config in configs:
        try:
            with config.pkl_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if args.verify_hash and digest != OFFICIAL_SHA256[config.name]:
                raise ValueError("SHA-256 differs from the official MMSA reference; confirm the data version")
            raw = _load_pickle(config.pkl_path)
            splits = {}
            for split in ("train", "valid", "test"):
                dataset = MultimodalDataset(config, split)
                if len(dataset) == 0:
                    raise ValueError(f"{split}: empty split")
                if (dataset.labels < config.label_low - 1e-6).any() or (
                    dataset.labels > config.label_high + 1e-6
                ).any():
                    raise ValueError(f"{split}: labels outside the configured sentiment scale")
                audio = np.asarray(raw[split]["audio"])
                original_lengths = np.asarray(
                    raw[split].get("audio_lengths", [audio.shape[1]] * len(dataset))
                ).reshape(-1)
                splits[split] = {
                    "samples": len(dataset),
                    "audio_shape": list(audio.shape),
                    "original_label_shape": list(np.shape(raw[split]["regression_labels"])),
                    "label_shape": list(dataset.labels.shape),
                    "label_range": [float(dataset.labels.min()), float(dataset.labels.max())],
                    "non_finite_audio_values": int((~np.isfinite(audio)).sum()),
                    "length_range": [int(dataset.audio_lengths.min()), int(dataset.audio_lengths.max())],
                    "original_length_range": [int(original_lengths.min()), int(original_lengths.max())],
                    "lengths_extended_samples": int((dataset.audio_lengths > original_lengths).sum()),
                    "empty_audio_samples": int((dataset.audio_lengths == 0).sum()),
                    "lengths_supplied": "audio_lengths" in raw[split],
                }
                if "audio_lengths" not in raw[split]:
                    print(f"WARNING: {config.name}/{split} has no audio_lengths; full T is assumed")
            report[config.name] = {"file": str(config.pkl_path), "sha256": digest, "splits": splits}
            print(f"{config.name}: OK\n{json.dumps(report[config.name], indent=2)}")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print(f"{config.name}: data check failed: {exc}")
            return 1
        finally:
            _load_pickle.cache_clear()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
