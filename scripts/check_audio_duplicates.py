"""Inspect train/test audio duplicates and evaluate saved predictions without retraining."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from multimodal_ml.config import get_dataset_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset", choices=["mosi", "sims"], default="sims")
    args = parser.parse_args()
    base = args.directory.resolve()
    with get_dataset_config(args.dataset).pkl_path.open("rb") as source:
        data = pickle.load(source)
    lookup = {}
    for split in ("train", "test"):
        raw = data[split]
        audio = np.asarray(raw["audio"], dtype=np.float32)
        groups = defaultdict(list)
        for index in range(len(audio)):
            groups[hashlib.sha256(audio[index].tobytes()).hexdigest()].append(index)
        lookup[split] = groups
    matches, excluded = [], set()
    for digest in lookup["train"].keys() & lookup["test"].keys():
        for i in lookup["train"][digest]:
            for j in lookup["test"][digest]:
                train, test = data["train"], data["test"]
                matches.append({
                    "train_index": i, "test_index": j,
                    "train_id": str(train["id"][i]), "test_id": str(test["id"][j]),
                    "train_label": float(np.asarray(train["regression_labels"][i]).reshape(-1)[0]),
                    "test_label": float(np.asarray(test["regression_labels"][j]).reshape(-1)[0]),
                    "full_audio_equal": bool(np.array_equal(train["audio"][i], test["audio"][j])),
                    "text_features_equal": bool(np.array_equal(train["text"][i], test["text"][j])),
                    "vision_features_equal": bool(np.array_equal(train["vision"][i], test["vision"][j])),
                })
                excluded.add(j)
    result = {"dataset": args.dataset, "matches": matches, "test_indices_excluded": sorted(excluded),
              "test_samples_total": len(data["test"]["audio"]),
              "note": "Sensitivity only; official split, checkpoints and model selection remain unchanged", "predictions": []}
    files = [base / args.dataset / "original_predictions.npz"] + sorted((base / args.dataset).glob("*/test_predictions.npz"))
    for file in files:
        with np.load(file) as values:
            predictions, labels = values["predictions"], values["labels"]
        keep = np.ones(len(labels), dtype=bool)
        keep[list(excluded)] = False
        if not keep.any():
            raise ValueError("No test samples remain after excluding duplicates")
        result["predictions"].append({
            "source": str(file.relative_to(ROOT)),
            "original_mae": float(np.abs(predictions.astype("float64") - labels).mean()),
            "excluded_mae": float(np.abs(predictions[keep].astype("float64") - labels[keep]).mean()),
            "remaining_samples": int(keep.sum()),
        })
    path = base / args.dataset / "duplicate_audio_sensitivity.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
