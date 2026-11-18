"""Train and validate a standalone vision-only regressor from local feature files.

No remote repository access, uploads, or dataset changes are performed. Cached
length-masked pooled features accelerate this MLP; raw-sequence predictions are
checked against the cache. The other modalities never enter the training loader.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import pickle
import re
import time

import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from multimodal_ml.checkpoint import load_checkpoint
from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch
from multimodal_ml.metrics import compute_regression_metrics
from multimodal_ml.models.vision_mlp import VisionMLP
from multimodal_ml.reference_metrics import compute_mmsa_metrics
from multimodal_ml.seed import set_seed
from check_audio_data import OFFICIAL_SHA256

ROOT = Path(__file__).resolve().parents[1]


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class VisionData(Dataset):
    vision: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    ids: list[str]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.vision[index], self.lengths[index], self.labels[index], self.ids[index]


def collate(samples):
    vision, lengths, labels, ids = zip(*samples)
    n = len(samples)
    # Empty placeholders let the shared Batch.to() work; they contain no text/audio.
    return Batch(torch.empty(n, 0, 0), torch.empty(n, 0, 0), torch.stack(vision),
                 torch.zeros(n, dtype=torch.long), torch.stack(lengths), torch.stack(labels), list(ids))


def loader(dataset, shuffle=False, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def load_local_data(name):
    config = get_dataset_config(name)
    if not config.pkl_path.is_file():
        raise FileNotFoundError(f"Missing local dataset: {config.pkl_path}")
    with config.pkl_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != OFFICIAL_SHA256[name]:
        raise ValueError(f"{name}: unexpected dataset SHA-256; confirm the feature version")
    with config.pkl_path.open("rb") as stream:
        raw = pickle.load(stream)
    splits, examples, fingerprints = {}, {}, {}
    audit = {"sha256": digest, "vision_length_policy": "declared, unchanged", "splits": {}}
    pooler = VisionMLP(config, normalization="none")
    for split in ("train", "valid", "test"):
        source = raw[split]
        vision = np.asarray(source["vision"], dtype=np.float32)
        labels = np.asarray(source["regression_labels"], dtype=np.float32).reshape(-1)
        supplied = np.asarray(source["vision_lengths"]).reshape(-1)
        if vision.ndim != 3 or vision.shape[0] != len(labels) or vision.shape[2] != config.vision_dim:
            raise ValueError(f"{name}/{split}: unexpected visual feature shape")
        if supplied.shape != (len(labels),) or not np.isfinite(supplied).all() or (supplied != np.floor(supplied)).any():
            raise ValueError(f"{name}/{split}: invalid visual lengths")
        if (supplied < 0).any() or (supplied > vision.shape[1]).any() or not np.isfinite(labels).all():
            raise ValueError(f"{name}/{split}: invalid lengths or labels")
        if (labels < config.label_low - 1e-6).any() or (labels > config.label_high + 1e-6).any():
            raise ValueError(f"{name}/{split}: labels outside sentiment scale")
        invalid = int((~np.isfinite(vision)).sum())
        if invalid:
            vision = np.nan_to_num(vision, nan=0.0, posinf=0.0, neginf=0.0)
        lengths = supplied.astype(np.int64)
        ids = [str(value) for value in source["id"]]
        if len(ids) != len(labels):
            raise ValueError(f"{name}/{split}: mismatched sample IDs")
        observed = np.where(np.any(vision != 0, axis=-1), np.arange(vision.shape[1])[None, :] + 1, 0).max(1)
        pooled = []
        for start in range(0, len(labels), 64):
            n = min(64, len(labels) - start)
            batch = Batch(None, None, torch.from_numpy(vision[start:start+n]), None,
                          torch.from_numpy(lengths[start:start+n]), None, [])
            pooled.append(pooler.pool(batch)[0])
        tensor = torch.cat(pooled)
        splits[split] = VisionData(tensor.unsqueeze(1), torch.from_numpy((lengths > 0).astype(np.int64)),
                                   torch.from_numpy(labels.copy()), ids)
        count = min(8, len(labels))
        examples[split] = VisionData(torch.from_numpy(vision[:count].copy()), torch.from_numpy(lengths[:count].copy()),
                                     torch.from_numpy(labels[:count].copy()), ids[:count])
        hashes = {}
        for index in range(len(labels)):
            key = hashlib.sha256(vision[index].tobytes()).hexdigest()
            hashes.setdefault(key, []).append(index)
        fingerprints[split] = hashes
        audit["splits"][split] = {"samples": len(labels), "shape": list(vision.shape),
                                   "label_range": [float(labels.min()), float(labels.max())],
                                   "length_range": [int(lengths.min()), int(lengths.max())],
                                   "non_finite_values_replaced": invalid, "empty": int((lengths == 0).sum()),
                                   "nonzero_after_declared_length_samples": int((observed > lengths).sum())}
    audit["id_overlap"] = {a + "-" + b: len(set(splits[a].ids) & set(splits[b].ids))
                           for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    duplicate_test = []
    for key in fingerprints["train"].keys() & fingerprints["test"].keys():
        for i in fingerprints["train"][key]:
            for j in fingerprints["test"][key]:
                duplicate_test.append({"train_index": i, "test_index": j,
                                       "train_id": splits["train"].ids[i], "test_id": splits["test"].ids[j],
                                       "train_label": float(splits["train"].labels[i]), "test_label": float(splits["test"].labels[j]),
                                       "all_zero_visual_features": bool(np.count_nonzero(raw["train"]["vision"][i]) == 0),
                                       "text_features_equal": bool(np.array_equal(raw["train"]["text"][i], raw["test"]["text"][j])),
                                       "audio_features_equal": bool(np.array_equal(raw["train"]["audio"][i], raw["test"]["audio"][j]))})
    audit["exact_visual_train_test_feature_matches"] = duplicate_test
    del raw, vision, pooler
    return config, splits, examples, audit


@torch.no_grad()
def predictions(model, dataset):
    model.eval()
    return np.concatenate([model(batch).cpu().numpy() for batch in loader(dataset)])


def scores(prediction, labels, config):
    # The primary MAE is independently calculated, not taken from the logger.
    common = compute_regression_metrics(prediction, labels, config)
    independent = float(np.abs(prediction.astype(np.float64) - labels.astype(np.float64)).mean())
    assert abs(common["mae"] - independent) < 1e-10
    return {"common": common, "dataset_specific": compute_mmsa_metrics(prediction, labels, config.name)}


def train_run(config, splits, examples, directory, seed, epochs):
    set_seed(seed)
    model = VisionMLP(config)
    args = argparse.Namespace(dataset=config.name, model="vision_mlp", epochs=epochs,
                              batch_size=32, lr=0.001, patience=10, seed=seed, device="cpu",
                              hidden_dim=64, dropout=0.3, vision_normalization="train_zscore",
                              checkpoint_dir=directory, audio_length_policy="declared")
    directory.mkdir(parents=True, exist_ok=True)
    loaders = (loader(splits["train"], True), loader(splits["valid"]), loader(splits["test"]))
    with (directory / "console.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        start = time.perf_counter()
        returned = model.training_routine(loaders, torch.device("cpu"), args)
    checkpoint = directory / f"{config.name}_vision_mlp_best.pt"
    meta = load_checkpoint(checkpoint, model)
    result = {"seed": seed, "best_epoch": meta["epoch"] + 1, "best_valid_mae": meta["metric_value"],
              "checkpoint": str(checkpoint.relative_to(ROOT).as_posix()), "seconds": time.perf_counter() - start,
              "normalizer_training_samples": int(model.normalizer_samples),
              "normalizer_constant_features": int(model.constant_features.sum()),
              "parameters": sum(p.numel() for p in model.parameters()), "metrics": {}}
    logs = sorted((directory / "logs").glob("*.log"))
    text = logs[-1].read_text(encoding="utf-8")
    epochs_seen = re.findall(r"\| INFO \| epoch (\d+) \|", text)
    result["epochs_run"] = int(epochs_seen[-1])
    for split in ("valid", "test"):
        values = predictions(model, splits[split])
        target = splits[split].labels.numpy()
        result["metrics"][split] = scores(values, target, config)
        np.savez_compressed(directory / f"{split}_predictions.npz", predictions=values, labels=target,
                            ids=np.asarray(splits[split].ids))
        count = len(examples[split])
        cached = VisionData(splits[split].vision[:count], splits[split].lengths[:count],
                            splits[split].labels[:count], splits[split].ids[:count])
        original_predictions = predictions(model, examples[split])
        cached_predictions = predictions(model, cached)
        np.testing.assert_allclose(original_predictions, cached_predictions, rtol=1e-4, atol=1e-5)
        result[f"{split}_raw_vs_cached_max_error"] = float(np.max(np.abs(original_predictions - cached_predictions)))
    for key, expected in returned.items():
        measured = result["metrics"]["test"]["common"][key]
        assert (np.isnan(expected) and np.isnan(measured)) or abs(expected - measured) < 1e-8
    assert int(model.normalizer_samples) == int((splits["train"].lengths > 0).sum())
    save_json(directory / "result.json", result)
    print(f"{config.name}: seed={seed}, epochs={result['epochs_run']}, best={result['best_epoch']}, test_MAE={result['metrics']['test']['common']['mae']:.5f}", flush=True)
    return model, result


def constant_and_ridge(config, splits, directory):
    x = {k: ds.vision[:, 0].numpy().astype(np.float64) for k, ds in splits.items()}
    y = {k: ds.labels.numpy() for k, ds in splits.items()}
    mean, scale = x["train"].mean(0), x["train"].std(0)
    scale[scale < 1e-6] = 1
    x = {key: (values - mean) / scale for key, values in x.items()}
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        ridge = Ridge(alpha=alpha, solver="cholesky").fit(x["train"], y["train"])
        valid_mae = float(np.abs(ridge.predict(x["valid"]) - y["valid"]).mean())
        candidates.append((valid_mae, alpha, ridge))
    _, alpha, best = min(candidates, key=lambda item: item[0])
    median = float(np.median(y["train"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        constant_scores = scores(np.full(len(y["test"]), median), y["test"], config)
    constant_scores["common"]["corr"] = None
    constant_scores["dataset_specific"]["Corr"] = None
    result = {"constant": {"training_median": median, "test": constant_scores},
              "ridge": {"alpha_candidates_validation_mae": [{"alpha": a, "mae": v} for v, a, _ in candidates],
                        "selected_alpha": alpha, "selection_split": "valid only",
                        "test": scores(best.predict(x["test"]), y["test"], config)}}
    np.savez_compressed(directory / "ridge_test_predictions.npz", predictions=best.predict(x["test"]), labels=y["test"])
    save_json(directory / "simple_references.json", result)
    return result


def small_subset_fit(config, splits, directory):
    set_seed(42)
    train = splits["train"]
    indices = np.argsort(train.labels.numpy(), kind="stable")[np.linspace(0, len(train) - 1, 16, dtype=int)]
    batch = collate([train[int(i)] for i in indices])
    model = VisionMLP(config, dropout=0.0)
    model.fit_normalizer(loader(train))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = float(F.mse_loss(model(batch), batch.labels).detach())
    final = initial
    for step in range(400):
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch), batch.labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        with torch.no_grad():
            final = float(F.mse_loss(model(batch), batch.labels))
        if final <= 0.01 * initial:
            break
    result = {"diagnostic_only": True, "samples": 16, "dropout": 0.0, "lr": 0.003,
              "steps": step + 1, "initial_mse": initial, "final_mse": final,
              "relative_mse": final / initial, "passed_99_percent_reduction": final <= 0.01 * initial}
    save_json(directory / "small_subset_fit.json", result)
    return result


def interventions(model, config, splits, audit, directory):
    valid = splits["valid"]
    target = valid.labels.numpy()
    original = predictions(model, valid)
    rng = np.random.default_rng(2026)
    shuffled_maes = []
    for _ in range(20):
        index = torch.from_numpy(rng.permutation(len(valid)))
        changed = VisionData(valid.vision[index], valid.lengths[index], valid.labels, valid.ids)
        shuffled_maes.append(float(np.abs(predictions(model, changed).astype(np.float64) - target).mean()))
    original_mae = float(np.abs(original.astype(np.float64) - target).mean())
    zero = VisionData(torch.zeros_like(valid.vision), valid.lengths, valid.labels, valid.ids)
    mean_input = VisionData(model.feature_mean[None, None, :].expand_as(valid.vision).clone(),
                            valid.lengths, valid.labels, valid.ids)
    result = {"split": "valid", "original_mae": original_mae, "permuted_maes": shuffled_maes,
              "permuted_mean_mae": float(np.mean(shuffled_maes)),
              "permutations_worse": sum(value > original_mae for value in shuffled_maes),
              "zero_vision_mae": float(np.abs(predictions(model, zero).astype(np.float64) - target).mean()),
              "training_mean_vision_mae": float(np.abs(predictions(model, mean_input).astype(np.float64) - target).mean()),
              "zero_vision_note": "Raw zero may be out of distribution; the training mean corresponds to zero normalized input",
              "note": "Diagnostic, not an independent significance test; validation also selected epochs"}
    excluded = sorted({pair["test_index"] for pair in audit["exact_visual_train_test_feature_matches"]})
    if excluded:
        test = splits["test"]
        predicted = predictions(model, test)
        keep = np.ones(len(test), dtype=bool)
        keep[excluded] = False
        result["feature_match_exclusion_sensitivity"] = {"excluded_indices": excluded, "remaining_test_samples": int(keep.sum()),
                                            "test_without_duplicates": scores(predicted[keep], test.labels.numpy()[keep], config)}
    save_json(directory / "interventions.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "checkpoints/vision_baseline" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"datasets": ["mosi", "sims"], "model": "vision_mlp", "target": "overall regression_labels",
                "input": "vision features and declared vision_lengths only", "normalization": "training-only pooled-feature z-score",
                "cache": "exact masked means; raw sequence inference checked", "seeds": [42, 43, 44],
                "hidden_dim": 64, "hidden_layers": 3, "dropout": 0.3, "loss": "MSE", "optimizer": "Adam",
                "lr": 0.001, "batch_size": 32, "epochs": 60, "patience": 10,
                "selection": "validation MAE only", "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
                "ridge_selection": "validation MAE only", "device": "cpu", "threads": 1,
                "network_access": False, "remote_writes": False}
    save_json(output / "protocol.json", protocol)
    print(f"Vision-only artifacts: {output}", flush=True)
    result = {"protocol": protocol, "datasets": {}}
    for name in protocol["datasets"]:
        config, splits, examples, audit = load_local_data(name)
        directory = output / name
        directory.mkdir(exist_ok=True)
        save_json(directory / "data_audit.json", audit)
        print(f"{name}: local data audit passed; {len(splits['train'])} training samples", flush=True)
        _, smoke = train_run(config, splits, examples, directory / "smoke", 42, 1)
        entry = {"audit": audit, "smoke": smoke, "references": constant_and_ridge(config, splits, directory),
                 "small_subset_fit": small_subset_fit(config, splits, directory), "runs": []}
        result["datasets"][name] = entry
        save_json(output / "results.json", result)
        for seed in protocol["seeds"]:
            model, run = train_run(config, splits, examples, directory / f"seed_{seed}", seed, 60)
            entry["runs"].append(run)
            if seed == 42:
                entry["interventions"] = interventions(model, config, splits, audit, directory)
            save_json(output / "results.json", result)
        for handler in list(logging.getLogger("multimodal_ml").handlers):
            logging.getLogger("multimodal_ml").removeHandler(handler)
            handler.close()
        del splits, examples, model
    print(f"Completed vision-only training and validation: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
