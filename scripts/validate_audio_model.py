"""Reproducible diagnostics for the existing AudioGRU, not a test-set tuning loop.

All settings are written to protocol.json before any new experiments. Original
checkpoints are read-only; every new result is stored in a separate directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch
from multimodal_ml.metrics import compute_regression_metrics
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.seed import set_seed
from check_audio_data import OFFICIAL_SHA256

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "a94e65d07fa1ae0d44e552390074b29b0898edfd"


def repo_path(value):
    """Read historical Windows-relative artifact paths on any platform."""
    return ROOT / Path(str(value).replace("\\", "/"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def independent_metrics(predictions, labels):
    """Compute common metrics without sklearn or the project's metric function."""
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    assert p.shape == y.shape and p.ndim == 1
    assert np.isfinite(p).all() and np.isfinite(y).all()
    pc, yc = p - p.mean(), y - y.mean()
    denominator = np.sqrt((pc * pc).sum() * (yc * yc).sum())
    result = {"mae": float(np.abs(p - y).mean()), "corr": float((pc * yc).sum() / denominator) if denominator else None}
    for mode in ("has0", "non0"):
        mask = np.ones(len(y), dtype=bool) if mode == "has0" else y != 0
        a = p[mask] >= 0 if mode == "has0" else p[mask] > 0
        b = y[mask] >= 0 if mode == "has0" else y[mask] > 0
        confusion = [[int(((b == actual) & (a == guessed)).sum()) for guessed in (False, True)] for actual in (False, True)]
        weighted_f1 = 0.0
        for cls in (0, 1):
            tp = confusion[cls][cls]
            support = sum(confusion[cls])
            denominator_f1 = 2 * tp + confusion[1 - cls][cls] + confusion[cls][1 - cls]
            weighted_f1 += (2 * tp / denominator_f1 if denominator_f1 else 0) * support / len(b)
        result[f"acc2_{mode}"] = float((a == b).mean())
        result[f"f1_{mode}"] = weighted_f1
        result[f"confusion_{mode}_rows_true_neg_pos"] = confusion
    return result


@dataclass
class AudioSplit(Dataset):
    audio: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    ids: list[str]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.audio[index], self.lengths[index], self.labels[index]


def collate(samples):
    audio, lengths, labels = zip(*samples)
    lengths = torch.stack(lengths)
    end = max(1, int(lengths.max()))
    return Batch(None, torch.stack([x[:end] for x in audio]), None, lengths, None, torch.stack(labels), [])


def load_audio(name):
    config = replace(get_dataset_config(name), audio_length_policy="observed")
    with config.pkl_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != OFFICIAL_SHA256[name]:
        raise ValueError(f"{name}: dataset hash does not match the reference feature file")
    with config.pkl_path.open("rb") as source:
        data = pickle.load(source)
    splits, audit = {}, {"sha256": digest, "splits": {}}
    for split in ("train", "valid", "test"):
        raw = data[split]
        audio = np.asarray(raw["audio"], dtype=np.float32)
        labels = np.asarray(raw["regression_labels"], dtype=np.float32).reshape(-1)
        declared = np.asarray(raw["audio_lengths"], dtype=np.int64).reshape(-1)
        assert audio.shape[0] == len(labels) == len(declared) == len(raw["id"])
        assert np.isfinite(audio).all() and np.isfinite(labels).all()
        assert ((declared >= 0) & (declared <= audio.shape[1])).all()
        observed = np.where(np.any(audio != 0, axis=-1), np.arange(audio.shape[1])[None, :] + 1, 0).max(1)
        lengths = np.maximum(declared, observed)
        ids = [str(value) for value in raw["id"]]
        splits[split] = AudioSplit(torch.from_numpy(audio.copy()), torch.from_numpy(lengths.copy()), torch.from_numpy(labels.copy()), ids)
        audit["splits"][split] = {
            "samples": len(labels), "duplicate_ids_within_split": len(ids) - len(set(ids)),
            "extended_lengths": int((observed > declared).sum()),
            "max_extension": int(np.maximum(0, observed - declared).max()),
            "declared_lengths": declared.tolist(),
        }
    audit["id_overlap"] = {a + "-" + b: len(set(splits[a].ids) & set(splits[b].ids)) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    fingerprints = {}
    for split, ds in splits.items():
        fingerprints[split] = {hashlib.sha256(ds.audio[i].numpy().tobytes()).hexdigest() for i in range(len(ds))}
    audit["exact_audio_overlap"] = {a + "-" + b: len(fingerprints[a] & fingerprints[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    train = splits["train"]
    mask = torch.arange(train.audio.shape[1])[None, :] < train.lengths[:, None]
    frames = train.audio[mask].double()
    mean, std = frames.mean(0).float(), frames.std(0, correction=0).float().clamp_min(1e-6)
    audit["training_feature_mean"] = mean.tolist()
    audit["training_feature_std"] = std.tolist()
    audit["training_feature_min"] = frames.min(0).values.tolist()
    audit["training_feature_max"] = frames.max(0).values.tolist()
    audit["normalizer_fit_split"] = "non-padding frames of the training split only"
    return config, splits, audit, mean, std


def normalized_splits(splits, mean, std):
    output = {}
    for key, ds in splits.items():
        values = (ds.audio - mean) / std
        mask = torch.arange(values.shape[1])[None, :] < ds.lengths[:, None]
        values = values * mask.unsqueeze(-1)
        output[key] = AudioSplit(values, ds.lengths, ds.labels, ds.ids)
    return output


@torch.no_grad()
def predict(model, split):
    model.eval()
    values = [model(batch).numpy() for batch in DataLoader(split, batch_size=32, collate_fn=collate)]
    return np.concatenate(values)


def reference_sources(output):
    directory = output / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_file = directory / "manifest.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    commit = REFERENCE_COMMIT
    files = {}
    for path in ("src/MMSA/data_loader.py", "src/MMSA/utils/metricsTop.py", "src/MMSA/models/subNets/FeatureNets.py"):
        url = f"https://raw.githubusercontent.com/thuiar/MMSA/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        filename = directory / Path(path).name
        filename.write_bytes(content)
        files[path] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {"repository": "thuiar/MMSA", "commit": commit, "files": files}
    save_json(manifest_file, manifest)
    return manifest


def check_checkpoint(config, splits, saved, output, reference):
    model = AudioGRU(config).eval()
    state = torch.load(repo_path(saved["checkpoint"]), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    predictions = predict(model, splits["test"])
    labels = splits["test"].labels.numpy()
    independently = independent_metrics(predictions, labels)
    shared = compute_regression_metrics(predictions, labels, config)
    for key in ("mae", "corr", "acc2_has0", "acc2_non0", "f1_has0", "f1_non0"):
        assert abs(independently[key] - shared[key]) < 1e-10, (key, independently[key], shared[key])
        assert abs(shared[key] - saved["test_metrics"][key]) < 1e-6, ("saved result mismatch", key)
    official = reference(torch.from_numpy(predictions.copy()), torch.from_numpy(labels.copy()))
    # Compare the independent audio-only loader with a sample from the shared loader.
    from multimodal_ml.data import MultimodalDataset, _load_pickle
    shared_dataset = MultimodalDataset(config, "test")
    assert np.array_equal(shared_dataset.audio, splits["test"].audio.numpy())
    assert np.array_equal(shared_dataset.audio_lengths, splits["test"].lengths.numpy())
    assert np.array_equal(shared_dataset.labels, labels)
    del shared_dataset
    _load_pickle.cache_clear()
    single_batch = collate([splits["test"][i] for i in range(4)])
    with torch.no_grad():
        batched = model(single_batch)
        singles = torch.cat([model(collate([splits["test"][i]])) for i in range(4)])
    assert torch.allclose(batched, singles, atol=1e-6, rtol=1e-5)
    train = splits["train"]
    mask = torch.arange(train.audio.shape[1])[None, :] < train.lengths[:, None]
    sample = train.audio[mask][::max(1, int(mask.sum()) // 4096)]
    preactivation = F.linear(sample, model.encoder.weight_ih_l0[:128], model.encoder.bias_ih_l0[:128]).detach()
    report = {"saved_metrics_reproduced": True, "independent_metric_check": True, "loader_check": True,
              "packed_batch_order_check": True, "shared_metrics": shared, "independent_metrics": independently,
              "official_mmsa_metrics": {k: float(v) for k, v in official.items()},
              "input_only_reset_update_preactivation_abs_gt8_fraction": float((preactivation.abs() > 8).float().mean()),
              "input_only_gate_note": "Diagnostic of input projection only, not the full recurrent gate activation"}
    np.savez_compressed(output / "original_predictions.npz", predictions=predictions, labels=labels, ids=np.asarray(splits["test"].ids))
    save_json(output / "checkpoint_audit.json", report)
    return model, report


def small_fit(config, splits, output, standardized):
    path = output / f"overfit_{'standardized' if standardized else 'raw'}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("diagnostic_version") == 2:
            return previous
    set_seed(42)
    ds = splits["train"]
    order = np.argsort(ds.labels.numpy(), kind="stable")
    indices = order[np.linspace(0, len(order) - 1, 16, dtype=int)]
    batch = collate([ds[int(i)] for i in indices])
    model = AudioGRU(config, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = float(F.mse_loss(model(batch), batch.labels).detach())
    start = time.perf_counter()
    first_gradient = None
    for step in range(600):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch), batch.labels)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None) ** 0.5
        assert np.isfinite(gradient)
        if first_gradient is None:
            first_gradient = gradient
        optimizer.step()
        # Check the parameters that will actually be reported, not the loss
        # from before the last optimizer update (which can overshoot).
        with torch.no_grad():
            updated_loss = float(F.mse_loss(model(batch), batch.labels))
        if updated_loss <= initial * 0.01:
            break
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        final = float(F.mse_loss(predictions, batch.labels))
    result = {"diagnostic_version": 2, "stopping_rule": "post-update MSE <= 1% of initial MSE",
              "diagnostic_only": True, "dropout": 0.0, "lr": 0.003, "samples": 16,
              "train_indices": indices.tolist(), "steps": step + 1, "initial_mse": initial,
              "final_mse": final, "final_mae": float((predictions - batch.labels).abs().mean()),
              "relative_mse": final / initial, "passed_99_percent_mse_reduction": final <= initial * 0.01,
              "first_gradient_norm": first_gradient, "seconds": time.perf_counter() - start}
    save_json(path, result)
    print(f"{config.name}: small-fit standardized={standardized}: {json.dumps(result)}", flush=True)
    return result


def ablations(model, config, splits, output, audit):
    path = output / "validation_ablations.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ds = splits["valid"]
    base_predictions = predict(model, ds)
    labels = ds.labels.numpy()
    base = independent_metrics(base_predictions, labels)
    rng = np.random.default_rng(2026)
    permutations = []
    for _ in range(20):
        index = torch.from_numpy(rng.permutation(len(ds)))
        shuffled = AudioSplit(ds.audio[index], ds.lengths[index], ds.labels, ds.ids)
        permutations.append(independent_metrics(predict(model, shuffled), labels))
    zeroed = AudioSplit(torch.zeros_like(ds.audio), ds.lengths, ds.labels, ds.ids)
    original_lengths = torch.tensor(audit["splits"]["valid"]["declared_lengths"], dtype=torch.long)
    declared = AudioSplit(ds.audio, original_lengths, ds.labels, ds.ids)
    declared_metrics = independent_metrics(predict(model, declared), labels)
    baseline_prediction = np.full(len(labels), float(splits["train"].labels.median()))
    # Exact numpy median, including even-sized training sets.
    baseline_prediction[:] = np.median(splits["train"].labels.numpy())
    improvement = np.abs(baseline_prediction - labels) - np.abs(base_predictions - labels)
    boot = improvement[rng.integers(0, len(labels), size=(2000, len(labels)))].mean(1)
    result = {"split": "valid only", "original": base, "permutations": permutations,
              "mean_shuffled_mae": float(np.mean([p["mae"] for p in permutations])),
              "shuffle_worse_count": sum(p["mae"] > base["mae"] for p in permutations),
              "zero_audio_same_lengths": independent_metrics(predict(model, zeroed), labels),
              "declared_length_evaluation_same_checkpoint": declared_metrics,
              "train_median_baseline": independent_metrics(baseline_prediction, labels),
              "mae_improvement_vs_train_median": float(improvement.mean()),
              "clip_bootstrap_95_percent_interval": np.quantile(boot, [.025, .975]).tolist(),
              "bootstrap_note": "Exploratory clip-level interval; not speaker-level inference or a multi-seed test"}
    save_json(path, result)
    print(f"{config.name}: ablations completed; validation original MAE {base['mae']:.5f}, shuffled mean {result['mean_shuffled_mae']:.5f}", flush=True)
    return result


def train_comparison(config, splits, model_name, seed, output, standardized=False):
    name = f"{model_name}_{'standardized' if standardized else 'raw'}_seed{seed}"
    directory = output / name
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = (AudioGRU if model_name == "audio_gru" else AudioMLP)(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(splits["train"], batch_size=32, shuffle=True, collate_fn=collate)
    best, bad_epochs, history = float("inf"), 0, []
    for epoch in range(60):
        start = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            prediction = model(batch)
            loss = F.mse_loss(prediction, batch.labels)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            total += float(loss.detach()) * len(batch.labels)
            seen += len(batch.labels)
        metrics = independent_metrics(predict(model, splits["valid"]), splits["valid"].labels.numpy())
        improved = metrics["mae"] < best
        if improved:
            best, bad_epochs = metrics["mae"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "metric_value": best, "model": model_name, "seed": seed,
                        "standardized": standardized}, directory / "best.pt")
        else:
            bad_epochs += 1
        row = {"epoch": epoch + 1, "train_mse": total / seen, "valid": metrics,
               "seconds": time.perf_counter() - start, "new_best": improved}
        history.append(row)
        save_json(directory / "history.json", history)
        print(f"{config.name} {name}: epoch {epoch+1}/60 train_mse={total/seen:.5f} valid_mae={metrics['mae']:.5f} best={best:.5f}", flush=True)
        if bad_epochs >= 10:
            break
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    result = {"model": model_name, "seed": seed, "standardized": standardized,
              "best_epoch": state["epoch"] + 1, "epochs_run": len(history), "best_valid_mae": best,
              "checkpoint": str((directory / "best.pt").relative_to(ROOT)),
              "parameters": sum(p.numel() for p in model.parameters())}
    for split in (("valid",) if standardized else ("valid", "test")):
        predictions = predict(model, splits[split])
        result[split] = independent_metrics(predictions, splits[split].labels.numpy())
        np.savez_compressed(directory / f"{split}_predictions.npz", predictions=predictions, labels=splits[split].labels.numpy())
    save_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original-results", type=Path,
                        help="Optional archived seed-42 manifest; otherwise train seed 42 on a fresh checkout")
    args = parser.parse_args()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "checkpoints/audio_validation" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": [42, 43, 44], "epochs": 60, "patience": 10, "batch_size": 32,
                "lr": 0.001, "hidden_dim": 64, "dropout": 0.3,
                "selection": "minimum validation MAE; test does not select any setting",
                "primary": "raw AudioGRU and AudioMLP on both datasets, seeds 42/43/44; reuse seed42 if archived artifacts exist",
                "diagnostics": "16-sample overfit raw/standardized; 20 validation audio permutations; zero audio; declared length sensitivity; standardization GRU seed42 validation only",
                "length_policy": "max(supplied length, observed nonzero endpoint); explicit custom policy, not MMSA reproduction",
                "normalization": "train valid-frame mean/std only; never fit on valid/test",
                "normalization_test_evaluation": False,
                "small_fit_stopping_rule": "v2: criterion evaluated after the optimizer step",
                "device": "cpu", "threads": 1,
                "original_model_unchanged": True}
    save_json(output / "protocol.json", protocol)
    print(f"Validation artifacts: {output}", flush=True)
    reference_manifest = reference_sources(output)
    official_class = runpy.run_path(str(output / "reference/metricsTop.py"))["MetricsTop"]
    archive_path = args.original_results or ROOT / "checkpoints/audio_results_seed42.json"
    if args.original_results and not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    saved_results = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.is_file() else None
    results = {"protocol": protocol, "reference": reference_manifest, "datasets": {}}
    for name in ("mosi", "sims"):
        directory = output / name
        directory.mkdir(exist_ok=True)
        config, splits, audit, mean, std = load_audio(name)
        archived = saved_results["datasets"][name] if saved_results else None
        reuse = archived is not None and repo_path(archived["checkpoint"]).is_file()
        if reuse:
            old_audit = json.loads(repo_path(archived["data_audit"]).read_text(encoding="utf-8"))[name]
            assert audit["sha256"] == old_audit["sha256"], "Dataset changed since the original experiment"
            audit["sha256_matches_original_experiment"] = True
            seed42 = archived
        else:
            fresh = train_comparison(config, splits, "audio_gru", 42, directory)
            seed42 = {**fresh, "test_metrics": fresh["test"]}
        save_json(directory / "data_audit.json", audit)
        reference = official_class("regression").getMetics(name)
        model, checkpoint_audit = check_checkpoint(config, splits, seed42, directory, reference)
        normalized = normalized_splits(splits, mean, std)
        save_json(directory / "train_only_normalizer.json", {"mean": mean.tolist(), "std": std.tolist()})
        dataset_result = {"audit": audit, "checkpoint": checkpoint_audit,
                          "small_fit_raw": small_fit(config, splits, directory, False),
                          "small_fit_standardized": small_fit(config, normalized, directory, True),
                          "ablations": ablations(model, config, splits, directory, audit), "runs": []}
        # Reuse an available original seed42 instead of selecting a better test run.
        dataset_result["runs"].append({"model": "audio_gru", "seed": 42, "standardized": False,
                                       "best_valid_mae": seed42["best_valid_mae"],
                                       "test": checkpoint_audit["independent_metrics"],
                                       "checkpoint": seed42["checkpoint"], "reused_original": reuse})
        results["datasets"][name] = dataset_result
        save_json(output / "results.json", results)
        for model_name, seeds in (("audio_mlp", (42, 43, 44)), ("audio_gru", (43, 44))):
            for seed in seeds:
                dataset_result["runs"].append(train_comparison(config, splits, model_name, seed, directory))
                save_json(output / "results.json", results)
        dataset_result["standardization_validation_diagnostic"] = train_comparison(config, normalized, "audio_gru", 42, directory, True)
        save_json(output / "results.json", results)
        del splits, normalized, model
    print(f"All diagnostics completed: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
