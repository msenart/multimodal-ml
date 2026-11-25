"""OGM-GE study on CMU-MOSI: controlled comparison, alpha ablation, diagnostics.

Every run uses the same OGMGE architecture (per-modality Linear+ReLU encoders,
linear fusion); only the training switches change. One JSON per run is written to
results/ogm_ge/runs/, so the script can be stopped and resumed. Then run
`aggregate.py` to build the tables and figures.

    uv run python experiments/ogm_ge/run_study.py --phase lr      # step 1
    uv run python experiments/ogm_ge/run_study.py --phase alpha   # step 2 (uses step 1)

Protocol (all choices are made on the validation set, never on the test set):
  1. lr: each variant is trained with every lr of its grid and every seed; the lr
     with the lowest mean validation MAE is kept. OGM variants use alpha = 0.5 here.
  2. alpha: the two OGM variants are retrained at their selected lr with each alpha;
     the alpha with the lowest mean validation MAE is kept.
Diagnostics are computed on the best checkpoint of every run:
  - masked evaluation: test metrics when some modalities are replaced by zeros
    (= the train mean, since features are standardised);
  - linear probes: a ridge regression fit on the frozen h^u of each modality
    (train split, ridge penalty chosen on validation), evaluated on test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_ml.config import get_dataset_config  # noqa: E402
from multimodal_ml.data import Batch, make_dataloaders  # noqa: E402
from multimodal_ml.metrics import compute_regression_metrics  # noqa: E402
from multimodal_ml.models.ogm_ge import MODALITIES, OGMGE  # noqa: E402
from multimodal_ml.seed import set_seed  # noqa: E402
from multimodal_ml.training import evaluate  # noqa: E402

DATASET = "mosi"
SEEDS = list(range(1111, 1121))  # 10 seeds
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
HIDDEN_DIM = 128
DEFAULT_ALPHA = 0.5
ALPHAS = [0.1, 0.3, 0.5, 0.8, 1.0]

# name -> (optimizer, modulation, ge)
VARIANTS = {
    "linear_adam": ("adam", False, False),
    "linear_sgd": ("sgd", False, False),
    "ge_only": ("sgd", False, True),
    "ogm_only": ("sgd", True, False),
    "ogm_ge": ("sgd", True, True),
}
LR_GRID = {"adam": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], "sgd": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}
OGM_VARIANTS = ("ogm_only", "ogm_ge")

OUT_DIR = ROOT / "results" / "ogm_ge"
RUNS_DIR = OUT_DIR / "runs"
CKPT_DIR = ROOT / "checkpoints" / "ogm_ge_study"

# Modalities kept at test time (the others are zeroed).
MASKS = {
    "TAV": ("text", "audio", "vision"),
    "T": ("text",),
    "A": ("audio",),
    "V": ("vision",),
    "AV": ("audio", "vision"),
}


def run_id(variant: str, lr: float, alpha: float | None, seed: int) -> str:
    a = f"_a{alpha:g}" if alpha is not None else ""
    return f"{variant}_lr{lr:g}{a}_s{seed}"


def mask_batch(batch: Batch, keep: tuple[str, ...]) -> Batch:
    z = torch.zeros_like
    return Batch(
        text=batch.text if "text" in keep else z(batch.text),
        audio=batch.audio if "audio" in keep else z(batch.audio),
        vision=batch.vision if "vision" in keep else z(batch.vision),
        audio_lengths=batch.audio_lengths,
        vision_lengths=batch.vision_lengths,
        labels=batch.labels,
        ids=batch.ids,
    )


@torch.no_grad()
def masked_eval(model: OGMGE, loader, keep: tuple[str, ...]) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        preds.append(model(mask_batch(batch, keep)).numpy())
        labels.append(batch.labels.numpy())
    return compute_regression_metrics(np.concatenate(preds), np.concatenate(labels), model.config)


@torch.no_grad()
def encoder_features(model: OGMGE, loader) -> tuple[dict[str, np.ndarray], np.ndarray]:
    model.eval()
    feats = {m: [] for m in MODALITIES}
    labels = []
    for batch in loader:
        h = model.encode(batch)
        for m in MODALITIES:
            feats[m].append(h[m].numpy())
        labels.append(batch.labels.numpy())
    return {m: np.concatenate(v) for m, v in feats.items()}, np.concatenate(labels)


def linear_probes(model: OGMGE, loaders) -> dict[str, dict[str, float]]:
    """Ridge regression on each frozen encoder output; penalty picked on validation MAE."""
    (ftr, ytr), (fva, yva), (fte, yte) = (encoder_features(model, l) for l in loaders)
    out = {}
    for m in MODALITIES:
        best = None
        for pen in (0.1, 1.0, 10.0, 100.0, 1000.0):
            reg = Ridge(alpha=pen).fit(ftr[m], ytr)
            mae = float(np.mean(np.abs(reg.predict(fva[m]) - yva)))
            if best is None or mae < best[0]:
                best = (mae, reg)
        out[m] = compute_regression_metrics(best[1].predict(fte[m]), yte, model.config)
    return out


def train_one(loaders, variant: str, lr: float, alpha: float, seed: int) -> dict:
    opt, modulation, ge = VARIANTS[variant]
    config = get_dataset_config(DATASET)
    set_seed(seed)
    model = OGMGE(config, hidden_dim=HIDDEN_DIM, alpha=alpha, modulation=modulation, ge=ge, optimizer=opt)
    rid = run_id(variant, lr, alpha if variant in OGM_VARIANTS else None, seed)
    args = SimpleNamespace(
        dataset=DATASET, model="ogm_ge", epochs=EPOCHS, batch_size=BATCH_SIZE, lr=lr,
        patience=PATIENCE, seed=seed, hidden_dim=HIDDEN_DIM, alpha=alpha,
        no_modulation=not modulation, no_ge=not ge, optimizer=opt,
        checkpoint_dir=CKPT_DIR / rid,
    )
    start = time.perf_counter()
    test = model.training_routine(loaders, torch.device("cpu"), args)
    finite = [h for h in model.history if np.isfinite(h["val"]["mae"])]
    diverged = not finite or not np.isfinite(test["mae"])
    result = {
        "id": rid, "variant": variant, "optimizer": opt, "modulation": modulation, "ge": ge,
        "lr": lr, "alpha": alpha if variant in OGM_VARIANTS else None, "seed": seed,
        "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM,
        "diverged": diverged, "history": model.history,
    }
    if diverged:  # training blew up (NaN): no usable checkpoint, excluded at selection
        result["val"] = {"mae": float("inf")}
        return result | {"seconds": time.perf_counter() - start}
    best = min(finite, key=lambda h: h["val"]["mae"])
    return result | {
        "best_epoch": best["epoch"], "val": best["val"], "test": test,
        "masked_test": {name: masked_eval(model, loaders[2], keep) for name, keep in MASKS.items()},
        "probe_test": linear_probes(model, loaders),
        "seconds": time.perf_counter() - start,
    }


def load_runs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))]


def selected_lr(runs: list[dict], variant: str) -> float:
    """lr with the lowest mean validation MAE (OGM variants: at the default alpha)."""
    scores: dict[float, list[float]] = {}
    for r in runs:
        if r["variant"] != variant:
            continue
        if variant in OGM_VARIANTS and r["alpha"] != DEFAULT_ALPHA:
            continue
        scores.setdefault(r["lr"], []).append(r["val"]["mae"])
    complete = {lr: v for lr, v in scores.items() if len(v) == len(SEEDS)}
    if len(complete) != len(LR_GRID[VARIANTS[variant][0]]):
        raise SystemExit(f"lr phase incomplete for {variant}: run --phase lr first")
    return min(complete, key=lambda lr: np.mean(complete[lr]))


def jobs_for(phase: str) -> list[tuple[str, float, float, int]]:
    jobs = []
    if phase == "lr":
        for variant, (opt, _, _) in VARIANTS.items():
            for lr in LR_GRID[opt]:
                for seed in SEEDS:
                    jobs.append((variant, lr, DEFAULT_ALPHA, seed))
    else:
        runs = load_runs()
        for variant in OGM_VARIANTS:
            lr = selected_lr(runs, variant)
            for alpha in ALPHAS:
                for seed in SEEDS:
                    jobs.append((variant, lr, alpha, seed))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["lr", "alpha"], required=True)
    p.add_argument("--shard", type=int, default=0, help="this worker's index")
    p.add_argument("--num-shards", type=int, default=1, help="number of parallel workers")
    a = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() // a.num_shards))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.phase)[a.shard :: a.num_shards]

    config = get_dataset_config(DATASET)
    # The Standardize transform is fit on the train split only; its statistics do not
    # depend on the seed, so the loaders are built once and shared by all runs.
    loaders = make_dataloaders(config, BATCH_SIZE, transform=OGMGE(config).transform)

    for i, (variant, lr, alpha, seed) in enumerate(jobs, 1):
        rid = run_id(variant, lr, alpha if variant in OGM_VARIANTS else None, seed)
        path = RUNS_DIR / f"{rid}.json"
        if path.exists():
            continue
        print(f"[shard {a.shard}] {i}/{len(jobs)} {rid}", flush=True)
        result = train_one(loaders, variant, lr, alpha, seed)
        path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
