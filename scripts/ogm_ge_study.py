"""Run every training needed for the OGM-GE study (report/ogm_ge.typ).

Each run is saved to results/ogm_ge/runs/<name>.json and skipped if it already
exists, so the script can be stopped and resumed. Analyse with
scripts/ogm_ge_analysis.py.

    .venv/bin/python scripts/ogm_ge_study.py --workers 9

OGM runs use --ogm-ratio max. The earlier runs with --ogm-ratio mean (sensitivity analysis)
are kept in results/ogm_ge/runs_ratio_mean/.

Runs (10 paired seeds each; same encoders and training as scripts/ogm_reproduction.py):
- MOSI, SGD learning-rate selection for the baseline (the paper's optimizer);
- MOSI, baseline and OGM / OGM-GE for alpha in ALPHAS, with Adam and with SGD;
  every fusion run is followed by a linear probe of each encoder;
- MOSI, each modality trained alone (probes + predictions for the stacking ceiling);
- MOSI, linear probes of untrained (random) encoders;
- synthetic control (multimodal_ml/synthetic.py): same conditions, plus vision alone.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from multimodal_ml import ogm_ge
from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.probe import probe
from multimodal_ml.seed import set_seed
from multimodal_ml.synthetic import make_loaders, synthetic_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ogm_ge"
SEEDS = tuple(range(1111, 1121))
MODALITIES = ("text", "audio", "vision")
ALPHAS = (0.1, 0.3, 1.0, 3.0)
OGM_MODES = ("ogm", "ogm-ge")
ADAM_LR = 1e-3
SGD_LRS = (1e-3, 3e-3, 1e-2)  # 3e-2 diverges
# MOSI: as scripts/ogm_reproduction.py. Synthetic: no dropout and more epochs, so that the
# dominant text can memorise the training set (the situation OGM-GE targets).
SETUP = {
    "mosi": {"encoder": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.3},
             "training": {"epochs": 20, "patience": 5, "batch_size": 16}},
    "synthetic": {"encoder": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.0},
                  "training": {"epochs": 40, "patience": 10, "batch_size": 16}},
}
SYNTHETIC_LR = {"adam": 1e-3, "sgd": 1e-2}
SYNTHETIC_ALPHAS = (0.3, 1.0, 3.0)
DEVICE = torch.device("cpu")


def run_name(spec: dict) -> str:
    parts = [spec["data"], spec["kind"], spec.get("modality", "tav"), spec.get("optimizer", "-"),
             f"lr{spec.get('lr', 0):g}", spec.get("mode", "none"), f"a{spec.get('alpha', 0):g}", f"s{spec['seed']}"]
    return "_".join(parts)


def _setup(spec: dict) -> dict:
    return SETUP["mosi" if spec["data"] == "mosi" else "synthetic"]


def _build(spec: dict, modalities):
    config = get_dataset_config("mosi") if spec["data"] == "mosi" else synthetic_config(spec["data"])
    enc = _setup(spec)["encoder"]
    names = {"text": "text-mlp", "audio": "audio-mlp", "vision": "vision-mlp"}
    set_seed(spec["seed"])
    model = ConcatFusion(config, **{m: names[m] if m in modalities else "none" for m in MODALITIES}, **enc)
    if spec["data"] == "mosi":
        loaders = make_dataloaders(config, _setup(spec)["training"]["batch_size"], transform=model.transform)
    else:
        loaders = make_loaders(spec["data"], spec["seed"], _setup(spec)["training"]["batch_size"])
    return config, model, loaders


@torch.no_grad()
def _predictions(model, loader) -> list[float]:
    model.eval()
    return torch.cat([model(b.to(DEVICE)) for b in loader]).tolist()


def _labels(loader) -> list[float]:
    return torch.cat([b.labels for b in loader]).tolist()


def run(spec: dict) -> str:
    path = OUT / "runs" / f"{run_name(spec)}.json"
    if path.exists():
        return f"skip {path.name}"
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN predictions of diverged runs
    modalities = MODALITIES if spec["kind"] in ("fusion", "random") else (spec["modality"],)
    config, model, loaders = _build(spec, modalities)
    result = {"spec": spec}
    if spec["kind"] != "random":
        tr = _setup(spec)["training"]
        args = argparse.Namespace(dataset=spec["data"], model="concat_fusion", epochs=tr["epochs"],
                                  patience=tr["patience"], lr=spec["lr"], optimizer=spec["optimizer"],
                                  ogm_mode=spec.get("mode", "none"), ogm_alpha=spec.get("alpha", 0.0),
                                  ogm_ratio=spec.get("ratio", "max"))
        try:
            result["test"], result["history"] = ogm_ge.train(model, loaders, DEVICE, args, verbose=False)
        except RuntimeError as error:  # diverged at the first epoch
            result["error"] = str(error)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result) + "\n")
            return f"diverged {path.name}"
        result["best_val_mae"] = min(h["val_mae"] for h in result["history"])
        if spec["kind"] == "unimodal":
            result["predictions"] = {split: {"pred": _predictions(model, loaders[i]), "label": _labels(loaders[i])}
                                     for i, split in ((1, "valid"), (2, "test"))}
    result["probe"] = {m: probe(model.encoders[m], loaders, spec["seed"], DEVICE, config) for m in model.encoders}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result) + "\n")
    return f"done {path.name}"


def ogm_specs(data: str, optimizer: str, lr: float, alphas) -> list[dict]:
    specs = []
    for seed in SEEDS:
        base = {"data": data, "kind": "fusion", "optimizer": optimizer, "lr": lr, "seed": seed}
        specs.append({**base, "mode": "none", "alpha": 0.0})
        specs += [{**base, "mode": mode, "alpha": a, "ratio": "max"} for mode in OGM_MODES for a in alphas]
    return specs


def selected_sgd_lr() -> float:
    """Baseline SGD learning rate with the lowest mean best-validation MAE over the seeds."""
    scores = {}
    for lr in SGD_LRS:
        runs = [json.loads((OUT / "runs" / f"{run_name(s)}.json").read_text())
                for s in ogm_specs("mosi", "sgd", lr, ())]
        scores[lr] = float(np.mean([r.get("best_val_mae", np.inf) for r in runs]))
    best = min(scores, key=scores.get)
    (OUT / "sgd_lr_selection.json").write_text(json.dumps({"val_mae": scores, "selected": best}, indent=2) + "\n")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--part", choices=["all", "mosi", "synthetic"], default="all")
    args = parser.parse_args()

    first = []
    if args.part in ("all", "mosi"):
        first += [s for lr in SGD_LRS for s in ogm_specs("mosi", "sgd", lr, ())]
        first += ogm_specs("mosi", "adam", ADAM_LR, ALPHAS)
        first += [{"data": "mosi", "kind": "unimodal", "modality": m, "optimizer": "adam", "lr": ADAM_LR, "seed": s}
                  for m in MODALITIES for s in SEEDS]
        first += [{"data": "mosi", "kind": "random", "seed": s} for s in SEEDS]
    if args.part in ("all", "synthetic"):
        for scenario in ("independent", "redundant"):
            for opt, lr in SYNTHETIC_LR.items():
                first += ogm_specs(scenario, opt, lr, SYNTHETIC_ALPHAS)
                first += [{"data": scenario, "kind": "unimodal", "modality": "vision", "optimizer": opt, "lr": lr,
                           "seed": s} for s in SEEDS]
            first += [{"data": scenario, "kind": "random", "seed": s} for s in SEEDS]

    with Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, first), 1):
            print(f"[{i}/{len(first)}] {msg}", flush=True)
        if args.part in ("all", "mosi"):
            lr = selected_sgd_lr()
            print(f"selected SGD learning rate: {lr:g}", flush=True)
            second = ogm_specs("mosi", "sgd", lr, ALPHAS)
            for i, msg in enumerate(pool.imap_unordered(run, second), 1):
                print(f"[{i}/{len(second)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
