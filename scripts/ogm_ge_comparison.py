"""Concatenation fusion on CMU-MOSI with and without OGM-GE (Peng et al., CVPR 2022).

Two optimizers are compared, each with its own protocol:
- adam: the project protocol (same as scripts/ogm_reproduction.py), so the
  baseline is exactly the fusion of that script;
- sgd: the paper's optimizer (momentum 0.9, weight decay 1e-4, lr 1e-3).

For each optimizer and seed: the fusion without modulation (baseline), with
OGM-GE for each alpha in ALPHAS, and with OGM alone (no noise) at the alpha
selected for OGM-GE. Alpha is selected on the mean validation MAE over seeds,
as in the paper ("determined from [0, 1] according to the validation set").
For the baseline and the selected OGM / OGM-GE models, each encoder is also
tested alone with a linear probe (same protocol as ogm_reproduction.py).

Writes results/ogm_ge/summary.json and prints Typst table rows.

    uv run python scripts/ogm_ge_comparison.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.seed import get_device, set_seed
from multimodal_ml.training import evaluate
from ogm_reproduction import ENCODER, METRICS, MODALITIES, SEEDS, aggregate, probe

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
OPTIMIZERS = {
    "adam": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "momentum": 0.0,
             "epochs": 20, "patience": 5, "batch_size": 16},
    "sgd": {"optimizer": "sgd", "lr": 1e-3, "weight_decay": 1e-4, "momentum": 0.9,
            "epochs": 100, "patience": 10, "batch_size": 16},
}


def run(opt: str, ogm: str, alpha: float, seed: int, checkpoint_dir: Path, device, with_probes: bool = True) -> dict:
    """Train one fusion; returns test/val metrics, encoder probes and per-epoch OGM coefficients."""
    config, training = get_dataset_config("mosi"), OPTIMIZERS[opt]
    set_seed(seed)
    model = ConcatFusion(config, hidden_dim=ENCODER["hidden_dim"], dropout=ENCODER["dropout"], pooling="mean",
                         num_layers=ENCODER["num_layers"], ogm=ogm, ogm_alpha=alpha).to(device)
    loaders = make_dataloaders(config, training["batch_size"], transform=model.transform)
    name = f"concat_fusion_{opt}_{ogm}_{alpha}"
    args = argparse.Namespace(dataset="mosi", model=name, seed=seed, checkpoint_dir=checkpoint_dir, **training)
    out = {"test": model.training_routine(loaders, device, args),
           "val": evaluate(model, loaders[1], device, desc="validation")}
    if with_probes:
        out["probe"] = {m: probe(model.encoders[m], loaders, seed, device, config) for m in MODALITIES}
    epochs = sorted({e["epoch"] for e in model.ogm_log})
    out["coeff_per_epoch"] = {m: [float(np.mean([e["coeff"][m] for e in model.ogm_log if e["epoch"] == ep]))
                                  for ep in epochs] for m in MODALITIES}
    out["ratio_per_epoch"] = {m: [float(np.mean([e["ratio"][m] for e in model.ogm_log if e["epoch"] == ep]))
                                  for ep in epochs] for m in MODALITIES}
    return out


def summarize(runs: list[dict]) -> dict:
    out = {"test": aggregate([r["test"] for r in runs]), "val": aggregate([r["val"] for r in runs])}
    if "probe" in runs[0]:
        out["probe"] = {m: aggregate([r["probe"][m] for r in runs]) for m in MODALITIES}
    if runs[0]["coeff_per_epoch"]["text"]:
        out["mean_coeff"] = {m: float(np.mean([np.mean(r["coeff_per_epoch"][m]) for r in runs])) for m in MODALITIES}
        out["mean_ratio"] = {m: float(np.mean([np.mean(r["ratio_per_epoch"][m]) for r in runs])) for m in MODALITIES}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ogm_ge/summary.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/ogm_ge")
    parser.add_argument("--optimizers", nargs="+", choices=list(OPTIMIZERS), default=list(OPTIMIZERS))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto: CUDA when available, else CPU")
    args = parser.parse_args()
    device = get_device(args.device)
    print(f"device: {device}", flush=True)

    results = {}
    for opt in args.optimizers:
        ckpt = args.checkpoint_dir / opt
        res = {"baseline": [run(opt, "none", 0.0, s, ckpt, device, True) for s in SEEDS]}
        sweep = {a: [run(opt, "ogm-ge", a, s, ckpt, device, True) for s in SEEDS] for a in ALPHAS}
        val_mae = {a: float(np.mean([r["val"]["mae"] for r in rs])) for a, rs in sweep.items()}
        best = min(val_mae, key=val_mae.get)
        print(f"[{opt}] OGM-GE val MAE per alpha: {val_mae} -> alpha={best}", flush=True)
        res["ogm_ge"] = sweep[best]
        res["ogm"] = [run(opt, "ogm", best, s, ckpt, device, True) for s in SEEDS]
        results[opt] = {
            "protocol": OPTIMIZERS[opt], "selected_alpha": best, "sweep_val_mae": val_mae,
            "sweep": {a: summarize(rs) for a, rs in sweep.items()},
            "aggregate": {k: summarize(v) for k, v in res.items()}, "runs": res,
        }

    summary = {
        "task": "CMU-MOSI regression: concatenation fusion with and without OGM-GE",
        "device": device.type, "seeds": list(SEEDS), "alphas": list(ALPHAS), "encoder": ENCODER,
        "ogm": "score = sum_i exp(-(c_m,i - y_i)^2 / 2), sigma = 1; rho_m = s_m / mean of the other scores; "
               "k_m = 1 - tanh(alpha rho_m) if rho_m > 1 else 1; GE noise N(0, std(grad)^2) per encoder tensor",
        "selection": "alpha with the lowest mean validation MAE over seeds (OGM-GE); OGM uses the same alpha",
        "units": "fractions (accuracy, F1, corr) and label units (MAE); std over seeds with ddof=1",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    fmt = lambda a, m: f"[{a[m]['mean'] * 100:.2f} ± {a[m]['std'] * 100:.2f}]"
    for opt, r in results.items():
        print(f"\n== {opt} (alpha={r['selected_alpha']})")
        for k in ("baseline", "ogm", "ogm_ge"):
            agg = r["aggregate"][k]
            print(f"  {k}: " + ", ".join(fmt(agg["test"], m) for m in METRICS)
                  + "  shares " + ", ".join(f"{m}={agg['test'][m + '_share']['mean'] * 100:.1f}" for m in MODALITIES))
            for m in MODALITIES:
                print(f"    probe {m}: " + ", ".join(fmt(agg["probe"][m], x) for x in ("acc2_has0", "f1_has0", "corr")))
            if "mean_coeff" in agg:
                print(f"    mean k: {agg['mean_coeff']}  mean rho: {agg['mean_ratio']}")


if __name__ == "__main__":
    main()
