"""Reproduce the imbalance observation of OGM-GE (Peng et al., CVPR 2022) on CMU-MOSI.

For each seed:
1. Uni-modal: text_mlp, audio_mlp and vision_mlp trained alone.
2. Joint: concat_fusion (the same three encoders + one linear head) trained from scratch.
3. Each encoder tested alone with a *linear probe* (the paper's protocol, footnote 1):
   freeze the encoder, train only a new linear regressor on its features. Done for
   the encoders trained alone and for the encoders trained inside concat_fusion.

All models use the shared default routine (Adam, MSE, early stopping on val MAE).
Writes results/ogm_reproduction/summary.json and prints Typst table rows.

    uv run python scripts/ogm_reproduction.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.metrics import compute_regression_metrics
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.models.text_mlp import TextMLP
from multimodal_ml.models.vision_mlp import VisionMLP
from multimodal_ml.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1111, 1112, 1113)
# Chosen with scripts/ogm_capacity_check.py: the lightest shared setting with which
# all three encoders fit the MOSI train split (train corr > 0.98 without dropout).
ENCODER = {"hidden_dim": 256, "num_layers": 3, "dropout": 0.3}
TRAINING = {"epochs": 20, "lr": 1e-3, "patience": 5, "batch_size": 16}
PROBE = {"epochs": 200, "lr": 1e-3, "patience": 10, "batch_size": 16}
MODALITIES = ("text", "audio", "vision")
METRICS = ("acc2_has0", "f1_has0", "corr", "mae")


def unimodal(modality: str, config):
    if modality == "text":
        return TextMLP(config, ENCODER["hidden_dim"], ENCODER["dropout"], ENCODER["num_layers"])
    cls = AudioMLP if modality == "audio" else VisionMLP
    return cls(config, ENCODER["hidden_dim"], ENCODER["dropout"], "mean", ENCODER["num_layers"])


def train(model, name: str, seed: int, checkpoint_dir: Path, device):
    """Train with the shared default routine; returns (test metrics, loaders)."""
    loaders = make_dataloaders(get_dataset_config("mosi"), TRAINING["batch_size"], transform=model.transform)
    args = argparse.Namespace(dataset="mosi", model=name, seed=seed, checkpoint_dir=checkpoint_dir, **TRAINING)
    return model.training_routine(loaders, device, args), loaders


@torch.no_grad()
def features(encoder, loader, device):
    encoder.eval()
    xs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        xs.append(encoder.encode(batch).cpu())
        ys.append(batch.labels.cpu())
    return torch.cat(xs), torch.cat(ys)


def probe(encoder, loaders, seed: int, device, config) -> dict[str, float]:
    """Freeze `encoder`; train only a new Linear(feature_dim, 1); test metrics of the best-val probe."""
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (features(encoder, l, device) for l in loaders)
    set_seed(seed)
    head = nn.Linear(x_tr.size(1), 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE["lr"])
    best, best_state, waited = float("inf"), None, 0
    for _ in range(PROBE["epochs"]):
        for idx in torch.randperm(len(x_tr)).split(PROBE["batch_size"]):
            loss = F.mse_loss(head(x_tr[idx]).squeeze(-1), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_mae = (head(x_va).squeeze(-1) - y_va).abs().mean().item()
        if val_mae < best:
            best, best_state, waited = val_mae, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PROBE["patience"]:
                break
    head.load_state_dict(best_state)
    with torch.no_grad():
        preds = head(x_te).squeeze(-1).numpy()
    return compute_regression_metrics(preds, y_te.numpy(), config)


def aggregate(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs], ddof=1))}
            for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ogm_reproduction/summary.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/ogm_reproduction")
    args = parser.parse_args()
    config = get_dataset_config("mosi")
    device = torch.device("cpu")
    runs = {k: [] for k in ["fusion", *[f"{m}_alone" for m in MODALITIES],
                            *[f"{m}_alone_probe" for m in MODALITIES], *[f"{m}_joint_probe" for m in MODALITIES]]}
    for seed in SEEDS:
        ckpt = args.checkpoint_dir / f"seed_{seed}"
        for m in MODALITIES:
            set_seed(seed)
            model = unimodal(m, config).to(device)
            metrics, loaders = train(model, f"{m}_mlp", seed, ckpt, device)
            runs[f"{m}_alone"].append(metrics)
            runs[f"{m}_alone_probe"].append(probe(model, loaders, seed, device, config))
        set_seed(seed)
        fusion = ConcatFusion(config, hidden_dim=ENCODER["hidden_dim"], dropout=ENCODER["dropout"],
                              pooling="mean", num_layers=ENCODER["num_layers"]).to(device)
        metrics, loaders = train(fusion, "concat_fusion", seed, ckpt, device)
        runs["fusion"].append(metrics)
        for m in MODALITIES:
            runs[f"{m}_joint_probe"].append(probe(fusion.encoders[m], loaders, seed, device, config))
        print(f"seed {seed} done: fusion corr={metrics['corr']:.4f}", flush=True)

    summary = {
        "task": "CMU-MOSI overall sentiment regression; reproduction of the OGM-GE imbalance observation",
        "seeds": list(SEEDS), "encoder": {**ENCODER, "pooling": "mean (audio/vision over real frames; text over all tokens)"},
        "training": {**TRAINING, "optimizer": "Adam", "loss": "MSE", "early_stopping": "val MAE",
                     "checkpoint": "best val MAE"},
        "probe": {**PROBE, "optimizer": "Adam", "loss": "MSE", "head": "Linear(feature_dim, 1)",
                  "encoder": "frozen, eval mode", "selection": "best val MAE"},
        "units": "fractions (accuracy, F1, corr) and label units (MAE); std over seeds with ddof=1",
        "runs": runs, "aggregate": {k: aggregate(v) for k, v in runs.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    agg = summary["aggregate"]
    fmt = lambda k, m: f"[{agg[k][m]['mean'] * 100:.2f} ± {agg[k][m]['std'] * 100:.2f}]"
    print("\nTypst rows (Acc-2, F1, Corr, MAE; x100):")
    for k in [*[f"{m}_alone" for m in MODALITIES], "fusion"]:
        print(f"  {k}: " + ", ".join(fmt(k, m) for m in METRICS))
    print("Encoders tested alone (linear probe):")
    for m in MODALITIES:
        for k in (f"{m}_alone_probe", f"{m}_joint_probe"):
            print(f"  {k}: " + ", ".join(fmt(k, x) for x in METRICS))
    print("Inside the fusion (contributions):")
    for m in MODALITIES:
        print(f"  {m}: share={fmt('fusion', m + '_share')} corr={fmt('fusion', m + '_corr')} mae={fmt('fusion', m + '_mae')}")


if __name__ == "__main__":
    main()
