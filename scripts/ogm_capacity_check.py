"""Can each basic encoder fit (overfit) the CMU-MOSI train split?

Trains text-mlp, audio-mlp and vision-mlp alone with dropout 0 and no early
stopping, and prints train-split MAE / Corr at a few epochs. Used only to choose
encoder capacity before the OGM reproduction; not part of the report.

    uv run python scripts/ogm_capacity_check.py --epochs 200
    uv run python scripts/ogm_capacity_check.py --models audio_mlp --hidden-dim 256 --num-layers 3 --pooling stats
"""
from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.models import MODEL_REGISTRY
from multimodal_ml.seed import set_seed
from multimodal_ml.training import evaluate


def build(name: str, config, args):
    kwargs = {"hidden_dim": args.hidden_dim, "dropout": 0.0}
    if args.num_layers is not None:
        kwargs["num_layers"] = args.num_layers
    if name != "text_mlp":
        kwargs["pooling"] = args.pooling
    return MODEL_REGISTRY[name](config, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["text_mlp", "audio_mlp", "vision_mlp"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "stats"], default="mean")
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()
    config = get_dataset_config("mosi")
    device = torch.device("cpu")
    milestones = {1, 10, 25, 50, 100, 200, 400, 800, args.epochs}
    for name in args.models:
        set_seed(args.seed)
        model = build(name, config, args).to(device)
        train_loader, _, _ = make_dataloaders(config, args.batch_size, transform=model.transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        params = sum(p.numel() for p in model.parameters())
        print(f"== {name} ({params:,} parameters, hidden={args.hidden_dim}, "
              f"layers={args.num_layers or 1}, pooling={args.pooling if name != 'text_mlp' else '-'})")
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                loss = F.mse_loss(model(batch), batch.labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch in milestones:
                m = evaluate(model, train_loader, device, desc="train")
                print(f"  epoch {epoch:4d}  train mae={m['mae']:.4f}  corr={m['corr']:.4f}  "
                      f"acc2={m['acc2_has0']:.4f}", flush=True)


if __name__ == "__main__":
    main()
