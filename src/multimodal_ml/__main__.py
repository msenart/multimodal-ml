"""Train a model on CMU-MOSI or CH-SIMS entirely through the CLI.

    uv run python -m multimodal_ml --dataset mosi --model baseline --epochs 1

Add your own architecture by writing `models/your_model.py` (a `MultimodalModel`
subclass) and registering it in `models/__init__.py`'s `MODEL_REGISTRY` — then
`--model your_model` picks it up here automatically. The training routine itself
(the loop below) is ours to edit directly if a new model needs something the loop
doesn't already do (a different loss, an LR schedule, ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multimodal_ml.checkpoint import EarlyStopper, load_checkpoint, save_checkpoint
from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.metrics import compute_regression_metrics
from multimodal_ml.models import MODEL_REGISTRY
from multimodal_ml.seed import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mosi", "sims"], required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader: DataLoader, config, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    model.train()
    return compute_regression_metrics(preds, labels, config)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    config = get_dataset_config(args.dataset)

    train_loader, valid_loader, test_loader = make_dataloaders(config, args.batch_size)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")
    checkpoint_path = args.checkpoint_dir / f"{args.dataset}_{args.model}_best.pt"

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = F.mse_loss(preds, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, valid_loader, config, device)
        print(f"epoch {epoch}: val {val_metrics}")

        should_stop = stopper.step(val_metrics["mae"])
        if stopper.is_best:
            save_checkpoint(checkpoint_path, model, epoch, val_metrics["mae"])
        if should_stop:
            print(f"early stopping at epoch {epoch}")
            break

    load_checkpoint(checkpoint_path, model)
    test_metrics = evaluate(model, test_loader, config, device)
    print(f"test: {test_metrics}")


if __name__ == "__main__":
    main()
