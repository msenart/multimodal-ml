"""The default training routine, used by any model that doesn't override
`MultimodalModel.training_routine` (see models/base.py).

Edit `default_routine` only if you want to change the behavior for every model
that relies on the default. If just YOUR model needs something different
(a different loss, an LR schedule, gradient accumulation, ...), override
`training_routine` on your own model's class instead — don't edit this file.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multimodal_ml.checkpoint import EarlyStopper, load_checkpoint, save_checkpoint
from multimodal_ml.metrics import compute_regression_metrics

if TYPE_CHECKING:
    from multimodal_ml.models.base import MultimodalModel

Loaders = tuple[DataLoader, DataLoader, DataLoader]


@torch.no_grad()
def evaluate(model: "MultimodalModel", loader: DataLoader, device: torch.device) -> dict[str, float]:
    """No-grad pass over `loader`; returns compute_regression_metrics(...) using
    model.config. Reusable from inside a custom training_routine."""
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
    return compute_regression_metrics(preds, labels, model.config)


def default_routine(
    model: "MultimodalModel", loaders: Loaders, device: torch.device, args: argparse.Namespace
) -> dict[str, float]:
    """Standard MSE-regression train/validate/early-stop/test loop."""
    train_loader, valid_loader, test_loader = loaders

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

        val_metrics = evaluate(model, valid_loader, device)
        print(f"epoch {epoch}: val {val_metrics}")

        should_stop = stopper.step(val_metrics["mae"])
        if stopper.is_best:
            save_checkpoint(checkpoint_path, model, epoch, val_metrics["mae"])
        if should_stop:
            print(f"early stopping at epoch {epoch}")
            break

    load_checkpoint(checkpoint_path, model)
    return evaluate(model, test_loader, device)
