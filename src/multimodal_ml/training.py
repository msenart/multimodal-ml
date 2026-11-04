"""The default training routine, used by any model that doesn't override
`MultimodalModel.training_routine` (see models/base.py).

Edit `default_routine` only if you want to change the behavior for every model
that relies on the default. If just YOUR model needs something different
(a different loss, an LR schedule, gradient accumulation, ...), override
`training_routine` on your own model's class instead — don't edit this file.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_ml.checkpoint import EarlyStopper, load_checkpoint, save_checkpoint
from multimodal_ml.metrics import compute_regression_metrics

if TYPE_CHECKING:
    from multimodal_ml.models.base import MultimodalModel

Loaders = tuple[DataLoader, DataLoader, DataLoader]

logger = logging.getLogger("multimodal_ml.training")


def setup_logging(log_path: Path) -> None:
    """Send the `multimodal_ml` logger to `log_path` (full detail, timestamped).

    Console output is handled by tqdm/`say` so progress bars don't get garbled;
    the file gets everything. Re-callable: previous file handlers are replaced.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("multimodal_ml")
    root.setLevel(logging.INFO)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)


def say(message: str = "") -> None:
    """Print without breaking an active tqdm bar."""
    tqdm.write(message)


def format_metrics(metrics: dict[str, float]) -> str:
    """One-line `key=value` rendering of a metrics dict."""
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def print_metrics_table(title: str, metrics: dict[str, float]) -> None:
    """Aligned two-column table of metrics."""
    width = max(len(k) for k in metrics)
    line = "-" * (width + 11)
    say(f"\n{title}\n{line}")
    for k, v in metrics.items():
        say(f"{k:<{width}} | {v:>8.4f}")
    say(line)


@torch.no_grad()
def evaluate(
    model: "MultimodalModel", loader: DataLoader, device: torch.device, desc: str = "eval"
) -> dict[str, float]:
    """No-grad pass over `loader`; returns compute_regression_metrics(...) using
    model.config. Reusable from inside a custom training_routine."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    model.train()
    return compute_regression_metrics(preds, labels, model.config)


def describe_device(device: torch.device) -> str:
    """The accelerator the model actually runs on (GPU name on CUDA)."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def print_run_summary(
    model: "MultimodalModel", loaders: Loaders, device: torch.device, args: argparse.Namespace
) -> None:
    """Terminal + log header: general run info, then every hyperparameter."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    general = {
        "dataset": args.dataset,
        "model": args.model,
        "device": describe_device(device),
        "parameters": f"{n_params:,} ({n_trainable:,} trainable)",
        "samples (train/val/test)": " / ".join(str(len(l.dataset)) for l in loaders),
    }
    shown = set(general) | {"device", "checkpoint_dir"}
    hyper = {k: v for k, v in vars(args).items() if k not in shown}
    hyper["checkpoint_dir"] = args.checkpoint_dir

    width = max(len(k) for k in [*general, *hyper])
    lines = ["=" * 60, "Run configuration", "-" * 60]
    lines += [f"{k:<{width}} : {v}" for k, v in general.items()]
    lines.append("-" * 60)
    lines += [f"{k:<{width}} : {v}" for k, v in hyper.items()]
    lines.append("=" * 60)
    for line in lines:
        say(line)
        logger.info(line)


def default_routine(
    model: "MultimodalModel", loaders: Loaders, device: torch.device, args: argparse.Namespace
) -> dict[str, float]:
    """Standard MSE-regression train/validate/early-stop/test loop.

    Shows a tqdm bar per epoch (live loss), prints a validation summary after each
    epoch, a metrics table for the test set, and logs everything to
    `<checkpoint_dir>/logs/<dataset>_<model>_<timestamp>.log`.
    """
    train_loader, valid_loader, test_loader = loaders

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")
    checkpoint_path = args.checkpoint_dir / f"{args.dataset}_{args.model}_best.pt"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.checkpoint_dir / "logs" / f"{args.dataset}_{args.model}_{stamp}.log"
    setup_logging(log_path)
    say(f"Logging to {log_path}")
    print_run_summary(model, loaders, device, args)

    best_mae = float("inf")
    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        total_loss, n_seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch in bar:
            batch = batch.to(device)
            preds = model(batch)
            loss = F.mse_loss(preds, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = batch.labels.numel()
            total_loss += loss.item() * n
            n_seen += n
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / n_seen:.4f}")
        train_loss = total_loss / max(n_seen, 1)

        val_metrics = evaluate(model, valid_loader, device, desc="validation")
        elapsed = time.perf_counter() - start

        should_stop = stopper.step(val_metrics["mae"])
        marker = ""
        if stopper.is_best:
            save_checkpoint(checkpoint_path, model, epoch, val_metrics["mae"])
            best_mae = val_metrics["mae"]
            marker = "  * new best, checkpoint saved"

        say(
            f"Epoch {epoch + 1}/{args.epochs} | {elapsed:.1f}s | train_loss={train_loss:.4f}"
            f"\n  val: {format_metrics(val_metrics)}{marker}"
        )
        logger.info(
            "epoch %d | time %.1fs | train_loss %.5f | val %s%s",
            epoch + 1, elapsed, train_loss, format_metrics(val_metrics), marker,
        )

        if should_stop:
            say(f"Early stopping at epoch {epoch + 1} (best val mae={best_mae:.4f})")
            logger.info("early stopping at epoch %d (best val mae %.5f)", epoch + 1, best_mae)
            break

    load_checkpoint(checkpoint_path, model)
    test_metrics = evaluate(model, test_loader, device, desc="test")
    print_metrics_table("Test results (best checkpoint)", test_metrics)
    logger.info("test | %s", format_metrics(test_metrics))
    return test_metrics
