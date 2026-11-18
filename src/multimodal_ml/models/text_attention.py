
from __future__ import annotations

import logging
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel


class TextAttentionMLP(MultimodalModel):
    """Text-only sentiment regression using attention pooling and MLP."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        # Attention layer: calculate the importance score of each token.
        self.attention = nn.Linear(
            config.text_dim,
            1,
            bias=False,
        )

        # MLP prediction network.
        self.net = nn.Sequential(
            nn.Linear(config.text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Apply masked attention pooling and predict sentiment scores."""

        x = batch.text

        # Identify zero-vector padding.
        mask = x.abs().sum(dim=-1) > 1e-8

        # Calculate attention scores.
        scores = self.attention(x).squeeze(-1)

        # Exclude padding tokens from attention.
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        # Convert scores into normalized attention weights.
        weights = torch.softmax(scores, dim=1)

        # Weighted sum of token features.
        text_repr = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        # Predict continuous sentiment scores.
        return self.net(text_repr).squeeze(-1)

    def training_routine(self, loaders, device, args) -> dict[str, float]:
        """Train the attention model using L1 loss and MAE-based early stopping."""

        from multimodal_ml.checkpoint import (
            EarlyStopper,
            load_checkpoint,
            save_checkpoint,
        )

        from multimodal_ml.training import (
            evaluate,
            format_metrics,
            print_metrics_table,
            print_run_summary,
            say,
            setup_logging,
        )

        logger = logging.getLogger("multimodal_ml.training")

        train_loader, valid_loader, test_loader = loaders

        # Optimizer.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
        )

        # Early stopping based on validation MAE.
        stopper = EarlyStopper(
            patience=args.patience,
            mode="min",
        )

        # Use a separate checkpoint for Attention + L1.
        checkpoint_path = (
            args.checkpoint_dir
            / f"{args.dataset}_{args.model}_l1_best.pt"
        )

        # Logging.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_path = (
            args.checkpoint_dir
            / "logs"
            / f"{args.dataset}_{args.model}_l1_{stamp}.log"
        )

        setup_logging(log_path)

        say(f"Logging to {log_path}")
        print_run_summary(self, loaders, device, args)
        say("Loss function: L1")

        best_mae = float("inf")

        # Training loop.
        for epoch in range(args.epochs):

            self.train()
            start = time.perf_counter()

            total_loss = 0.0
            n_seen = 0

            bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
            )

            for batch in bar:

                batch = batch.to(device)

                # Forward propagation.
                preds = self(batch)

                # L1 loss.
                loss = F.l1_loss(
                    preds,
                    batch.labels,
                )

                # Backpropagation.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record training loss.
                n = batch.labels.numel()

                total_loss += loss.item() * n
                n_seen += n

                bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{total_loss / n_seen:.4f}",
                )

            train_loss = total_loss / max(n_seen, 1)

            # Validation.
            val_metrics = evaluate(
                self,
                valid_loader,
                device,
                desc="validation",
            )

            elapsed = time.perf_counter() - start

            # Early stopping.
            should_stop = stopper.step(val_metrics["mae"])

            marker = ""

            if stopper.is_best:

                save_checkpoint(
                    checkpoint_path,
                    self,
                    epoch,
                    val_metrics["mae"],
                )

                best_mae = val_metrics["mae"]
                marker = "  * new best, checkpoint saved"

            # Print epoch results.
            say(
                f"Epoch {epoch + 1}/{args.epochs}"
                f" | {elapsed:.1f}s"
                f" | train_loss={train_loss:.4f}"
                f"\n  val: {format_metrics(val_metrics)}{marker}"
            )

            logger.info(
                "epoch %d | time %.1fs | train_loss %.5f | val %s%s",
                epoch + 1,
                elapsed,
                train_loss,
                format_metrics(val_metrics),
                marker,
            )

            if should_stop:

                say(
                    f"Early stopping at epoch {epoch + 1}"
                    f" (best val mae={best_mae:.4f})"
                )

                logger.info(
                    "early stopping at epoch %d (best val mae %.5f)",
                    epoch + 1,
                    best_mae,
                )

                break

        # Load the best checkpoint.
        load_checkpoint(
            checkpoint_path,
            self,
        )

        # Final test evaluation.
        test_metrics = evaluate(
            self,
            test_loader,
            device,
            desc="test",
        )

        print_metrics_table(
            "Test results (best Attention + L1 checkpoint)",
            test_metrics,
        )

        logger.info(
            "test | %s",
            format_metrics(test_metrics),
        )

        return test_metrics
