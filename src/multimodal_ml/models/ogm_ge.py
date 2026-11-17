"""OGM-GE: Balanced Multimodal Learning via On-the-fly Gradient Modulation (Peng et al., CVPR 2022).

SKELETON — the structure is in place, but the two method-specific pieces are still
stubs (`unimodal_scores`, `modulate_gradients`), so running this model raises
NotImplementedError until they are filled in.

Structure:
  - one encoder per modality (mean-pool over time, then Linear+ReLU)
  - one unimodal head per modality (used to measure each modality's contribution)
  - one fusion head on the concatenated features (the prediction we report)
  - training loop that adds the unimodal losses and modulates each modality's
    encoder gradients before the optimizer step.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from multimodal_ml.checkpoint import EarlyStopper, load_checkpoint, save_checkpoint
from multimodal_ml.config import DatasetConfig
from multimodal_ml.data import Batch
from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.baseline import _masked_mean
from multimodal_ml.training import (
    evaluate,
    format_metrics,
    print_metrics_table,
    print_run_summary,
    say,
    setup_logging,
)

if TYPE_CHECKING:
    from multimodal_ml.training import Loaders

MODALITIES = ("text", "audio", "vision")


class OGMGE(MultimodalModel):
    def __init__(self, config: DatasetConfig, hidden_dim: int = 128, alpha: float = 1.0):
        super().__init__(config)
        dims = {"text": config.text_dim, "audio": config.audio_dim, "vision": config.vision_dim}
        self.alpha = alpha  # modulation strength (paper's α)

        self.encoders = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU()) for m, d in dims.items()}
        )
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(hidden_dim, 1) for m in MODALITIES})
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(MODALITIES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--alpha", type=float, default=1.0, help="OGM-GE modulation strength")

    @classmethod
    def from_args(cls, config: DatasetConfig, args: argparse.Namespace) -> "OGMGE":
        return cls(config, hidden_dim=args.hidden_dim, alpha=args.alpha)

    def encode(self, batch: Batch) -> dict[str, Tensor]:
        pooled = {
            "text": batch.text.mean(dim=1),
            "audio": _masked_mean(batch.audio, batch.audio_lengths),
            "vision": _masked_mean(batch.vision, batch.vision_lengths),
        }
        return {m: self.encoders[m](pooled[m]) for m in MODALITIES}

    def forward_with_unimodal(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        """Fused prediction (B,) plus one unimodal prediction (B,) per modality."""
        feats = self.encode(batch)
        fused = self.fusion(torch.cat([feats[m] for m in MODALITIES], dim=-1)).squeeze(-1)
        unimodal = {m: self.unimodal_heads[m](feats[m]).squeeze(-1) for m in MODALITIES}
        return fused, unimodal

    def forward(self, batch: Batch) -> Tensor:
        fused, _ = self.forward_with_unimodal(batch)
        return fused

    def unimodal_scores(self, unimodal: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        """TODO: per-modality contribution score s_m for the current batch.

        Paper (classification): s_m = softmax(unimodal logits)[ground-truth class].
        Regression has no class, so this needs an adaptation, e.g. a score that is
        higher when the unimodal prediction is closer to the label. Decide and
        document this choice in the PR.
        """
        raise NotImplementedError("OGM-GE: unimodal contribution score not implemented yet")

    def modulate_gradients(self, scores: dict[str, Tensor]) -> None:
        """TODO: scale each modality's encoder gradients in place (after backward,
        before optimizer.step), using the discrepancy ratio between modalities.

        Modality params are reachable via `self.encoders[m].parameters()`.
        Check the exact coefficient formula (and the noise term, if any) against the
        paper before implementing.
        """
        raise NotImplementedError("OGM-GE: gradient modulation not implemented yet")

    def training_routine(
        self, loaders: "Loaders", device: torch.device, args: argparse.Namespace
    ) -> dict[str, float]:
        """Same train/validate/early-stop/test loop as the default routine, plus:
        unimodal auxiliary losses and gradient modulation before each step."""
        train_loader, valid_loader, test_loader = loaders

        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        stopper = EarlyStopper(patience=args.patience, mode="min")
        checkpoint_path = args.checkpoint_dir / f"{args.dataset}_{args.model}_best.pt"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = args.checkpoint_dir / "logs" / f"{args.dataset}_{args.model}_{stamp}.log"
        setup_logging(log_path)
        say(f"Logging to {log_path}")
        print_run_summary(self, loaders, device, args)

        best_mae = float("inf")
        for epoch in range(args.epochs):
            self.train()
            start = time.perf_counter()
            total_loss, n_seen = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                fused, unimodal = self.forward_with_unimodal(batch)
                loss = F.mse_loss(fused, batch.labels)
                for m in MODALITIES:
                    loss = loss + F.mse_loss(unimodal[m], batch.labels)

                optimizer.zero_grad()
                loss.backward()
                scores = self.unimodal_scores(unimodal, batch.labels)
                self.modulate_gradients(scores)
                optimizer.step()

                n = batch.labels.numel()
                total_loss += loss.item() * n
                n_seen += n
            train_loss = total_loss / max(n_seen, 1)

            val_metrics = evaluate(self, valid_loader, device, desc="validation")
            elapsed = time.perf_counter() - start

            should_stop = stopper.step(val_metrics["mae"])
            marker = ""
            if stopper.is_best:
                save_checkpoint(checkpoint_path, self, epoch, val_metrics["mae"])
                best_mae = val_metrics["mae"]
                marker = "  * new best, checkpoint saved"

            say(
                f"Epoch {epoch + 1}/{args.epochs} | {elapsed:.1f}s | train_loss={train_loss:.4f}"
                f"\n  val: {format_metrics(val_metrics)}{marker}"
            )

            if should_stop:
                say(f"Early stopping at epoch {epoch + 1} (best val mae={best_mae:.4f})")
                break

        load_checkpoint(checkpoint_path, self)
        test_metrics = evaluate(self, test_loader, device, desc="test")
        print_metrics_table("Test results (best checkpoint)", test_metrics)
        return test_metrics
