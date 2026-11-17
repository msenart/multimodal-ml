"""OGM-GE: Balanced Multimodal Learning via On-the-fly Gradient Modulation (Peng et al., CVPR 2022).

Adapted to multimodal sentiment *regression* (MOSI) and to three modalities
(text, audio, vision). Deviations from the paper are listed in the PR description:

  - Contribution score: s_i^u = exp(-|pred_i^u - y_i|) instead of the softmax
    probability of the ground-truth class (regression has no classes). The batch
    score S^u = sum_i s_i^u.
  - Discrepancy ratio: rho^u = S^u / mean_{v != u} S^v (paper: two modalities,
    rho^a = 1 / rho^v).
  - Gradient noise (Generalization Enhancement): the covariance of the mini-batch
    gradient is estimated per parameter with an exponential moving average of
    squared deviations, instead of the per-sample covariance of the paper.
  - Optimizer: SGD (momentum 0.9, weight decay 1e-4) as in the paper, instead of
    the Adam used by the default routine. Adam is invariant to a constant gradient
    rescaling, so modulation would have almost no effect with it.

Structure:
  - one encoder per modality (mean-pool over time, then Linear+ReLU)
  - one unimodal head per modality (used to measure each modality's contribution)
  - one fusion head on the concatenated features (the prediction we report)
  - the training loop adds the unimodal losses, then modulates the encoders'
    gradients and injects GE noise before the optimizer step.
"""

from __future__ import annotations

import argparse
import math
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

SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-4
GRAD_STAT_BETA = 0.9  # EMA decay for the gradient mean/variance used by GE noise
EPS = 1e-12


class OGMGE(MultimodalModel):
    def __init__(self, config: DatasetConfig, hidden_dim: int = 128, alpha: float = 0.5):
        super().__init__(config)
        dims = {"text": config.text_dim, "audio": config.audio_dim, "vision": config.vision_dim}
        self.alpha = alpha  # modulation strength, paper's α in [0, 1]

        self.encoders = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU()) for m, d in dims.items()}
        )
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(hidden_dim, 1) for m in MODALITIES})
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(MODALITIES), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Non-registered state (not part of the checkpoint): per-parameter EMA of the
        # mini-batch gradient mean and variance, and the last modulation coefficients.
        self._grad_stats: dict[str, tuple[Tensor, Tensor]] = {}
        self.last_coeffs: dict[str, float] = {m: 1.0 for m in MODALITIES}

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--alpha", type=float, default=0.5, help="OGM-GE modulation strength in [0, 1]")

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

    @torch.no_grad()
    def unimodal_scores(self, unimodal: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        """Batch contribution score S^u = sum_i exp(-|pred_i^u - y_i|) per modality.

        Closer unimodal predictions give a score nearer 1, so a larger S^u means the
        modality currently contributes more to the objective.
        """
        return {m: torch.exp(-(unimodal[m] - labels).abs()).sum() for m in MODALITIES}

    @torch.no_grad()
    def modulate_gradients(self, sums: dict[str, Tensor]) -> None:
        """Apply OGM-GE to the encoders' gradients in place (after backward, before step).

        For each modality u:
          rho^u = S^u / mean_{v != u} S^v
          k^u   = 1 - tanh(alpha * rho^u) if rho^u > 1, else 1           (paper eq. 10)
          grad  = k^u * g + h,  h ~ N(0, var)                             (paper eq. 16)
        where var is an EMA estimate of the per-parameter variance of the mini-batch
        gradient g (GE noise, paper section 3.3).
        """
        totals = {m: float(sums[m]) for m in MODALITIES}
        for m in MODALITIES:
            others = [totals[v] for v in MODALITIES if v != m]
            rho = totals[m] / max(sum(others) / len(others), EPS)
            k = 1.0 - math.tanh(self.alpha * rho) if rho > 1.0 else 1.0
            self.last_coeffs[m] = k

            for name, p in self.encoders[m].named_parameters():
                if p.grad is None:
                    continue
                key = f"{m}.{name}"
                g = p.grad
                mean, var = self._grad_stats.get(key, (torch.zeros_like(g), torch.zeros_like(g)))
                mean = GRAD_STAT_BETA * mean + (1 - GRAD_STAT_BETA) * g
                var = GRAD_STAT_BETA * var + (1 - GRAD_STAT_BETA) * (g - mean) ** 2
                self._grad_stats[key] = (mean, var)
                p.grad = k * g + torch.sqrt(var) * torch.randn_like(g)

    def training_routine(
        self, loaders: "Loaders", device: torch.device, args: argparse.Namespace
    ) -> dict[str, float]:
        """Same train/validate/early-stop/test loop as the default routine, plus:
        unimodal auxiliary losses, OGM-GE gradient modulation, and SGD."""
        train_loader, valid_loader, test_loader = loaders

        optimizer = torch.optim.SGD(
            self.parameters(), lr=args.lr, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY
        )
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
            coeff_sum = {m: 0.0 for m in MODALITIES}
            n_batches = 0
            for batch in train_loader:
                batch = batch.to(device)
                fused, unimodal = self.forward_with_unimodal(batch)
                loss = F.mse_loss(fused, batch.labels)
                for m in MODALITIES:
                    loss = loss + F.mse_loss(unimodal[m], batch.labels)

                optimizer.zero_grad()
                loss.backward()
                sums = self.unimodal_scores(unimodal, batch.labels)
                self.modulate_gradients(sums)
                for m in MODALITIES:
                    coeff_sum[m] += self.last_coeffs[m]
                n_batches += 1
                optimizer.step()

                n = batch.labels.numel()
                total_loss += loss.item() * n
                n_seen += n
            train_loss = total_loss / max(n_seen, 1)
            coeffs = "  ".join(f"{m}={coeff_sum[m] / max(n_batches, 1):.3f}" for m in MODALITIES)

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
                f"\n  coeff (mean k): {coeffs}"
                f"\n  val: {format_metrics(val_metrics)}{marker}"
            )

            if should_stop:
                say(f"Early stopping at epoch {epoch + 1} (best val mae={best_mae:.4f})")
                break

        load_checkpoint(checkpoint_path, self)
        test_metrics = evaluate(self, test_loader, device, desc="test")
        print_metrics_table("Test results (best checkpoint)", test_metrics)
        return test_metrics
