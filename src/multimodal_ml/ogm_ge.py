"""OGM-GE (Peng et al., CVPR 2022) adapted to the additive regression fusion of ConcatFusion.

Every part can be switched off from the CLI (see ConcatFusion.add_cli_args):

    --ogm none      plain joint training (numerically identical to training.default_routine)
    --ogm ogm       on-the-fly gradient modulation only (OM)
    --ogm ogm-ge    modulation + generalization enhancement (Gaussian gradient noise)
    --ogm-alpha     strength alpha of the modulation (alpha = 0 gives k = 1, i.e. no modulation)
    --ogm-ratio     max (default) or mean: compare a modality with the best / the mean of the others
    --ogm-start / --ogm-end   epochs (0-based, inclusive) during which OGM is applied
    --optimizer adam|sgd, --momentum, --weight-decay

Adaptation to regression and to three modalities (justified in report/ogm_ge.typ):
- score of modality m on a batch: s_m = sum_i exp(-(y_i - yhat_m,i)^2 / 2), where
  yhat_m = c_m + sum_{j != m} mean_batch(c_j) is the prediction of m alone with the
  other modalities held at their batch mean (c_m: additive part of m, see
  ConcatRegressionFusion.contributions). This is the Gaussian likelihood of the
  true label, the regression counterpart of the softmax probability of the true
  class used in the paper. It is always positive, so the ratios below are defined.
- ratio rho_m = s_m / max_{j != m} s_j (the paper's s_a / s_v when there are two modalities), so at
  most one modality (the best one) is slowed down. With "mean" instead of "max", a useless modality
  lowers the mean and every other modality is slowed down;
- coefficient k_m = 1 - tanh(alpha * (rho_m - 1)) if rho_m > 1 else 1 (paper, Eq. 10);
- the encoder gradients of m are multiplied by k_m; the fusion head is not modulated;
- GE (official code): to every encoder weight matrix, add N(0, std(original grad)^2) noise.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from multimodal_ml import unimodal
from multimodal_ml.checkpoint import EarlyStopper, save_checkpoint
from multimodal_ml.training import (
    Loaders, evaluate, format_metrics, print_metrics_table, print_run_summary, say, setup_logging,
)

logger = logging.getLogger("multimodal_ml.ogm_ge")

OGM_MODES = ("none", "ogm", "ogm-ge")
OPTIMIZERS = ("adam", "sgd")
OGM_RATIOS = ("max", "mean")


@dataclass(frozen=True)
class OGMSettings:
    mode: str = "none"
    alpha: float = 0.1
    ratio: str = "max"
    start: int = 0
    end: int | None = None  # None: until the last epoch
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0

    def __post_init__(self):
        if self.mode not in OGM_MODES:
            raise ValueError(f"ogm mode must be one of {OGM_MODES}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.ratio not in OGM_RATIOS:
            raise ValueError(f"ratio must be one of {OGM_RATIOS}")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OGMSettings":
        """Missing attributes take the defaults, so callers that predate OGM (e.g.
        scripts/ogm_reproduction.py) still get plain Adam training."""
        d = cls()
        return cls(mode=getattr(args, "ogm_mode", d.mode), alpha=getattr(args, "ogm_alpha", d.alpha),
                   ratio=getattr(args, "ogm_ratio", d.ratio),
                   start=getattr(args, "ogm_start", d.start), end=getattr(args, "ogm_end", d.end),
                   optimizer=getattr(args, "optimizer", d.optimizer), momentum=getattr(args, "momentum", d.momentum),
                   weight_decay=getattr(args, "weight_decay", d.weight_decay))

    def active(self, epoch: int) -> bool:
        return self.mode != "none" and self.start <= epoch and (self.end is None or epoch <= self.end)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ogm", dest="ogm_mode", choices=OGM_MODES, default="none",
                        help="none: plain training; ogm: gradient modulation; ogm-ge: modulation + gradient noise")
    parser.add_argument("--ogm-alpha", type=float, default=0.1, help="Modulation strength (0: k = 1)")
    parser.add_argument("--ogm-ratio", choices=OGM_RATIOS, default="max",
                        help="Score of a modality divided by the best (max) or the mean of the others")
    parser.add_argument("--ogm-start", type=int, default=0, help="First epoch (0-based) with OGM")
    parser.add_argument("--ogm-end", type=int, default=None, help="Last epoch (0-based, inclusive) with OGM")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0)


def modality_scores(contributions: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
    """s_m = sum_i exp(-(y_i - yhat_m,i)^2 / 2), yhat_m = c_m + sum_{j != m} mean(c_j)."""
    means = {m: c.mean() for m, c in contributions.items()}
    scores = {}
    for m, c in contributions.items():
        alone = c + sum(means[j] for j in contributions if j != m)
        scores[m] = torch.exp(-0.5 * (labels - alone).square()).sum()
    return scores


def modulation_coefficients(scores: dict[str, Tensor], alpha: float,
                            ratio: str = "max") -> tuple[dict[str, float], dict[str, float]]:
    """(rho, k) per modality. rho_m = s_m / max (or mean)_{j != m} s_j; k_m = 1 - tanh(alpha (rho_m - 1)) if rho_m > 1."""
    if len(scores) < 2:
        return {m: 1.0 for m in scores}, {m: 1.0 for m in scores}
    rho, k = {}, {}
    for m, s in scores.items():
        others = torch.stack([scores[j] for j in scores if j != m])
        others = others.max() if ratio == "max" else others.mean()
        rho[m] = float(s / others.clamp_min(1e-12))
        k[m] = 1.0 - float(np.tanh(alpha * (rho[m] - 1.0))) if rho[m] > 1.0 else 1.0
    return rho, k


def modulate_gradients(encoders: dict[str, torch.nn.Module], k: dict[str, float], ge: bool) -> None:
    """grad <- k_m * grad (+ N(0, std(grad)^2) on weight matrices if ge), as in the official code."""
    for m, encoder in encoders.items():
        for p in encoder.parameters():
            if p.grad is None:
                continue
            noise_std = p.grad.std().item() + 1e-8 if ge and p.ndim >= 2 else None
            p.grad.mul_(k[m])
            if noise_std is not None:
                p.grad.add_(torch.randn_like(p.grad) * noise_std)


def make_optimizer(params, lr: float, settings: OGMSettings) -> torch.optim.Optimizer:
    if settings.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=settings.momentum, weight_decay=settings.weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=settings.weight_decay)


def _flat(module: torch.nn.Module) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(grads).sum().sqrt()) if grads else 0.0  # no gradient: frozen encoder


def train(model, loaders: Loaders, device: torch.device, args: argparse.Namespace,
          verbose: bool = True) -> tuple[dict[str, float], list[dict]]:
    """Train/validate/early-stop/test like training.default_routine, with optional OGM(-GE)
    and the optional uni-modal aids of multimodal_ml/unimodal.py.

    With --ogm none and --optimizer adam the computation is the same as default_routine
    (tested in tests/test_ogm_ge.py). Returns (test metrics of the best-val-MAE model,
    per-epoch history). The history holds, per modality, the mean rho and k, the mean
    gradient norm of the encoder before modulation and the mean norm of its parameter
    update per step (to measure what the optimizer does with the modulated gradient).
    """
    settings = OGMSettings.from_args(args)
    aids = unimodal.UnimodalSettings.from_args(args)
    if aids.pretrain != "none":  # before the optimizer, so frozen encoders get no update
        model.pretrain_history = unimodal.pretrain_encoders(model, loaders, device, args)
    train_loader, valid_loader, test_loader = loaders
    encoders = dict(model.encoders.items())
    optimizer = make_optimizer(model.parameters(), args.lr, settings)
    stopper = EarlyStopper(patience=args.patience, mode="min")
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if verbose:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_logging(checkpoint_dir / "logs" / f"{args.dataset}_{args.model}_{stamp}.log")
        print_run_summary(model, loaders, device, args)

    best_state, history, diverged = None, [], False
    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        active = settings.active(epoch)
        sums = {key: {m: 0.0 for m in encoders} for key in ("rho", "k", "grad_norm", "update_norm")}
        total_loss, n_seen, steps = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            features = model.encode(batch)
            preds = model.head(unimodal.drop_text(features, aids.text_dropout))
            loss = F.mse_loss(preds, batch.labels)
            if aids.aux_weight > 0:
                loss = loss + aids.aux_weight * unimodal.aux_loss(encoders, features, batch.labels)
            if not torch.isfinite(loss):
                diverged = True
                break

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                parts = model.head.contributions({m: f.detach() for m, f in features.items()})
                rho, k = modulation_coefficients(modality_scores(parts, batch.labels), settings.alpha,
                                                  settings.ratio)
            for m, enc in encoders.items():
                sums["grad_norm"][m] += _grad_norm(enc)
            if active:
                modulate_gradients(encoders, k, ge=settings.mode == "ogm-ge")
            before = {m: _flat(enc) for m, enc in encoders.items()}
            optimizer.step()
            for m, enc in encoders.items():
                sums["update_norm"][m] += float((_flat(enc) - before[m]).norm())
                sums["rho"][m] += rho[m]
                sums["k"][m] += k[m] if active else 1.0

            n = batch.labels.numel()
            total_loss += loss.item() * n
            n_seen += n
            steps += 1
        if diverged:  # non-finite loss (e.g. SGD with a too large learning rate): keep the best epoch so far
            if verbose:
                say(f"Non-finite training loss at epoch {epoch + 1}: stopping")
            break
        train_loss = total_loss / max(n_seen, 1)

        val_metrics = evaluate(model, valid_loader, device, desc="validation")
        should_stop = stopper.step(val_metrics["mae"] if np.isfinite(val_metrics["mae"]) else float("inf"))
        if stopper.is_best:
            best_state = copy.deepcopy(model.state_dict())
            if verbose:
                save_checkpoint(checkpoint_dir / f"{args.dataset}_{args.model}_best.pt", model, epoch,
                                val_metrics["mae"])
        record = {"epoch": epoch, "train_loss": train_loss, "ogm_active": active,
                  "val_mae": val_metrics["mae"], "val_corr": val_metrics["corr"],
                  **{f"{key}_{m}": v / max(steps, 1) for key, d in sums.items() for m, v in d.items()}}
        history.append(record)
        if verbose:
            mods = "  ".join(f"k_{m}={record[f'k_{m}']:.3f}" for m in encoders)
            line = (f"Epoch {epoch + 1}/{args.epochs} | {time.perf_counter() - start:.1f}s | "
                    f"train_loss={train_loss:.4f} | {mods}\n  val: {format_metrics(val_metrics)}"
                    + ("  * new best" if stopper.is_best else ""))
            say(line)
            logger.info(line)
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("Training diverged before the first validation")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, desc="test")
    if verbose:
        print_metrics_table("Test results (best checkpoint)", test_metrics)
        logger.info("test | %s", format_metrics(test_metrics))
    if history:
        history[-1]["diverged"] = diverged
    return test_metrics, history
