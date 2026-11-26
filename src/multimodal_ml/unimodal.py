"""Three ways to make every encoder of ConcatFusion learn on its own (alternatives to OGM-GE).

Every option is off by default and can be switched on from the CLI (see add_cli_args):

    --aux-weight W          add W * mean_m MSE(regressor_m(phi_m), y): each encoder also
                            predicts the label alone, with its own (otherwise unused) regressor
    --text-dropout P        during training, replace the text features of each sample by
                            zeros with probability P, so the fusion cannot rely on text only
    --unimodal-pretrain none|finetune|freeze
                            first train each modality alone (same training as the fusion),
                            copy its encoder into the fusion, then train the whole fusion
                            (finetune) or only the fusion head (freeze)

With the defaults, training is exactly the one of ogm_ge.train without these options.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

PRETRAIN_MODES = ("none", "finetune", "freeze")


@dataclass(frozen=True)
class UnimodalSettings:
    aux_weight: float = 0.0
    text_dropout: float = 0.0
    pretrain: str = "none"

    def __post_init__(self):
        if self.aux_weight < 0:
            raise ValueError("aux_weight must be >= 0")
        if not 0 <= self.text_dropout < 1:
            raise ValueError("text_dropout must be in [0, 1)")
        if self.pretrain not in PRETRAIN_MODES:
            raise ValueError(f"unimodal pretrain must be one of {PRETRAIN_MODES}")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "UnimodalSettings":
        d = cls()
        return cls(aux_weight=getattr(args, "aux_weight", d.aux_weight),
                   text_dropout=getattr(args, "text_dropout", d.text_dropout),
                   pretrain=getattr(args, "unimodal_pretrain", d.pretrain))


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="Weight of the uni-modal auxiliary losses (0: off)")
    parser.add_argument("--text-dropout", type=float, default=0.0,
                        help="Probability of zeroing the text features of a training sample (0: off)")
    parser.add_argument("--unimodal-pretrain", choices=PRETRAIN_MODES, default="none",
                        help="Train each encoder alone first, then fine-tune the fusion or train only its head")


def drop_text(features: dict[str, Tensor], p: float) -> dict[str, Tensor]:
    """Zero the text features of each sample with probability p (no rescaling: at test time text is always there)."""
    if p <= 0 or "text" not in features:
        return features
    keep = (torch.rand(features["text"].size(0), 1, device=features["text"].device) >= p).to(features["text"].dtype)
    return {**features, "text": features["text"] * keep}


def aux_loss(encoders, features: dict[str, Tensor], labels: Tensor) -> Tensor:
    """Mean over modalities of the MSE of each encoder's own regressor (net[-1])."""
    losses = [F.mse_loss(enc.net[-1](features[m]).squeeze(-1), labels) for m, enc in encoders.items()]
    return torch.stack(losses).mean()


def pretrain_encoders(model, loaders, device, args) -> list[dict]:
    """Train each modality alone (plain training, no OGM, no option of this module) and copy its
    encoder into `model`. `model.unimodal_copy(m)` builds the same model with modality m only.
    With --unimodal-pretrain freeze, the encoders are then frozen (and kept in eval mode)."""
    from multimodal_ml import ogm_ge  # avoid a circular import

    plain = copy.copy(args)
    for name, value in (("ogm_mode", "none"), ("aux_weight", 0.0), ("text_dropout", 0.0),
                        ("unimodal_pretrain", "none")):
        setattr(plain, name, value)
    histories = []
    for m in model.encoders:
        alone = model.unimodal_copy(m).to(device)
        if hasattr(alone.encoders[m], "fit_normalizer"):
            alone.encoders[m].fit_normalizer(loaders[0])
        _, history = ogm_ge.train(alone, loaders, device, plain, verbose=False)
        model.encoders[m].load_state_dict(alone.encoders[m].state_dict())
        histories.append({"modality": m, "history": history})
    if UnimodalSettings.from_args(args).pretrain == "freeze":
        model.freeze_encoders()
    return histories
