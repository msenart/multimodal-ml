"""Train a model on CMU-MOSI or CH-SIMS entirely through the CLI.

    uv run python -m multimodal_ml --dataset mosi --model baseline --epochs 1

Add your own architecture by writing `models/your_model.py` (a `MultimodalModel`
subclass) and registering it in `models/__init__.py`'s `MODEL_REGISTRY` — then
`--model your_model` picks it up here automatically. The default training
routine lives in `training.py`; if your model needs different training
behavior, override `training_routine` on your own model class instead of
editing anything shared. Likewise, if your model needs its own hyperparameters
(e.g. `--hidden-dim`), override `add_cli_args`/`from_args` on your own model
class instead of editing the shared parser below (see models/base.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import make_dataloaders
from multimodal_ml.models import MODEL_REGISTRY
from multimodal_ml.seed import get_device, set_seed


def _peek_model_name() -> str:
    """First-pass parse of just --model, so we know which model's add_cli_args
    to merge into the real parser below (its own flags, --help, and the
    required --dataset check all happen on that second, full parse)."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="baseline")
    known_args, _ = pre_parser.parse_known_args()
    return known_args.model


def parse_args() -> argparse.Namespace:
    model_cls = MODEL_REGISTRY[_peek_model_name()]

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
    model_cls.add_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    config = get_dataset_config(args.dataset)
    loaders = make_dataloaders(config, args.batch_size)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls.from_args(config, args).to(device)

    test_metrics = model.training_routine(loaders, device, args)
    print(f"test: {test_metrics}")


if __name__ == "__main__":
    main()
