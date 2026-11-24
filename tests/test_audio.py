"""Synthetic checks only: these tests do not measure real sentiment accuracy."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import logging
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from multimodal_ml.checkpoint import load_checkpoint
from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch, MultimodalDataset, _load_pickle, make_dataloaders
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.baseline import ConcatMLP
from multimodal_ml.seed import set_seed


@contextmanager
def temporary_training_logs():
    # Windows cannot remove a temporary log while FileHandler keeps it open.
    try:
        yield
    finally:
        logger = logging.getLogger("multimodal_ml")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


class AudioTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.config = get_dataset_config("mosi")
        self.model = AudioGRU(self.config).eval()
        self.batch = Batch(
            text=torch.randn(4, 2, 768), audio=torch.randn(4, 6, 5),
            vision=torch.randn(4, 2, 20), audio_lengths=torch.tensor([6, 3, 0, 1]),
            vision_lengths=torch.tensor([2, 2, 2, 2]), labels=torch.zeros(4),
            ids=["a", "b", "c", "d"],
        )

    def test_other_modalities_do_not_enter_predictions(self):
        actual = self.model(self.batch)
        audio_only = replace(self.batch, text=None, vision=None, vision_lengths=None)
        torch.testing.assert_close(self.model(audio_only), actual)

    def test_padding_and_empty_audio_do_not_change_predictions(self):
        changed = self.batch.audio.clone()
        for i, length in enumerate(self.batch.audio_lengths.tolist()):
            changed[i, length:] = 1e6
        torch.testing.assert_close(self.model(replace(self.batch, audio=changed)), self.model(self.batch))

    def test_non_finite_audio_is_zeroed(self):
        damaged = self.batch.audio.clone()
        damaged[0, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        cleaned = torch.nan_to_num(damaged, nan=0.0, posinf=0.0, neginf=0.0)
        torch.testing.assert_close(
            self.model(replace(self.batch, audio=damaged)),
            self.model(replace(self.batch, audio=cleaned)),
        )

    def test_padding_receives_no_gradient(self):
        audio = self.batch.audio.clone().requires_grad_()
        self.model(replace(self.batch, audio=audio)).sum().backward()
        self.assertGreater(audio.grad[0].abs().sum().item(), 0)
        for i, length in enumerate(self.batch.audio_lengths.tolist()):
            self.assertEqual(audio.grad[i, length:].abs().sum().item(), 0)

    def test_invalid_lengths_rejected(self):
        for lengths in (torch.tensor([7, 3, 0, 1]), torch.tensor([-1, 3, 0, 1])):
            with self.assertRaises(ValueError):
                self.model(replace(self.batch, audio_lengths=lengths))

    def test_model_cli_registration(self):
        from multimodal_ml.__main__ import parse_args
        with patch("sys.argv", ["train", "--dataset", "sims", "--model", "audio_gru", "--hidden-dim", "16"]):
            args = parse_args()
        model = AudioGRU.from_args(get_dataset_config("sims"), args)
        self.assertEqual(model.encoder.hidden_size, 16)

    def test_both_datasets_training_checkpoint_and_scalar_labels(self):
        # Synthetic files live only in a temporary directory, never data/.
        with tempfile.TemporaryDirectory(prefix="audio_synthetic_") as folder, temporary_training_logs():
            root = Path(folder)
            for name in ("mosi", "sims"):
                with self.subTest(dataset=name):
                    config = replace(get_dataset_config(name), pkl_path=root / f"{name}.pkl")
                    rng = np.random.default_rng(42)
                    splits = {}
                    for split in ("train", "valid", "test"):
                        splits[split] = {
                            "text": rng.normal(size=(8, 2, config.text_dim)).astype("float32"),
                            "audio": rng.normal(size=(8, 6, config.audio_dim)).astype("float32"),
                            "vision": rng.normal(size=(8, 2, config.vision_dim)).astype("float32"),
                            "audio_lengths": np.array([6, 4, 1, 0, 5, 2, 3, 6]),
                            "vision_lengths": np.full(8, 2),
                            "regression_labels": np.linspace(-1, 1, 8, dtype="float32").reshape(8, 1),
                            "id": [f"synthetic_{i}" for i in range(8)],
                        }
                        for i, length in enumerate(splits[split]["audio_lengths"]):
                            splits[split]["audio"][i, length:] = 0
                    splits["train"]["audio"][0, 0, 0] = np.inf
                    with config.pkl_path.open("wb") as source:
                        pickle.dump(splits, source)
                    with self.assertWarns(RuntimeWarning):
                        loaders = make_dataloaders(config, batch_size=4)
                    batch = next(iter(loaders[0]))
                    self.assertEqual(tuple(batch.labels.shape), (4,))
                    self.assertTrue(torch.isfinite(batch.audio).all())
                    self.assertEqual(tuple(ConcatMLP(config)(batch).shape), (4,))
                    args = argparse.Namespace(
                        dataset=name, model="audio_gru", epochs=2, batch_size=4,
                        lr=0.001, patience=1, seed=42, hidden_dim=64, dropout=0.3,
                        checkpoint_dir=root / name, device="cpu",
                    )
                    model = AudioGRU(config)
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        metrics = model.training_routine(loaders, torch.device("cpu"), args)
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                    checkpoint = root / name / f"{name}_audio_gru_best.pt"
                    restored = AudioGRU(config)
                    metadata = load_checkpoint(checkpoint, restored)
                    self.assertTrue(np.isfinite(metadata["metric_value"]))
                    self.assertTrue(list((root / name / "logs").glob("*.log")))
                    torch.testing.assert_close(restored.eval()(batch), model.eval()(batch))
            _load_pickle.cache_clear()

    def test_observed_audio_after_declared_length_is_retained(self):
        with tempfile.TemporaryDirectory(prefix="audio_endpoint_") as folder:
            config = replace(self.config, pkl_path=Path(folder) / "endpoint.pkl", audio_length_policy="observed")
            audio = np.zeros((1, 6, 5), dtype=np.float32)
            audio[0, 0] = 1
            audio[0, 4] = 2
            split = {
                "text": np.zeros((1, 2, 768)), "audio": audio,
                "vision": np.zeros((1, 2, 20)), "regression_labels": [0.0],
                "id": ["synthetic"], "audio_lengths": [2],
            }
            with config.pkl_path.open("wb") as source:
                pickle.dump({"train": split}, source)
            with self.assertWarnsRegex(RuntimeWarning, "audio lengths extended"):
                dataset = MultimodalDataset(config, "train")
            self.assertEqual(dataset.audio_lengths.tolist(), [5])
            np.testing.assert_array_equal(dataset.audio, audio)
            original = MultimodalDataset(replace(config, audio_length_policy="declared"), "train")
            self.assertEqual(original.audio_lengths.tolist(), [2])
            _load_pickle.cache_clear()

    def test_malformed_audio_lengths_rejected_by_loader(self):
        with tempfile.TemporaryDirectory(prefix="audio_invalid_") as folder:
            config = replace(self.config, pkl_path=Path(folder) / "invalid.pkl")
            split = {
                "text": np.zeros((1, 2, 768)), "audio": np.zeros((1, 6, 5)),
                "vision": np.zeros((1, 2, 20)), "regression_labels": np.zeros((1, 1)),
                "id": ["synthetic"], "audio_lengths": [1.5],
            }
            with config.pkl_path.open("wb") as source:
                pickle.dump({"train": split}, source)
            with self.assertRaisesRegex(ValueError, "audio_lengths"):
                MultimodalDataset(config, "train")
            _load_pickle.cache_clear()


if __name__ == "__main__":
    unittest.main()
