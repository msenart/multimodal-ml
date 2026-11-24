"""Fresh-checkout checks: no feature files or archived weights are required."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_audio_model import AudioSplit, repo_path, train_comparison
from multimodal_ml.config import get_dataset_config


class ReproducibilityTests(unittest.TestCase):
    def test_historical_windows_paths_are_portable(self):
        self.assertEqual(repo_path(r"checkpoints\audio\best.pt"), ROOT / "checkpoints/audio/best.pt")

    def test_shared_cli_preserves_declared_length_default(self):
        from multimodal_ml.__main__ import parse_args
        with patch("sys.argv", ["train", "--dataset", "mosi", "--model", "baseline"]):
            self.assertEqual(parse_args().audio_length_policy, "declared")
        with patch("sys.argv", ["train", "--dataset", "mosi", "--model", "audio_gru", "--audio-length-policy", "observed"]):
            self.assertEqual(parse_args().audio_length_policy, "observed")

    def test_new_seed42_comparison_needs_no_archived_weights(self):
        torch.set_num_threads(1)
        root = ROOT / "checkpoints"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="synthetic_reproduction_") as folder:
            config = replace(get_dataset_config("mosi"), audio_length_policy="observed")
            generator = torch.Generator().manual_seed(42)
            splits = {
                split: AudioSplit(torch.randn(8, 4, 5, generator=generator), torch.full((8,), 4),
                                  torch.linspace(-1, 1, 8), [f"synthetic_{i}" for i in range(8)])
                for split in ("train", "valid", "test")
            }
            for model in ("audio_mlp", "audio_gru"):
                with self.subTest(model=model):
                    with redirect_stdout(io.StringIO()):
                        result = train_comparison(config, splits, model, 42, Path(folder))
                    self.assertEqual(result["seed"], 42)
                    self.assertTrue((ROOT / result["checkpoint"]).is_file())
                    self.assertIn("test", result)
                    self.assertTrue(torch.isfinite(torch.tensor(result["test"]["mae"])))


if __name__ == "__main__":
    unittest.main()
