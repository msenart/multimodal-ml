import unittest

import torch
import numpy as np

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.reference_metrics import compute_mmsa_metrics


class ValidationTests(unittest.TestCase):
    def test_mmsa_sims_clipping_and_zero_as_negative(self):
        metrics = compute_mmsa_metrics(np.array([-2.0, 0.1, 2.0]), np.array([-1.0, 0.0, 1.0]), "sims")
        self.assertEqual(metrics["Mult_acc_2"], 0.6667)
        self.assertEqual(metrics["MAE"], 0.0333)

    def test_mmsa_mosi_acc5_uses_integer_rounding(self):
        metrics = compute_mmsa_metrics(np.array([-2.8, -0.4, 0.8]), np.array([-1.9, 0.2, 1.0]), "mosi")
        self.assertEqual(metrics["Mult_acc_5"], 1.0)

    def test_packed_gru_preserves_sample_order(self):
        torch.manual_seed(42)
        model = AudioGRU(get_dataset_config("mosi")).eval()
        audio = torch.randn(4, 8, 5)
        lengths = torch.tensor([3, 8, 1, 6])
        batch = Batch(None, audio, None, lengths, None, None, [])
        batched = model(batch)
        individual = torch.cat([
            model(Batch(None, audio[i:i + 1], None, lengths[i:i + 1], None, None, []))
            for i in range(4)
        ])
        torch.testing.assert_close(batched, individual)

    def test_mean_pooling_audio_model_ignores_padding_and_other_modalities(self):
        torch.manual_seed(42)
        model = AudioMLP(get_dataset_config("mosi")).eval()
        audio = torch.randn(3, 8, 5)
        lengths = torch.tensor([8, 2, 0])
        changed = audio.clone()
        for i, length in enumerate(lengths):
            changed[i, length:] = 1e6
        original = model(Batch(None, audio, None, lengths, None, None, []))
        actual = model(Batch(None, changed, None, lengths, None, None, []))
        torch.testing.assert_close(original, actual)
        self.assertEqual(tuple(actual.shape), (3,))


if __name__ == "__main__":
    unittest.main()
