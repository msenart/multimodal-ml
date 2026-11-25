from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.models.concat_fusion import ConcatRegressionFusion
from multimodal_ml.models.vision_mlp_deep import VisionMLPDeep


def vision_batch(samples):
    return Batch(None, None, torch.stack([s[0] for s in samples]), None,
                 torch.tensor([s[1] for s in samples]), None, [])


class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = get_dataset_config("mosi")
        self.vision = torch.randn(4, 6, 20)
        self.lengths = torch.tensor([6, 3, 0, 1])
        self.batch = Batch(None, None, self.vision, None, self.lengths, None, [])

    def test_only_vision_and_valid_frames_enter_predictions(self):
        model = VisionMLPDeep(self.config, normalization="none").eval()
        original = model(self.batch)
        changed = self.vision.clone()
        for i, length in enumerate(self.lengths):
            changed[i, length:] = 1e6
        other = replace(self.batch, vision=changed, text=torch.full((4, 2, 768), float("nan")),
                        audio=torch.full((4, 2, 5), float("nan")))
        torch.testing.assert_close(model(other), original)
        self.assertEqual(tuple(model.encode(self.batch).shape), (4, 64))
        self.assertEqual(model.encode(self.batch)[2].abs().sum().item(), 0)

    def test_train_only_normalization_and_rng_preservation(self):
        config = replace(self.config, vision_dim=2)
        model = VisionMLPDeep(config)
        train = [(torch.tensor([[1., 2.], [3., 4.]]), 2),
                 (torch.tensor([[5., 6.], [999., 999.]]), 1)]
        loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=vision_batch)
        state = torch.get_rng_state().clone()
        model.fit_normalizer(loader)
        torch.testing.assert_close(torch.get_rng_state(), state)
        torch.testing.assert_close(model.feature_mean, torch.tensor([3.5, 4.5]))
        torch.testing.assert_close(model.feature_scale, torch.tensor([1.5, 1.5]))
        before = model.feature_mean.clone()
        model.eval()(Batch(None, None, torch.full((1, 2, 2), 10000.), None, torch.tensor([2]), None, []))
        torch.testing.assert_close(model.feature_mean, before)

    def test_constant_features_use_unit_scale(self):
        model = VisionMLPDeep(replace(self.config, vision_dim=2))
        loader = DataLoader([(torch.ones(2, 2), 2)] * 3, batch_size=3, collate_fn=vision_batch)
        model.fit_normalizer(loader)
        torch.testing.assert_close(model.feature_scale, torch.ones(2))
        self.assertTrue(model.constant_features.all())

    def test_normalizer_is_saved_and_restored_with_encoder(self):
        model = VisionMLPDeep(self.config)
        loader = DataLoader([(v, int(n)) for v, n in zip(self.vision, self.lengths)], batch_size=4, collate_fn=vision_batch)
        model.fit_normalizer(loader)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = VisionMLPDeep(self.config)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(model.eval()(self.batch), restored.eval()(self.batch))
        self.assertEqual(restored.normalizer_samples.item(), 3)

    def test_normalization_must_be_fitted_and_lengths_valid(self):
        with self.assertRaisesRegex(RuntimeError, "training loader"):
            VisionMLPDeep(self.config)(self.batch)
        model = VisionMLPDeep(self.config, normalization="none")
        for lengths in (torch.tensor([7, 3, 0, 1]), torch.tensor([-1, 3, 0, 1]), torch.tensor([1.5, 3., 0., 1.])):
            with self.assertRaises(ValueError):
                model(replace(self.batch, vision_lengths=lengths))

    def test_heterogeneous_fusion_decomposes_and_updates_all_branches(self):
        vision = VisionMLPDeep(self.config, normalization="none")
        audio = AudioMLP(self.config)
        text = nn.Linear(self.config.text_dim, 11)
        batch = replace(self.batch, text=torch.randn(4, 2, 768), audio=torch.randn(4, 6, 5), audio_lengths=self.lengths)
        features = {"text": text(batch.text.mean(1)), "audio": audio.encode(batch), "vision": vision.encode(batch)}
        fusion = ConcatRegressionFusion({"text": 11, "audio": audio.feature_dim, "vision": vision.feature_dim})
        prediction = fusion(features)
        torch.testing.assert_close(prediction, sum(fusion.contributions(features).values()))
        (prediction - torch.linspace(-1, 1, 4)).square().mean().backward()
        for layer in (text, audio.net[0], vision.encoder[1], fusion.head):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_vision_cli_registration(self):
        from multimodal_ml.__main__ import parse_args
        with patch("sys.argv", ["train", "--dataset", "sims", "--model", "vision_mlp_deep", "--vision-normalization", "none"]):
            args = parse_args()
        model = VisionMLPDeep.from_args(get_dataset_config("sims"), args)
        self.assertEqual(model.config.vision_dim, 709)
        self.assertTrue(model.normalizer_fitted)


if __name__ == "__main__":
    unittest.main()
