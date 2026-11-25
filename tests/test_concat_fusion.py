from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_ml.config import get_dataset_config
from multimodal_ml.data import Batch, collate_fn
from multimodal_ml.metrics import compute_contribution_metrics
from multimodal_ml.training import evaluate
from multimodal_ml.models import MODEL_REGISTRY
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.models.sequence_mlp import SequenceMLP, _masked_stats
from multimodal_ml.models.text_mlp import TextMLP
from multimodal_ml.models.vision_mlp import VisionMLP
from multimodal_ml.models.vision_mlp_deep import VisionMLPDeep


class ConcatFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = get_dataset_config("mosi")
        lengths = torch.tensor([6, 3, 0, 1])
        self.batch = Batch(
            text=torch.randn(4, 2, self.config.text_dim),
            audio=torch.randn(4, 6, self.config.audio_dim),
            vision=torch.randn(4, 6, self.config.vision_dim),
            audio_lengths=lengths, vision_lengths=lengths,
            labels=torch.zeros(4), ids=[],
        )

    def test_forward_decomposes_and_updates_all_branches(self):
        # eval(): dropout would otherwise draw different masks for each encode call.
        model = ConcatFusion(self.config, vision_normalization="none").eval()
        prediction = model(self.batch)
        self.assertEqual(tuple(prediction.shape), (4,))
        torch.testing.assert_close(prediction, sum(model.contributions(self.batch).values()))
        (prediction - torch.linspace(-1, 1, 4)).square().mean().backward()
        self.assertEqual(list(model.encoders), ["text", "audio", "vision"])
        for layer in (model.encoders["text"].net[0], model.encoders["audio"].net[0],
                      model.encoders["vision"].net[0], model.head.head):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_text_none_ignores_text(self):
        model = ConcatFusion(self.config, text="none", vision_normalization="none").eval()
        self.assertEqual(list(model.encoders), ["audio", "vision"])
        nan_text = replace(self.batch, text=torch.full_like(self.batch.text, float("nan")))
        torch.testing.assert_close(model(nan_text), model(self.batch))

    def test_vision_encoder_choice(self):
        self.assertIsInstance(ConcatFusion(self.config).encoders["vision"], VisionMLP)
        ConcatFusion(self.config)(self.batch)  # basic encoder needs no fitting step
        deep = ConcatFusion(self.config, vision="vision-mlp-deep")
        self.assertIsInstance(deep.encoders["vision"], VisionMLPDeep)
        with self.assertRaisesRegex(RuntimeError, "training loader"):
            deep(self.batch)

    def test_none_drops_modality_and_gru_option(self):
        audio_only = ConcatFusion(self.config, text="none", audio="audio-gru", vision="none", hidden_dim=8).eval()
        self.assertEqual(list(audio_only.encoders), ["audio"])
        self.assertEqual(audio_only.head.feature_dims, {"audio": 16})  # bidirectional GRU: 2 * hidden
        nan_vision = replace(self.batch, vision=torch.full_like(self.batch.vision, float("nan")))
        torch.testing.assert_close(audio_only(nan_vision), audio_only(self.batch))

        vision_only = ConcatFusion(self.config, text="none", audio="none", vision_normalization="none").eval()
        self.assertEqual(list(vision_only.encoders), ["vision"])
        nan_audio = replace(self.batch, audio=torch.full_like(self.batch.audio, float("nan")))
        torch.testing.assert_close(vision_only(nan_audio), vision_only(self.batch))

        text_only = ConcatFusion(self.config, audio="none", vision="none").eval()
        self.assertEqual(list(text_only.encoders), ["text"])
        torch.testing.assert_close(text_only(replace(nan_audio, vision=nan_vision.vision)), text_only(self.batch))

        with self.assertRaisesRegex(ValueError, "at least one modality"):
            ConcatFusion(self.config, text="none", audio="none", vision="none")
        with self.assertRaises(ValueError):
            ConcatFusion(self.config, audio="transformer")
        with self.assertRaises(ValueError):
            ConcatFusion(self.config, text="bert")

    def test_cli_registration(self):
        from multimodal_ml.__main__ import parse_args
        self.assertIs(MODEL_REGISTRY["concat_fusion"], ConcatFusion)
        with patch("sys.argv", ["train", "--dataset", "sims", "--model", "concat_fusion",
                                "--hidden-dim", "16", "--vision", "vision-mlp-deep",
                                "--vision-normalization", "none"]):
            args = parse_args()
        model = ConcatFusion.from_args(get_dataset_config("sims"), args)
        self.assertEqual(model.head.feature_dims, {"text": 16, "audio": 16, "vision": 16})
        self.assertIsInstance(model.encoders["vision"], VisionMLPDeep)
        self.assertTrue(model.encoders["vision"].normalizer_fitted)
        with patch("sys.argv", ["train", "--dataset", "sims", "--model", "concat_fusion",
                                "--text", "none", "--audio", "audio-gru", "--vision", "none"]):
            args = parse_args()
        model = ConcatFusion.from_args(get_dataset_config("sims"), args)
        self.assertEqual(list(model.encoders), ["audio"])


class ContributionMetricsTests(unittest.TestCase):
    def test_synthetic_contributions(self):
        labels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        m = compute_contribution_metrics(
            {"text": labels.copy(), "audio": np.full(5, 0.5), "vision": -0.5 * labels}, labels)
        self.assertAlmostEqual(m["text_share"] + m["audio_share"] + m["vision_share"], 1.0)
        self.assertAlmostEqual(m["text_share"], 0.8)    # Var 2 vs 0.5 (audio constant)
        self.assertEqual(m["audio_share"], 0.0)
        self.assertEqual(m["audio_corr"], 0.0)
        self.assertAlmostEqual(m["text_corr"], 1.0)
        self.assertAlmostEqual(m["vision_corr"], -1.0)
        # text alone + audio mean (0.5) + vision mean (0) -> off by 0.5 everywhere
        self.assertAlmostEqual(m["text_mae"], 0.5)
        single = compute_contribution_metrics({"text": labels}, labels)
        self.assertEqual((single["text_share"], single["text_mae"]), (1.0, 0.0))

    def test_evaluate_reports_contributions_only_for_fusion(self):
        config = get_dataset_config("mosi")
        torch.manual_seed(0)
        samples = [{"text": torch.randn(2, config.text_dim).numpy(), "audio": torch.randn(3, config.audio_dim).numpy(),
                    "vision": torch.randn(3, config.vision_dim).numpy(), "audio_length": 3, "vision_length": 2,
                    "label": float(i) - 2.5, "id": str(i)} for i in range(6)]
        loader = DataLoader(samples, batch_size=4, collate_fn=collate_fn)
        model = ConcatFusion(config)
        metrics = evaluate(model, loader, torch.device("cpu"))
        self.assertTrue(model.training)  # evaluate restores train mode
        self.assertIn("mae", metrics)
        model.eval()
        with torch.no_grad():
            parts = [model.contributions(b) for b in loader]
        expected = compute_contribution_metrics(
            {n: torch.cat([p[n] for p in parts]).numpy() for n in ("text", "audio", "vision")},
            np.array([s["label"] for s in samples]))
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=5)
        self.assertNotIn("text_share", evaluate(TextMLP(config), loader, torch.device("cpu")))


class SequencePoolingTests(unittest.TestCase):
    def test_masked_stats_values_padding_and_empty_clip(self):
        x = torch.tensor([[[1.0], [3.0], [1e6]],      # 2 real frames + padding
                          [[5.0], [-7.0], [9.0]]])    # empty clip
        pooled = _masked_stats(x, torch.tensor([2, 0]))
        torch.testing.assert_close(pooled[0], torch.tensor([2.0, 1.0, 3.0]))  # mean, std, max
        torch.testing.assert_close(pooled[1], torch.tensor([0.0, 1e-4, 0.0]))

    def test_audio_and_vision_share_the_encoder(self):
        config = get_dataset_config("mosi")
        for cls, dim in ((AudioMLP, config.audio_dim), (VisionMLP, config.vision_dim)):
            self.assertTrue(issubclass(cls, SequenceMLP))
            mean = cls(config)
            # Unchanged layout for mean pooling, so earlier checkpoints still load.
            self.assertEqual(list(mean.state_dict()), ["net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"])
            self.assertEqual(mean.net[0].in_features, dim)
            self.assertEqual(cls(config, pooling="stats").net[0].in_features, 3 * dim)
        with self.assertRaises(ValueError):
            AudioMLP(config, pooling="max")

    def test_num_layers(self):
        config = get_dataset_config("mosi")
        for model in (AudioMLP(config, hidden_dim=8, num_layers=3), TextMLP(config, hidden_dim=8, num_layers=3)):
            self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model.net), 4)  # 3 hidden + regressor
            self.assertEqual(model.net[-1].in_features, 8)
        fusion = ConcatFusion(config, hidden_dim=8, num_layers=2, vision_normalization="none")
        for name in ("text", "audio", "vision"):
            self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in fusion.encoders[name].net), 3)
        with self.assertRaises(ValueError):
            VisionMLP(config, num_layers=0)

    def test_stats_gradients_finite_for_single_frame_and_constant_clips(self):
        config = get_dataset_config("mosi")
        model = AudioMLP(config, pooling="stats")
        audio = torch.randn(3, 4, config.audio_dim)
        audio[2] = 1.0  # constant clip: std 0
        batch = Batch(None, audio, None, torch.tensor([1, 3, 4]), None, torch.zeros(3), [])
        model(batch).square().sum().backward()
        self.assertTrue(torch.isfinite(model.net[0].weight.grad).all())

    def test_cli_pooling(self):
        from multimodal_ml.__main__ import parse_args
        with patch("sys.argv", ["train", "--dataset", "mosi", "--model", "vision_mlp", "--pooling", "stats"]):
            args = parse_args()
        self.assertEqual(VisionMLP.from_args(get_dataset_config("mosi"), args).pooling, "stats")
        with patch("sys.argv", ["train", "--dataset", "mosi", "--model", "concat_fusion", "--pooling", "stats"]):
            args = parse_args()
        model = ConcatFusion.from_args(get_dataset_config("mosi"), args)
        self.assertEqual((model.encoders["audio"].pooling, model.encoders["vision"].pooling), ("stats", "stats"))
        gru = ConcatFusion(get_dataset_config("mosi"), audio="audio-gru", pooling="stats")
        self.assertIsInstance(gru.encoders["audio"], AudioGRU)  # pooling only affects the shared MLP encoder
        with self.assertRaises(ValueError):
            ConcatFusion(get_dataset_config("mosi"), pooling="max")


class TextMLPTests(unittest.TestCase):
    def test_text_only_shapes_and_validation(self):
        config = get_dataset_config("mosi")
        model = TextMLP(config, hidden_dim=8).eval()
        batch = Batch(torch.randn(3, 5, config.text_dim), torch.full((3, 2, 5), float("nan")),
                      torch.full((3, 2, 20), float("nan")), torch.tensor([2, 2, 2]),
                      torch.tensor([2, 2, 2]), torch.zeros(3), [])
        self.assertEqual(tuple(model.encode(batch).shape), (3, 8))
        prediction = model(batch)
        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertTrue(torch.isfinite(prediction).all())  # NaN audio/vision never enter
        self.assertIs(MODEL_REGISTRY["text_mlp"], TextMLP)
        with self.assertRaises(ValueError):
            model(replace(batch, text=torch.randn(3, 5, 10)))


class VisionBasicMLPTests(unittest.TestCase):
    def test_only_real_frames_enter_and_empty_clip_pools_to_zero(self):
        config = get_dataset_config("mosi")
        model = VisionMLP(config, hidden_dim=8).eval()
        vision, lengths = torch.randn(3, 4, config.vision_dim), torch.tensor([4, 2, 0])
        batch = Batch(torch.full((3, 2, config.text_dim), float("nan")), torch.full((3, 2, 5), float("nan")),
                      vision, torch.tensor([2, 2, 2]), lengths, torch.zeros(3), [])
        changed = vision.clone()
        changed[1, 2:] = 1e6
        changed[2] = 1e6
        torch.testing.assert_close(model(replace(batch, vision=changed)), model(batch))
        empty = model.net[:-1](torch.zeros(1, config.vision_dim))
        torch.testing.assert_close(model.encode(batch)[2:], empty)
        self.assertEqual(tuple(model.encode(batch).shape), (3, 8))
        self.assertIs(MODEL_REGISTRY["vision_mlp"], VisionMLP)
        with self.assertRaises(ValueError):
            model(replace(batch, vision=torch.randn(3, 4, 7)))


if __name__ == "__main__":
    unittest.main()
