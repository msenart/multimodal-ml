"""Dataset-free checks of the OGM-GE implementation (multimodal_ml/ogm_ge.py)."""
import argparse
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from multimodal_ml import ogm_ge
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.seed import set_seed
from multimodal_ml.synthetic import make_loaders, synthetic_config
from multimodal_ml.training import default_routine

CPU = torch.device("cpu")


def small_model(scenario="independent"):
    return ConcatFusion(synthetic_config(scenario), hidden_dim=16, dropout=0.3, num_layers=2)


def namespace(**kw):
    base = dict(dataset="synthetic", model="concat_fusion", epochs=3, lr=1e-3, patience=5, batch_size=16)
    return argparse.Namespace(**{**base, **kw})


def train_once(routine, seed=7, **kw):
    """Same seed -> same initialisation, same data and same shuffling for every routine."""
    set_seed(seed)
    model = small_model()
    loaders = make_loaders("independent", seed)
    with tempfile.TemporaryDirectory() as tmp:
        args = namespace(checkpoint_dir=Path(tmp), **kw)
        if routine == "default":
            metrics = default_routine(model, loaders, CPU, args)
        else:
            metrics, _ = ogm_ge.train(model, loaders, CPU, args, verbose=False)
    return metrics, model


class FormulaTests(unittest.TestCase):
    def test_scores_are_gaussian_likelihoods_of_each_modality_alone(self):
        parts = {"a": torch.tensor([1.0, 3.0]), "b": torch.tensor([0.0, 2.0])}
        labels = torch.tensor([2.0, 4.0])
        scores = ogm_ge.modality_scores(parts, labels)
        # a alone: c_a + mean(c_b) = [2, 4] (exact); b alone: c_b + mean(c_a) = [2, 4] (exact too)
        self.assertAlmostEqual(scores["a"].item(), 2.0)
        self.assertAlmostEqual(scores["b"].item(), 2.0)
        scores = ogm_ge.modality_scores({"a": torch.tensor([1.0, 3.0]), "b": torch.tensor([0.0, 0.0])}, labels)
        self.assertAlmostEqual(scores["b"].item(), 1 + math.exp(-2), places=6)  # b alone predicts [2, 2]

    def test_coefficients_match_the_paper_for_two_modalities(self):
        scores = {"a": torch.tensor(3.0), "b": torch.tensor(1.5)}
        rho, k = ogm_ge.modulation_coefficients(scores, alpha=0.5)
        self.assertAlmostEqual(rho["a"], 2.0)
        self.assertAlmostEqual(rho["b"], 0.5)
        self.assertAlmostEqual(k["a"], 1 - math.tanh(0.5 * (2.0 - 1)))
        self.assertEqual(k["b"], 1.0)  # only the dominant modality is slowed down

    def test_three_modalities_compare_with_the_best_other_modality(self):
        scores = {"t": torch.tensor(4.0), "a": torch.tensor(1.0), "v": torch.tensor(3.0)}
        rho, k = ogm_ge.modulation_coefficients(scores, alpha=1.0)
        self.assertAlmostEqual(rho["t"], 4.0 / 3.0)
        self.assertAlmostEqual(rho["v"], 3.0 / 4.0)
        self.assertEqual((k["a"], k["v"]), (1.0, 1.0))  # only the best modality is slowed down
        self.assertLess(k["t"], 1.0)

    def test_three_modalities_mean_ratio(self):
        scores = {"t": torch.tensor(4.0), "a": torch.tensor(1.0), "v": torch.tensor(3.0)}
        rho, k = ogm_ge.modulation_coefficients(scores, alpha=1.0, ratio="mean")
        self.assertAlmostEqual(rho["t"], 4.0 / 2.0)
        self.assertAlmostEqual(rho["a"], 1.0 / 3.5)
        self.assertAlmostEqual(rho["v"], 3.0 / 2.5)
        self.assertEqual(k["a"], 1.0)
        self.assertLess(k["t"], k["v"])
        self.assertLess(k["v"], 1.0)

    def test_alpha_zero_gives_no_modulation(self):
        _, k = ogm_ge.modulation_coefficients({"t": torch.tensor(9.0), "a": torch.tensor(1.0)}, alpha=0.0)
        self.assertEqual(k, {"t": 1.0, "a": 1.0})

    def test_modulation_window(self):
        settings = ogm_ge.OGMSettings(mode="ogm", start=2, end=4)
        self.assertEqual([settings.active(e) for e in range(6)], [False, False, True, True, True, False])
        self.assertFalse(ogm_ge.OGMSettings(mode="none").active(0))


class GradientTests(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.model = small_model().eval()
        batch = next(iter(make_loaders("independent", 0)[0]))
        F.mse_loss(self.model(batch), batch.labels).backward()
        # Each encoder's own regressor is unused by the fusion, so it has no gradient.
        self.before = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}

    def test_modulation_scales_only_the_encoder_of_each_modality(self):
        k = {"text": 0.25, "audio": 1.0, "vision": 1.0}
        ogm_ge.modulate_gradients(dict(self.model.encoders.items()), k, ge=False)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            factor = 0.25 if name.startswith("encoders.text.") else 1.0  # fusion head never modulated
            torch.testing.assert_close(p.grad, factor * self.before[name], rtol=0, atol=0)

    def test_ge_adds_zero_mean_noise_with_the_gradient_std_on_weight_matrices(self):
        layer = torch.nn.Linear(1000, 1000)
        layer.weight.grad = torch.randn(1000, 1000) * 3.0
        layer.bias.grad = torch.randn(1000)
        weight, bias = layer.weight.grad.clone(), layer.bias.grad.clone()
        ogm_ge.modulate_gradients({"m": layer}, {"m": 1.0}, ge=True)
        noise = layer.weight.grad - weight
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(noise.std().item(), weight.std().item(), delta=0.01)
        torch.testing.assert_close(layer.bias.grad, bias, rtol=0, atol=0)  # 1-D parameters: no noise


class OptimizerTests(unittest.TestCase):
    """Why the optimizer matters: Adam normalises the gradient, SGD does not."""

    def step_update(self, optimizer: str, k: float) -> torch.Tensor:
        set_seed(0)
        model = small_model()
        batch = next(iter(make_loaders("independent", 0)[0]))
        model.eval()  # same output for both calls
        opt = ogm_ge.make_optimizer(model.parameters(), 1e-2, ogm_ge.OGMSettings(optimizer=optimizer, momentum=0.0))
        F.mse_loss(model(batch), batch.labels).backward()
        ogm_ge.modulate_gradients({"text": model.encoders["text"]}, {"text": k}, ge=False)
        before = ogm_ge._flat(model.encoders["text"])
        opt.step()
        return ogm_ge._flat(model.encoders["text"]) - before

    def test_sgd_update_is_scaled_by_k(self):
        torch.testing.assert_close(self.step_update("sgd", 0.3), 0.3 * self.step_update("sgd", 1.0))

    def test_adam_update_ignores_a_constant_k(self):
        # Adam's first step is lr * g / (|g| + eps): multiplying g by k changes nothing, except for
        # the few weights whose gradient is about as small as eps.
        ratio = self.step_update("adam", 0.3).norm() / self.step_update("adam", 1.0).norm()
        self.assertAlmostEqual(ratio.item(), 1.0, delta=0.01)


class RoutineTests(unittest.TestCase):
    def test_ogm_none_is_the_default_routine(self):
        reference, ref_model = train_once("default")
        metrics, model = train_once("ogm", ogm_mode="none")
        self.assertEqual(metrics, reference)
        for p, q in zip(ref_model.parameters(), model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_alpha_zero_without_ge_is_the_baseline(self):
        reference, _ = train_once("ogm", ogm_mode="none")
        metrics, _ = train_once("ogm", ogm_mode="ogm", ogm_alpha=0.0)
        self.assertEqual(metrics, reference)

    def test_switches_change_the_training(self):
        reference, _ = train_once("ogm", ogm_mode="none")
        for kw in ({"ogm_mode": "ogm", "ogm_alpha": 1.0}, {"ogm_mode": "ogm-ge", "ogm_alpha": 0.0},
                   {"optimizer": "sgd"}):
            metrics, _ = train_once("ogm", **kw)
            self.assertNotEqual(metrics["mae"], reference["mae"], kw)

    def test_dominant_text_is_slowed_down_on_synthetic_data(self):
        set_seed(1)
        model = ConcatFusion(synthetic_config("independent"), hidden_dim=64, dropout=0.0, num_layers=2)
        args = namespace(epochs=4, lr=1e-2, ogm_mode="ogm", ogm_alpha=1.0, optimizer="sgd")
        _, history = ogm_ge.train(model, make_loaders("independent", 1), CPU, args, verbose=False)
        k = {m: np.mean([h[f"k_{m}"] for h in history]) for m in ("text", "audio", "vision")}
        self.assertLess(k["text"], 1.0)
        self.assertLess(k["text"], min(k["audio"], k["vision"]))


class CliTests(unittest.TestCase):
    def parse(self, argv):
        parser = argparse.ArgumentParser()
        ConcatFusion.add_cli_args(parser)
        return ogm_ge.OGMSettings.from_args(parser.parse_args(argv))

    def test_defaults_are_plain_adam_training(self):
        self.assertEqual(self.parse([]), ogm_ge.OGMSettings())
        self.assertEqual(ogm_ge.OGMSettings.from_args(argparse.Namespace()), ogm_ge.OGMSettings())
        self.assertEqual(ogm_ge.OGMSettings().mode, "none")

    def test_every_option_can_be_set(self):
        s = self.parse(["--ogm", "ogm-ge", "--ogm-alpha", "0.5", "--ogm-ratio", "mean", "--ogm-start", "1",
                        "--ogm-end", "3", "--optimizer", "sgd", "--momentum", "0.5", "--weight-decay", "1e-4"])
        self.assertEqual(s, ogm_ge.OGMSettings("ogm-ge", 0.5, "mean", 1, 3, "sgd", 0.5, 1e-4))
        with self.assertRaises(SystemExit):
            self.parse(["--ogm", "on"])


if __name__ == "__main__":
    unittest.main()
