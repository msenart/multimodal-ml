"""Dataset-free checks of the uni-modal aids of ConcatFusion (multimodal_ml/unimodal.py)."""
import argparse
import unittest

import torch

from multimodal_ml import ogm_ge, unimodal
from multimodal_ml.models.concat_fusion import ConcatFusion
from multimodal_ml.seed import set_seed
from multimodal_ml.synthetic import make_loaders, synthetic_config
from tests.test_ogm_ge import CPU, namespace, small_model, train_once


class OptionTests(unittest.TestCase):
    def test_defaults_are_off(self):
        self.assertEqual(unimodal.UnimodalSettings.from_args(argparse.Namespace()), unimodal.UnimodalSettings())
        parser = argparse.ArgumentParser()
        ConcatFusion.add_cli_args(parser)
        self.assertEqual(unimodal.UnimodalSettings.from_args(parser.parse_args([])), unimodal.UnimodalSettings())
        s = unimodal.UnimodalSettings.from_args(parser.parse_args(
            ["--aux-weight", "0.5", "--text-dropout", "0.3", "--unimodal-pretrain", "freeze"]))
        self.assertEqual(s, unimodal.UnimodalSettings(0.5, 0.3, "freeze"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--unimodal-pretrain", "yes"])

    def test_zero_values_give_the_baseline(self):
        reference, _ = train_once("ogm")
        metrics, _ = train_once("ogm", aux_weight=0.0, text_dropout=0.0, unimodal_pretrain="none")
        self.assertEqual(metrics, reference)

    def test_each_option_changes_the_training(self):
        reference, _ = train_once("ogm")
        for kw in ({"aux_weight": 1.0}, {"text_dropout": 0.5}, {"unimodal_pretrain": "finetune"}):
            metrics, _ = train_once("ogm", **kw)
            self.assertNotEqual(metrics["mae"], reference["mae"], kw)


class MechanismTests(unittest.TestCase):
    def test_text_dropout_zeroes_whole_samples(self):
        torch.manual_seed(0)
        feats = {"text": torch.ones(10_000, 4), "audio": torch.ones(10_000, 4)}
        out = unimodal.drop_text(feats, 0.3)
        rows = out["text"].sum(1)
        self.assertTrue(torch.all((rows == 0) | (rows == 4)))
        self.assertAlmostEqual((rows == 0).float().mean().item(), 0.3, delta=0.02)
        self.assertIs(out["audio"], feats["audio"])

    def test_aux_loss_uses_each_encoder_regressor(self):
        set_seed(0)
        model = small_model()
        batch = next(iter(make_loaders("independent", 0)[0]))
        feats = model.encode(batch)
        expected = sum(torch.nn.functional.mse_loss(model.encoders[m].net[-1](feats[m]).squeeze(-1), batch.labels)
                       for m in feats) / len(feats)
        torch.testing.assert_close(unimodal.aux_loss(model.encoders, feats, batch.labels), expected)

    def test_freeze_copies_the_pretrained_encoders_and_trains_only_the_head(self):
        set_seed(3)
        model = small_model()
        loaders = make_loaders("independent", 3)
        args = namespace(epochs=2, unimodal_pretrain="freeze")
        unimodal.pretrain_encoders(model, loaders, CPU, args)
        pretrained = {k: v.clone() for k, v in model.encoders.state_dict().items()}
        head = model.head.head.weight.detach().clone()
        args.unimodal_pretrain = "none"  # pretraining done: now train the head only
        ogm_ge.train(model, loaders, CPU, args, verbose=False)
        for k, v in model.encoders.state_dict().items():
            torch.testing.assert_close(v, pretrained[k], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.head.head.weight, head))
        self.assertFalse(model.encoders.training)  # no dropout in frozen encoders

    def test_unimodal_copy_has_one_modality(self):
        model = ConcatFusion(synthetic_config("independent"), hidden_dim=8, num_layers=1)
        self.assertEqual(list(model.unimodal_copy("vision").encoders), ["vision"])


if __name__ == "__main__":
    unittest.main()
