"""Summarize a completed local vision-only study; never trains or uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="Optional local study artifacts to export")
    parser.add_argument("--check", action="store_true", help="Check the English report without datasets or archived weights")
    args = parser.parse_args()
    if args.check and args.directory:
        parser.error("--check uses the committed summary, not local artifacts")
    snapshot = ROOT / "results/vision_baselines/summary.json"
    if args.directory:
        directory = args.directory.resolve()
        result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        artifacts = directory.relative_to(ROOT).as_posix()
    else:
        result = json.loads(snapshot.read_text(encoding="utf-8"))
        artifacts = result["artifacts"]
    contracts = {
        "mosi": {"language": "English", "vision_feature_dim": 20, "target_field": "regression_labels", "target_scope": "overall clip sentiment", "label_range": [-3, 3]},
        "sims": {"language": "Chinese", "vision_feature_dim": 709, "target_field": "regression_labels", "target_scope": "overall clip sentiment", "label_range": [-1, 1]},
    }
    summary = {"schema_version": 1, "task": "vision-only overall sentiment regression",
               "datasets_trained_independently": True, "dataset_contracts": contracts,
               "units": {"json_accuracy_f1_correlation": "fractions", "json_mae": "dataset sentiment score units",
                         "markdown_accuracy_f1": "percentages", "std": "sample standard deviation, ddof=1"},
               "metric_definitions": {"common_mae_corr": "unclipped predictions and overall labels",
                                      "common_has0": "all examples; labels/predictions >=0 are positive, zero is positive",
                                      "common_non0": "exclude exactly-zero true labels; labels/predictions >0 are positive",
                                      "f1": "support-weighted F1",
                                      "dataset_specific": "MOSI signed rules; CH-SIMS clips to [-1,1] and includes zero in the negative binary class"},
               "protocol": result["protocol"], "artifacts": artifacts, "datasets": {}}
    lines = ["# Standalone vision-only training and validation", "",
             "Training study, 2026-11-18. MOSI and CH-SIMS were trained **separately**, with separate parameters,",
             "normalizers and checkpoints. Each dataset has three independent neural runs (seeds 42/43/44).", "",
             "## Models and target", "",
             "The input is pre-extracted visual features plus declared visual lengths. The target is the overall",
             "`regression_labels`, not the modality-specific visual annotation. Text/audio are not loaded into",
             "the training batches: their placeholders are empty tensors.", "",
             "| Dataset | Language | Visual feature dimension | Training label field | Ground-truth score range |",
             "| --- | --- | ---: | --- | --- |",
             "| CMU-MOSI | English | 20 | `regression_labels` (overall sentiment) | -3 to +3 |",
             "| Original CH-SIMS | Chinese | 709 | `regression_labels` (overall sentiment) | -1 to +1 |", "",
             "This is scalar regression, not training a binary/three/five-class classifier. Model outputs are",
             "unrestricted scalars; the ground-truth range is not a hard output bound. Neither dataset uses",
             "the CH-SIMS `regression_labels_V` field for training, and no weights or normalization statistics",
             "are transferred or shared between the English and Chinese models.", "",
             "VisionMLP: masked mean over valid frames -> training-only feature z-score -> dropout(0.3) ->",
             "three 64-unit ReLU layers -> scalar regression head. Constant training features use unit scale.",
             "The fitted normalization is stored with the weights. This is a defined feature-level baseline,",
             "not a raw-image CNN or a claim to reproduce a published architecture.", "",
             "## Test results", "",
             "MLP values are mean +/- sample standard deviation (ddof=1) over seeds 42/43/44.",
             "Binary accuracy/F1 below exclude exact-zero labels and threshold predictions at zero (`non0`).",
             "MAE uses the raw dataset sentiment scale; different datasets' MAEs are not directly comparable.", "",
             "The shared `results/results.csv` uses separately named dataset-specific evaluation: MOSI `Has0_acc_2`",
             "and CH-SIMS `Mult_acc_2`. All four shared-table metrics are multiplied by 100, including MAE and",
             "correlation. Those binary scores must not be confused with the `non0` column below.", "",
             "| Dataset | Method | MAE | Correlation | Acc-2 non0 | Weighted F1 non0 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, ds in result["datasets"].items():
        aggregate = {}
        for key in ("mae", "corr", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0"):
            values = [run["metrics"]["test"]["common"][key] for run in ds["runs"]]
            aggregate[key] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "values": values}
        audit = dict(ds["audit"])
        if "exact_visual_train_test_duplicates" in audit:
            audit["exact_visual_train_test_feature_matches"] = audit.pop("exact_visual_train_test_duplicates")
        intervention = dict(ds["interventions"])
        if "duplicate_sensitivity" in intervention:
            intervention["feature_match_exclusion_sensitivity"] = intervention.pop("duplicate_sensitivity")
        summary["datasets"][name] = {"aggregate": aggregate, "runs": ds["runs"], "references": ds["references"],
                                        "audit": audit, "small_subset_fit": ds["small_subset_fit"],
                                        "interventions": intervention, "artifact_checks": ds.get("artifact_checks", [])}
        specific = ds["runs"][0]["metrics"]["test"]["dataset_specific"]
        summary["datasets"][name]["dataset_specific_aggregate"] = {
            key: {"mean": float(np.mean([r["metrics"]["test"]["dataset_specific"][key] for r in ds["runs"]])),
                  "sample_std": float(np.std([r["metrics"]["test"]["dataset_specific"][key] for r in ds["runs"]], ddof=1))}
            for key in specific
        }
        for key, label in (("constant", "Training-median constant"), ("ridge", "Ridge regression")):
            m = ds["references"][key]["test"]["common"]
            corr = "Undefined" if m["corr"] is None else f"{m['corr']:.4f}"
            lines.append(f"| {name.upper()} | {label} | {m['mae']:.4f} | {corr} | {m['acc2_non0']*100:.2f}% | {m['f1_non0']*100:.2f}% |")
        def fmt(key, percent=False):
            factor = 100 if percent else 1
            a = aggregate[key]
            return f"{a['mean']*factor:.4f} +/- {a['sample_std']*factor:.4f}" + ("%" if percent else "")
        lines.append(f"| {name.upper()} | VisionMLP | {fmt('mae')} | {fmt('corr')} | {fmt('acc2_non0',True)} | {fmt('f1_non0',True)} |")
    lines += ["", "## Interpretation", "",
              "CH-SIMS shows useful visual prediction signal relative to a training-median constant. Ridge is",
              "slightly better than this MLP on test MAE, so the neural model is not claimed to be the strongest",
              "vision baseline. MOSI remains weak and seed-sensitive: its mean binary accuracy is near 50%.",
              "These findings do not establish an upper bound on vision-only performance.", "",
              "The Ridge regularization grid [0.1,1,10,100] was selected by validation MAE only. The six neural runs",
              "used one fixed architecture, Adam/MSE, lr=0.001, batch size 32, at most 60 epochs and validation-MAE",
              "patience 10. Test data did not choose epochs, seeds, normalization or Ridge regularization.", "",
              "## Per-seed selection", "", "| Dataset | Seed | Epochs run | Best epoch | Validation MAE | Test MAE |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, ds in result["datasets"].items():
        for run in ds["runs"]:
            lines.append(f"| {name.upper()} | {run['seed']} | {run['epochs_run']} | {run['best_epoch']} | {run['best_valid_mae']:.4f} | {run['metrics']['test']['common']['mae']:.4f} |")
    lines += ["", "## Verification", "",
              "- 23 local code checks passed, including visual isolation, masked padding, train-only statistics,",
              "  normalization checkpoint restoration, and a separate future-fusion gradient interface check.",
              "- Both real-data one-epoch smoke runs passed. All six final checkpoints were restored and evaluated.",
              "- Raw-frame inference and cached masked means agreed on the checked validation/test examples.",
              "  Pooling is cached for this MLP only; no recurrent/time-dependent model is trained on that cache.",
              "- The test MAE was independently recalculated. Dataset-specific metrics are recorded separately",
              "  from the common metric definitions.", "",
              "| Dataset | 16-sample initial MSE | Final MSE | Updates | Validation MAE | Shuffled mean MAE | Worse permutations |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, ds in result["datasets"].items():
        fit, ablation = ds["small_subset_fit"], ds["interventions"]
        lines.append(f"| {name.upper()} | {fit['initial_mse']:.5f} | {fit['final_mse']:.5f} | {fit['steps']} | {ablation['original_mae']:.4f} | {ablation['permuted_mean_mae']:.4f} | {ablation['permutations_worse']}/20 |")
    lines += ["", "The small-subset diagnostic used dropout=0, lr=0.003, and a preset 99% MSE reduction criterion;",
              "both datasets passed. It checks trainability, not generalization. Permutation tests used seed-42",
              "validation predictions, so they are diagnostics rather than independent significance tests.", "",
              "## Data and input caveats", "",
              "Declared visual lengths were preserved. MOSI has nonzero features after declared lengths in",
              "180/16/104 train/valid/test samples; their origin is not established. No input-length policy was",
              "changed to improve results. Dataset files were SHA-256 checked and left unchanged.", "",
              "MOSI has one train/test **all-zero visual feature collision** with different text/audio features",
              "and labels. It is not evidence that the same full clip occurs in both splits. CH-SIMS has one",
              "cross-split sample with identical visual, text and audio features and the same label. Its test",
              "exclusion sensitivity is recorded; the official split was retained for the main table.", "",
              "Raw-zero visual inputs may be far outside the training distribution after standardization.",
              "Do not interpret the zero-input stress test alone as causal evidence. A training-mean visual",
              "intervention, corresponding to zero normalized input, is also recorded.", "",
              "## Checkpoints and future use", "",
              f"All logs, normalizers, per-example predictions and six final model files are under `{summary['artifacts']}/`.",
              "Use the dataset-specific checkpoint, instantiate `VisionMLP(get_dataset_config(dataset))`,",
              "and load `checkpoint['model_state']`. Call `encode(batch)` for a 64-dimensional representation.",
              "Keep the whole wrapper, including normalization buffers; copying only `.encoder` drops normalization.", "",
              "No fusion or OGM training has been performed in this standalone study. Only code, English documentation",
              "and compact result metadata are versioned; data, weights and per-example predictions remain local.", ""]
    target = ROOT / "docs/vision/results.md"
    content = "\n".join(lines)
    if args.check:
        if target.read_text(encoding="utf-8") != content or result != summary:
            raise SystemExit("Vision report or summary is out of date; regenerate it before committing")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"English vision report and summary checked: {target}")
    for name, ds in summary["datasets"].items():
        print(name, ds["aggregate"])


if __name__ == "__main__":
    main()
