"""Export recorded audio experiments and render the English team handoff.

Without a directory, rebuild documentation from committed JSON summaries only.
With a local validation directory, export metadata and verify available saved
predictions first. Feature pickles, predictions and weights are never exported.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from multimodal_ml.reference_metrics import compute_mmsa_metrics

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/audio_baselines"
DOCS = ROOT / "docs/audio"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def artifact_path(path):
    return ROOT / Path(str(path).replace("\\", "/"))


def export(directory):
    raw, summary = read(directory / "results.json"), read(directory / "summary.json")
    metrics = {"schema_version": 1, "experiment_date": "2026-11-11", "seeds": [42, 43, 44],
               "std_definition": "sample standard deviation, ddof=1",
               "metric_units": "raw: accuracy/F1/correlation are fractions, MAE is in dataset score units",
               "protocol": raw["protocol"], "reference": raw["reference"], "datasets": {}}
    validation = {"schema_version": 1, "historical_unit_checks_passed": 13,
                  "official_prediction_checks": 0, "datasets": {}}
    constants = read(ROOT / "checkpoints/audio_constant_baselines_seed42.json")
    for name, ds in summary["datasets"].items():
        entry = {"constant_baseline": constants[name], "models": {}}
        for model in ("audio_gru", "audio_mlp"):
            runs = []
            for run in ds[model]["runs"]:
                original = directory / name / "original_predictions.npz"
                file = original if run["seed"] == 42 and model == "audio_gru" else artifact_path(run["checkpoint"]).parent / "test_predictions.npz"
                with np.load(file) as predictions:
                    official = compute_mmsa_metrics(predictions["predictions"], predictions["labels"], name)
                expected = run["official_metrics_recomputed"]
                assert set(official) == set(expected)
                assert all(abs(official[k] - expected[k]) <= 1e-4 for k in official)
                validation["official_prediction_checks"] += 1
                runs.append({"seed": run["seed"], "best_valid_mae": run["best_valid_mae"],
                             "metrics_common": run["test"], "metrics_mmsa": official})
            official_aggregate = {key: {"mean": float(np.mean([r["metrics_mmsa"][key] for r in runs])),
                                        "sample_std": float(np.std([r["metrics_mmsa"][key] for r in runs], ddof=1))}
                                  for key in runs[0]["metrics_mmsa"]}
            entry["models"][model] = {"common_aggregate": ds[model]["aggregate"],
                                        "mmsa_aggregate": official_aggregate, "runs": runs}
        metrics["datasets"][name] = entry
        source = raw["datasets"][name]
        audit, ablation = source["audit"], source["ablations"]
        v = {"sha256": audit["sha256"], "samples": {k: v["samples"] for k, v in audit["splits"].items()},
             "extended_length_samples": {k: v["extended_lengths"] for k, v in audit["splits"].items()},
             "id_overlap": audit["id_overlap"], "exact_audio_overlap": audit["exact_audio_overlap"],
             "checkpoint_checks": {k: source["checkpoint"][k] for k in ("saved_metrics_reproduced", "independent_metric_check", "loader_check", "packed_batch_order_check")},
             "original_seed42_mmsa": source["checkpoint"]["official_mmsa_metrics"],
             "small_fit": {},
             "ablations": {k: ablation[k] for k in ("split", "original", "mean_shuffled_mae", "shuffle_worse_count", "zero_audio_same_lengths", "declared_length_evaluation_same_checkpoint", "train_median_baseline", "clip_bootstrap_95_percent_interval")},
             "standardization_valid_mae": source["standardization_validation_diagnostic"]["best_valid_mae"],
             "standardization_test_evaluated": False}
        v["ablations"]["permutation_maes"] = [p["mae"] for p in ablation["permutations"]]
        for key in ("raw", "standardized"):
            fit = source[f"small_fit_{key}"]
            v["small_fit"][key] = {k: fit[k] for k in ("diagnostic_version", "steps", "initial_mse", "final_mse", "relative_mse", "passed_99_percent_mse_reduction", "samples", "dropout", "lr")}
        sensitivity_file = directory / name / "duplicate_audio_sensitivity.json"
        if sensitivity_file.exists():
            sensitivity = read(sensitivity_file)
            for row in sensitivity["predictions"]:
                row["run"] = "audio_gru_raw_seed42" if "original_predictions" in row["source"] else Path(row["source"].replace("\\", "/")).parent.name
                del row["source"]
            v["duplicate_sensitivity"] = sensitivity
        validation["datasets"][name] = v
    assert validation["official_prediction_checks"] == 12
    write(PUBLISHED / "metrics.json", metrics)
    write(PUBLISHED / "validation.json", validation)


def format_average(entry, percentage=False):
    factor = 100 if percentage else 1
    suffix = "%" if percentage else ""
    return f"{entry['mean']*factor:.4f} +/- {entry['sample_std']*factor:.4f}{suffix}"


def render(metrics, validation):
    lines = ["# Audio-only baseline results", "",
             "These are exploratory custom audio-only regressors trained on **overall sentiment labels**,",
             "not CH-SIMS audio-specific labels. The experiments ran on 2026-11-11 using CPU PyTorch 2.14.1",
             "and Python 3.13.7. They are not claimed to reproduce a published MMSA architecture.", "",
             "## Three-seed comparison", "",
             "Mean +/- sample standard deviation over seeds **42, 43, 44** (ddof=1). Lower MAE is better.",
             "The common binary metrics below exclude exactly-zero ground-truth labels (`non0`).", "",
             "| Dataset | Model | MAE | Correlation | Acc-2 non0 | Weighted F1 non0 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, ds in metrics["datasets"].items():
        baseline = ds["constant_baseline"]
        lines.append(f"| {name.upper()} | Training-median constant | {baseline['constant_train_median_test_mae']:.4f} | Undefined | {baseline['non0']['test_accuracy']*100:.2f}% | {baseline['non0']['test_weighted_f1']*100:.2f}% |")
        for model in ("audio_mlp", "audio_gru"):
            a = ds["models"][model]["common_aggregate"]
            lines.append(f"| {name.upper()} | {model} | {format_average(a['mae'])} | {format_average(a['corr'])} | {format_average(a['acc2_non0'], True)} | {format_average(a['f1_non0'], True)} |")
    lines += ["", "## Interpretation", "",
              "The saved results can be independently recalculated and the models respond to correctly paired audio.",
              "However, AudioGRU is worse than the simpler MLP in all three MOSI runs. On CH-SIMS its average",
              "MAE is slightly better, with variation across seeds. This supports a **weak custom baseline**,",
              "not a claim that audio has little value or that this architecture is competitive.", "",
              "Three seeds are insufficient for a strong significance claim. The teammates' MMSA experiments use",
              "different seeds (1111-1113), implementations and preprocessing; cross-source rows are descriptive comparisons.", "",
              "## Training settings", "",
              "Adam, MSE loss, learning rate 0.001, batch size 32, hidden dimension 64, dropout 0.3,",
              "at most 60 epochs, and patience 10 on **validation MAE**. Each test score uses that run's best",
              "validation checkpoint. Test scores did not select seeds, thresholds or normalization settings.", "",
              "The reported runs use the custom `observed` audio-length policy. The repository default remains",
              "`declared`; pass `--audio-length-policy observed` to reproduce this contribution.", "",
              "## Metric definitions and shared table units", "",
              "Common `has0` assigns zero to the positive class (`>= 0`); `non0` excludes zero labels and uses `> 0`.",
              "MMSA MOSI uses both rules; MMSA CH-SIMS binary evaluation includes zero in the **negative** class.",
              "MMSA also uses dataset-specific multi-class bins. Keep common and official metrics separately named.", "",
              "`results/audio_baselines/metrics.json` stores raw score units and fractions. The shared",
              "`results/results.csv` follows the existing MMSA convention of multiplying **all four metrics by 100**,",
              "including MAE and correlation. The four `custom_audio` rows use MMSA-compatible metrics and ddof=1.", "",
              "For example, shared-table MOSI MAE around 144 means raw MAE around 1.44, not an error of 144 sentiment points.", "",
              "## Per-seed MAE", "", "| Dataset | Model | 42 | 43 | 44 |", "| --- | --- | ---: | ---: | ---: |"]
    for name, ds in metrics["datasets"].items():
        for model in ("audio_mlp", "audio_gru"):
            values = ds["models"][model]["common_aggregate"]["mae"]["values"]
            lines.append(f"| {name.upper()} | {model} | " + " | ".join(f"{x:.4f}" for x in values) + " |")
    lines += ["", "See [validation](validation.md), [design and integration](design.md), and the",
              "[machine-readable evidence](../../results/audio_baselines/README.md).", ""]

    vlines = ["# Audio baseline validation", "",
              "## Verdict", "",
              "The checks support reproducible weak custom baselines that use some audio information.",
              "They do not prove the absence of every bug, establish strong predictive performance, or constitute",
              "a full reproduction of MMSA. Original weights and data were retained during the audit.", "",
              "## Implementation and data checks", "",
              "The historical audit passed 13 synthetic checks, including padding, gradient flow, modality isolation,",
              "packed-sequence sample order, checkpoint restoration and metric boundaries. The current PR adds",
              "merge-safety and fresh-checkout checks. CI runs without downloading datasets.", "",
              "Saved-checkpoint MAE/correlation/binary metrics were independently recalculated. A separate",
              "NumPy implementation agreed with the shared metrics. Twelve saved test prediction files were checked",
              "against pinned MMSA definitions. Re-evaluating a checkpoint is not a claim to replay every original training step.", "",
              "| Dataset | Train / valid / test | ID overlap across splits | Exact stored-audio overlap |",
              "| --- | --- | --- | --- |"]
    for name, ds in validation["datasets"].items():
        counts = ds["samples"]
        vlines.append(f"| {name.upper()} | {counts['train']} / {counts['valid']} / {counts['test']} | {ds['id_overlap']} | {ds['exact_audio_overlap']} |")
    vlines += ["", "These checks do not establish speaker independence or exclude near-duplicates.", "",
               "## Small-subset fitting", "",
               "Diagnostic only: 16 training samples, dropout disabled, lr=0.003, up to 600 updates.",
               "The fixed criterion was at least 99% reduction in MSE, evaluated **after** the optimizer update.",
               "Both datasets passed with raw and training-standardized features; this checks trainability, not generalization.", "",
               "| Dataset | Features | Initial MSE | Final MSE | Updates |", "| --- | --- | ---: | ---: | ---: |"]
    for name, ds in validation["datasets"].items():
        for variant, fit in ds["small_fit"].items():
            vlines.append(f"| {name.upper()} | {variant} | {fit['initial_mse']:.6f} | {fit['final_mse']:.6f} | {fit['steps']} |")
    vlines += ["", "## Validation-set interventions", "",
               "Audio and its length were shuffled together while labels stayed fixed. Every one of 20 permutations",
               "increased MAE on each dataset. This supports use of paired audio, not a strong causal interpretation.",
               "Zero-audio evaluation retained lengths and therefore retained possible duration information.", "",
               "| Dataset | Original MAE | Mean shuffled MAE | Zero-audio MAE | Constant MAE |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for name, ds in validation["datasets"].items():
        a = ds["ablations"]
        vlines.append(f"| {name.upper()} | {a['original']['mae']:.4f} | {a['mean_shuffled_mae']:.4f} | {a['zero_audio_same_lengths']['mae']:.4f} | {a['train_median_baseline']['mae']:.4f} |")
    vlines += ["", "Stored clip-bootstrap intervals are exploratory: this validation set selected epochs, and the intervals",
               "do not model dependence between clips from the same speaker. They are not independent significance tests.", "",
               "## Normalization and length-policy diagnostics", "",
               "Standardization used means/stds from non-padding **training frames only**; padding stayed zero.",
               "These seed-42 experiments were evaluated only on validation data, not on the test set.", "",
               "| Dataset | Raw-feature validation MAE | Standardized validation MAE |",
               "| --- | ---: | ---: |"]
    for name, ds in validation["datasets"].items():
        vlines.append(f"| {name.upper()} | {ds['ablations']['original']['mae']:.4f} | {ds['standardization_valid_mae']:.4f} |")
    vlines += ["", "Standardization helped MOSI validation but worsened CH-SIMS validation under the fixed settings.",
               "Feature scale alone is therefore not an established explanation for the weak results.", "",
               "MOSI has 43/5/23 train/valid/test samples with nonzero features after the declared length.",
               "The custom `observed` policy retains them; MMSA uses the declared lengths. The origin of the extra",
               "frames is not established, so this must not be called a confirmed official-data bug fix.",
               "The default shared loader preserves declared lengths; use the explicit flag for the recorded audio experiments.", "",
               "## CH-SIMS cross-split duplicate", "",
               "One official test sample has the same audio, text and visual features and overall label as a training",
               "sample, despite a different clip ID. The feature-file SHA-256 matches the reference release.",
               "This dataset-level overlap is disclosed; the main scores retain the official split.", ""]
    duplicate = validation["datasets"]["sims"].get("duplicate_sensitivity")
    if duplicate:
        for match in duplicate["matches"]:
            vlines.append(f"- Train `{match['train_id']}` and test `{match['test_id']}`, overall label {match['test_label']}.")
        vlines += ["", "| Run | Original test MAE | Excluding the duplicate test sample |", "| --- | ---: | ---: |"]
        for row in duplicate["predictions"]:
            vlines.append(f"| {row['run']} | {row['original_mae']:.6f} | {row['excluded_mae']:.6f} |")
    vlines += ["", "Each change was below 0.0003 MAE and did not change the comparison. The team should use one documented",
               "duplicate policy across models; do not silently change each model's split independently.", "",
               "## Reference and evidence", ""]
    commit = metrics["reference"]["commit"]
    vlines += [f"MMSA reference commit: [`{commit[:12]}`](https://github.com/thuiar/MMSA/tree/{commit}).",
               "The committed JSON includes source URLs/hashes, per-seed scores, intervention summaries and diagnostic settings.",
               "Weights, feature files and per-example predictions stay under ignored local artifact directories.", "",
               "See the [reproduction guide](README.md) and [results](results.md).", ""]
    return {DOCS / "results.md": "\n".join(lines), DOCS / "validation.md": "\n".join(vlines)}


def update_shared(metrics, by):
    table = ROOT / "results/results.csv"
    with table.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    for name, ds in metrics["datasets"].items():
        for model, values in ds["models"].items():
            rows = [row for row in rows if (row["dataset"], row["model"], row["variant"], row["source"]) != (name, model, "A", "custom_audio")]
            row = {key: "" for key in columns}
            row.update(dataset=name, model=model, variant="A", source="custom_audio", seeds="42,43,44", by=by,
                       date=metrics["experiment_date"], notes="MMSA-compatible metrics x100; std ddof=1; observed lengths; see docs/audio")
            names = {"acc2": "Has0_acc_2" if name == "mosi" else "Mult_acc_2",
                     "f1": "Has0_F1_score" if name == "mosi" else "F1_score", "mae": "MAE", "corr": "Corr"}
            for column, key in names.items():
                aggregate = values["mmsa_aggregate"][key]
                row[column] = f"{aggregate['mean']*100:.4f}"
                row[column + "_std"] = f"{aggregate['sample_std']*100:.4f}"
            rows.append(row)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="Optional local audit artifacts to export")
    parser.add_argument("--check", action="store_true", help="Check committed reports without modifying them")
    parser.add_argument("--update-shared-table", action="store_true")
    parser.add_argument("--by", help="Contributor name for the shared table")
    args = parser.parse_args()
    if args.directory:
        if args.check:
            parser.error("--check cannot export local artifacts")
        export(args.directory.resolve())
    metrics, validation = read(PUBLISHED / "metrics.json"), read(PUBLISHED / "validation.json")
    for path, content in render(metrics, validation).items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Report out of date: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if args.update_shared_table:
        if not args.by or args.check:
            parser.error("--update-shared-table requires --by and cannot be used with --check")
        update_shared(metrics, args.by)
    print("English audio reports and result summaries are consistent.")


if __name__ == "__main__":
    main()
