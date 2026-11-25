# Vision-only results: label and metric contract

[`summary.json`](summary.json) records the local experiments from 2026-11-18.
It contains no visual feature arrays, raw transcripts, weights or per-example predictions.
The [English report](../../docs/vision/results.md) presents the same results.

## Training labels and dataset separation

| Dataset key | Dataset | Language | Input features | Target field | Ground-truth score scale |
| --- | --- | --- | --- | --- | --- |
| `mosi` | CMU-MOSI | English | Visual features, dimension 20 | `regression_labels` | -3 to +3 |
| `sims` | Original CH-SIMS | Chinese | Visual features, dimension 709 | `regression_labels` | -1 to +1 |

Both targets mean **overall clip sentiment**. `regression_labels_V` (visual-specific
sentiment) and classification-label fields are not used for training. The model
predicts a continuous scalar, not a class index. A score range describes the
ground-truth scale; it is not a constraint forcing every raw prediction inside it.

The models, normalizers, train/validation/test splits and checkpoints are separate
for each dataset. Three seeds (42,43,44) were independently trained for each, making
six neural checkpoints. No combined Chinese/English model or cross-language weight
transfer was used. `artifact_checks` records input shapes, training-only normalizer
counts and hashes of the six distinct checkpoint files.

## Evaluation names

- `metrics.*.common.mae` and `corr`: raw, unclipped predictions versus overall labels.
- `acc2_has0` / `f1_has0`: all examples; both labels and predictions use `>= 0` as
  positive, so exact-zero labels are included in the positive class.
- `acc2_non0` / `f1_non0`: exclude examples whose true label is exactly zero; use
  `> 0` as positive. The main report uses these columns.
- Every F1 value is **support-weighted F1**, not macro-F1 or positive-class F1.
- `metrics.*.dataset_specific`: pinned MMSA-compatible evaluation. MOSI binary
  `Has0_acc_2` includes zero as positive. CH-SIMS binary `Mult_acc_2` includes zero
  as **negative**, and clips scores to [-1,1]. Its three/five-class thresholds
  differ from the shared generic bins. These are evaluation transformations,
  not a change in the regression training task or labels.

## Units and shared-table rows

JSON accuracy/F1/correlation values are **fractions**; MAE is in the dataset's raw
sentiment score units. Report accuracy/F1 columns use percentages. Neural means
and standard deviations use three runs and sample std (`ddof=1`). Ridge and the
training-median constant are deterministic references, not three-seed neural runs.

The two `source=custom_vision`, `model=vision_mlp_deep`, `variant=V` rows in the shared
[`results.csv`](../results.csv) use dataset-specific evaluation and the table's
existing **x100 convention for all metrics**, including MAE and correlation.
They use MOSI `Has0_acc_2`/`Has0_F1_score` and CH-SIMS `Mult_acc_2`/`F1_score`.
They must not be read as the report's `non0` scores or as raw MAE values.
`variant=V` identifies the retained **input modality**, not a switch to visual-specific target labels.

The existing dataset splits were retained. See the report for the MOSI all-zero
visual-feature collision, the CH-SIMS repeated clip features, and sensitivity
checks. A matching feature array is not automatically evidence of a repeated clip.
