# Standalone vision-only training and validation

Training study, 2026-11-18. MOSI and CH-SIMS were trained **separately**, with separate parameters,
normalizers and checkpoints. Each dataset has three independent neural runs (seeds 42/43/44).

## Models and target

The input is pre-extracted visual features plus declared visual lengths. The target is the overall
`regression_labels`, not the modality-specific visual annotation. Text/audio are not loaded into
the training batches: their placeholders are empty tensors.

| Dataset | Language | Visual feature dimension | Training label field | Ground-truth score range |
| --- | --- | ---: | --- | --- |
| CMU-MOSI | English | 20 | `regression_labels` (overall sentiment) | -3 to +3 |
| Original CH-SIMS | Chinese | 709 | `regression_labels` (overall sentiment) | -1 to +1 |

This is scalar regression, not training a binary/three/five-class classifier. Model outputs are
unrestricted scalars; the ground-truth range is not a hard output bound. Neither dataset uses
the CH-SIMS `regression_labels_V` field for training, and no weights or normalization statistics
are transferred or shared between the English and Chinese models.

VisionMLP: masked mean over valid frames -> training-only feature z-score -> dropout(0.3) ->
three 64-unit ReLU layers -> scalar regression head. Constant training features use unit scale.
The fitted normalization is stored with the weights. This is a defined feature-level baseline,
not a raw-image CNN or a claim to reproduce a published architecture.

## Test results

MLP values are mean +/- sample standard deviation (ddof=1) over seeds 42/43/44.
Binary accuracy/F1 below exclude exact-zero labels and threshold predictions at zero (`non0`).
MAE uses the raw dataset sentiment scale; different datasets' MAEs are not directly comparable.

The shared `results/results.csv` uses separately named dataset-specific evaluation: MOSI `Has0_acc_2`
and CH-SIMS `Mult_acc_2`. All four shared-table metrics are multiplied by 100, including MAE and
correlation. Those binary scores must not be confused with the `non0` column below.

| Dataset | Method | MAE | Correlation | Acc-2 non0 | Weighted F1 non0 |
| --- | --- | ---: | ---: | ---: | ---: |
| MOSI | Training-median constant | 1.4559 | Undefined | 42.23% | 25.07% |
| MOSI | Ridge regression | 1.4624 | 0.1699 | 49.85% | 44.48% |
| MOSI | VisionMLP | 1.4402 +/- 0.0309 | 0.1398 +/- 0.0355 | 49.2886 +/- 2.5248% | 44.2452 +/- 5.8857% |
| SIMS | Training-median constant | 0.5886 | Undefined | 63.92% | 49.85% |
| SIMS | Ridge regression | 0.5249 | 0.4491 | 72.94% | 72.39% |
| SIMS | VisionMLP | 0.5336 +/- 0.0048 | 0.4234 +/- 0.0093 | 72.3368 +/- 2.1920% | 71.5262 +/- 2.1070% |

## Interpretation

CH-SIMS shows useful visual prediction signal relative to a training-median constant. Ridge is
slightly better than this MLP on test MAE, so the neural model is not claimed to be the strongest
vision baseline. MOSI remains weak and seed-sensitive: its mean binary accuracy is near 50%.
These findings do not establish an upper bound on vision-only performance.

The Ridge regularization grid [0.1,1,10,100] was selected by validation MAE only. The six neural runs
used one fixed architecture, Adam/MSE, lr=0.001, batch size 32, at most 60 epochs and validation-MAE
patience 10. Test data did not choose epochs, seeds, normalization or Ridge regularization.

## Per-seed selection

| Dataset | Seed | Epochs run | Best epoch | Validation MAE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOSI | 42 | 20 | 10 | 1.3582 | 1.4369 |
| MOSI | 43 | 26 | 16 | 1.3205 | 1.4111 |
| MOSI | 44 | 17 | 7 | 1.3650 | 1.4725 |
| SIMS | 42 | 27 | 17 | 0.5159 | 0.5284 |
| SIMS | 43 | 20 | 10 | 0.5194 | 0.5345 |
| SIMS | 44 | 24 | 14 | 0.5223 | 0.5379 |

## Verification

- 23 local code checks passed, including visual isolation, masked padding, train-only statistics,
  normalization checkpoint restoration, and a separate future-fusion gradient interface check.
- Both real-data one-epoch smoke runs passed. All six final checkpoints were restored and evaluated.
- Raw-frame inference and cached masked means agreed on the checked validation/test examples.
  Pooling is cached for this MLP only; no recurrent/time-dependent model is trained on that cache.
- The test MAE was independently recalculated. Dataset-specific metrics are recorded separately
  from the common metric definitions.

| Dataset | 16-sample initial MSE | Final MSE | Updates | Validation MAE | Shuffled mean MAE | Worse permutations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOSI | 2.84186 | 0.02132 | 34 | 1.3582 | 1.4253 | 20/20 |
| SIMS | 0.57578 | 0.00536 | 16 | 0.5159 | 0.6643 | 20/20 |

The small-subset diagnostic used dropout=0, lr=0.003, and a preset 99% MSE reduction criterion;
both datasets passed. It checks trainability, not generalization. Permutation tests used seed-42
validation predictions, so they are diagnostics rather than independent significance tests.

## Data and input caveats

Declared visual lengths were preserved. MOSI has nonzero features after declared lengths in
180/16/104 train/valid/test samples; their origin is not established. No input-length policy was
changed to improve results. Dataset files were SHA-256 checked and left unchanged.

MOSI has one train/test **all-zero visual feature collision** with different text/audio features
and labels. It is not evidence that the same full clip occurs in both splits. CH-SIMS has one
cross-split sample with identical visual, text and audio features and the same label. Its test
exclusion sensitivity is recorded; the official split was retained for the main table.

Raw-zero visual inputs may be far outside the training distribution after standardization.
Do not interpret the zero-input stress test alone as causal evidence. A training-mean visual
intervention, corresponding to zero normalized input, is also recorded.

## Checkpoints and future use

All logs, normalizers, per-example predictions and six final model files are under `checkpoints/vision_baseline/20261118_standalone/`.
Use the dataset-specific checkpoint, instantiate `VisionMLP(get_dataset_config(dataset))`,
and load `checkpoint['model_state']`. Call `encode(batch)` for a 64-dimensional representation.
Keep the whole wrapper, including normalization buffers; copying only `.encoder` drops normalization.

No fusion or OGM training has been performed in this standalone study. Only code, English documentation
and compact result metadata are versioned; data, weights and per-example predictions remain local.
