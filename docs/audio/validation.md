# Audio baseline validation

## Verdict

The checks support reproducible weak custom baselines that use some audio information.
They do not prove the absence of every bug, establish strong predictive performance, or constitute
a full reproduction of MMSA. Original weights and data were retained during the audit.

## Implementation and data checks

The historical audit passed 13 synthetic checks, including padding, gradient flow, modality isolation,
packed-sequence sample order, checkpoint restoration and metric boundaries. The current PR adds
merge-safety and fresh-checkout checks. CI runs without downloading datasets.

Saved-checkpoint MAE/correlation/binary metrics were independently recalculated. A separate
NumPy implementation agreed with the shared metrics. Twelve saved test prediction files were checked
against pinned MMSA definitions. Re-evaluating a checkpoint is not a claim to replay every original training step.

| Dataset | Train / valid / test | ID overlap across splits | Exact stored-audio overlap |
| --- | --- | --- | --- |
| MOSI | 1284 / 229 / 686 | {'train-valid': 0, 'train-test': 0, 'valid-test': 0} | {'train-valid': 0, 'train-test': 0, 'valid-test': 0} |
| SIMS | 1368 / 456 / 457 | {'train-valid': 0, 'train-test': 0, 'valid-test': 0} | {'train-valid': 0, 'train-test': 1, 'valid-test': 0} |

These checks do not establish speaker independence or exclude near-duplicates.

## Small-subset fitting

Diagnostic only: 16 training samples, dropout disabled, lr=0.003, up to 600 updates.
The fixed criterion was at least 99% reduction in MSE, evaluated **after** the optimizer update.
Both datasets passed with raw and training-standardized features; this checks trainability, not generalization.

| Dataset | Features | Initial MSE | Final MSE | Updates |
| --- | --- | ---: | ---: | ---: |
| MOSI | raw | 2.845328 | 0.019731 | 44 |
| MOSI | standardized | 2.829745 | 0.024979 | 30 |
| SIMS | raw | 0.641665 | 0.003270 | 17 |
| SIMS | standardized | 0.652450 | 0.006452 | 21 |

## Validation-set interventions

Audio and its length were shuffled together while labels stayed fixed. Every one of 20 permutations
increased MAE on each dataset. This supports use of paired audio, not a strong causal interpretation.
Zero-audio evaluation retained lengths and therefore retained possible duration information.

| Dataset | Original MAE | Mean shuffled MAE | Zero-audio MAE | Constant MAE |
| --- | ---: | ---: | ---: | ---: |
| MOSI | 1.3674 | 1.4304 | 1.4662 | 1.4205 |
| SIMS | 0.5701 | 0.6028 | 0.6064 | 0.5882 |

Stored clip-bootstrap intervals are exploratory: this validation set selected epochs, and the intervals
do not model dependence between clips from the same speaker. They are not independent significance tests.

## Normalization and length-policy diagnostics

Standardization used means/stds from non-padding **training frames only**; padding stayed zero.
These seed-42 experiments were evaluated only on validation data, not on the test set.

| Dataset | Raw-feature validation MAE | Standardized validation MAE |
| --- | ---: | ---: |
| MOSI | 1.3674 | 1.3255 |
| SIMS | 0.5701 | 0.5925 |

Standardization helped MOSI validation but worsened CH-SIMS validation under the fixed settings.
Feature scale alone is therefore not an established explanation for the weak results.

MOSI has 43/5/23 train/valid/test samples with nonzero features after the declared length.
The custom `observed` policy retains them; MMSA uses the declared lengths. The origin of the extra
frames is not established, so this must not be called a confirmed official-data bug fix.
The default shared loader preserves declared lengths; use the explicit flag for the recorded audio experiments.

## CH-SIMS cross-split duplicate

One official test sample has the same audio, text and visual features and overall label as a training
sample, despite a different clip ID. The feature-file SHA-256 matches the reference release.
This dataset-level overlap is disclosed; the main scores retain the official split.

- Train `video_0049$_$0008` and test `video_0049$_$0007`, overall label -0.8.

| Run | Original test MAE | Excluding the duplicate test sample |
| --- | ---: | ---: |
| audio_gru_raw_seed42 | 0.568541 | 0.568498 |
| audio_gru_raw_seed43 | 0.558847 | 0.559130 |
| audio_gru_raw_seed44 | 0.583710 | 0.583989 |
| audio_mlp_raw_seed42 | 0.573798 | 0.573964 |
| audio_mlp_raw_seed43 | 0.583455 | 0.583574 |
| audio_mlp_raw_seed44 | 0.582554 | 0.582477 |

Each change was below 0.0003 MAE and did not change the comparison. The team should use one documented
duplicate policy across models; do not silently change each model's split independently.

## Reference and evidence

MMSA reference commit: [`a94e65d07fa1`](https://github.com/thuiar/MMSA/tree/a94e65d07fa1ae0d44e552390074b29b0898edfd).
The committed JSON includes source URLs/hashes, per-seed scores, intervention summaries and diagnostic settings.
Weights, feature files and per-example predictions stay under ignored local artifact directories.

See the [reproduction guide](README.md) and [results](results.md).
