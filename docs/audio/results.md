# Audio-only baseline results

These are exploratory custom audio-only regressors trained on **overall sentiment labels**,
not CH-SIMS audio-specific labels. The experiments ran on 2026-11-11 using CPU PyTorch 2.14.1
and Python 3.13.7. They are not claimed to reproduce a published MMSA architecture.

## Three-seed comparison

Mean +/- sample standard deviation over seeds **42, 43, 44** (ddof=1). Lower MAE is better.
The common binary metrics below exclude exactly-zero ground-truth labels (`non0`).

| Dataset | Model | MAE | Correlation | Acc-2 non0 | Weighted F1 non0 |
| --- | --- | ---: | ---: | ---: | ---: |
| MOSI | Training-median constant | 1.4559 | Undefined | 42.23% | 25.07% |
| MOSI | audio_mlp | 1.4187 +/- 0.0008 | 0.2695 +/- 0.0072 | 51.8293 +/- 0.1524% | 47.9452 +/- 0.3412% |
| MOSI | audio_gru | 1.4440 +/- 0.0191 | 0.2528 +/- 0.0299 | 48.6280 +/- 2.3812% | 42.8897 +/- 4.6950% |
| SIMS | Training-median constant | 0.5886 | Undefined | 63.92% | 49.85% |
| SIMS | audio_mlp | 0.5799 +/- 0.0053 | 0.1134 +/- 0.0377 | 64.0893 +/- 0.1488% | 53.5230 +/- 0.6416% |
| SIMS | audio_gru | 0.5704 +/- 0.0125 | 0.1929 +/- 0.0456 | 63.9175 +/- 1.1234% | 57.4659 +/- 2.1650% |

## Interpretation

The saved results can be independently recalculated and the models respond to correctly paired audio.
However, AudioGRU is worse than the simpler MLP in all three MOSI runs. On CH-SIMS its average
MAE is slightly better, with variation across seeds. This supports a **weak custom baseline**,
not a claim that audio has little value or that this architecture is competitive.

Three seeds are insufficient for a strong significance claim. The teammates' MMSA experiments use
different seeds (1111-1113), implementations and preprocessing; cross-source rows are descriptive comparisons.

## Training settings

Adam, MSE loss, learning rate 0.001, batch size 32, hidden dimension 64, dropout 0.3,
at most 60 epochs, and patience 10 on **validation MAE**. Each test score uses that run's best
validation checkpoint. Test scores did not select seeds, thresholds or normalization settings.

The reported runs use the custom `observed` audio-length policy. The repository default remains
`declared`; pass `--audio-length-policy observed` to reproduce this contribution.

## Metric definitions and shared table units

Common `has0` assigns zero to the positive class (`>= 0`); `non0` excludes zero labels and uses `> 0`.
MMSA MOSI uses both rules; MMSA CH-SIMS binary evaluation includes zero in the **negative** class.
MMSA also uses dataset-specific multi-class bins. Keep common and official metrics separately named.

`results/audio_baselines/metrics.json` stores raw score units and fractions. The shared
`results/results.csv` follows the existing MMSA convention of multiplying **all four metrics by 100**,
including MAE and correlation. The four `custom_audio` rows use MMSA-compatible metrics and ddof=1.

For example, shared-table MOSI MAE around 144 means raw MAE around 1.44, not an error of 144 sentiment points.

## Per-seed MAE

| Dataset | Model | 42 | 43 | 44 |
| --- | --- | ---: | ---: | ---: |
| MOSI | audio_mlp | 1.4193 | 1.4178 | 1.4192 |
| MOSI | audio_gru | 1.4225 | 1.4590 | 1.4506 |
| SIMS | audio_mlp | 0.5738 | 0.5835 | 0.5826 |
| SIMS | audio_gru | 0.5685 | 0.5588 | 0.5837 |

See [validation](validation.md), [design and integration](design.md), and the
[machine-readable evidence](../../results/audio_baselines/README.md).
