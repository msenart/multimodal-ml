# Audio model design and merge notes

## Architectures

Both models subclass `MultimodalModel`, accept the existing `Batch`, and return
one unrestricted sentiment score per sample, shape `(B,)`. They read only
`audio` and `audio_lengths`; text, vision, IDs and labels do not enter `forward`.

| CLI model | Architecture | Purpose |
| --- | --- | --- |
| `audio_mlp` | Length-masked temporal mean -> Linear(D,64) -> ReLU -> Dropout(0.3) -> Linear(64,1) | Simple trainable audio reference |
| `audio_gru` | Packed one-layer bidirectional GRU, 64 hidden units/direction -> concatenate final states -> Dropout -> Linear(128,64) -> ReLU -> Dropout -> Linear(64,1) | Test whether sequence modeling helps |

`D=5` for the reference MOSI features and `D=33` for CH-SIMS. These feature
dimensions come from the dataset configuration. Empty audio gets a zero
representation. GRU packing ignores trailing padding and preserves batch order.

Both models use the existing MSE/Adam training loop, validation-MAE early stopping,
checkpoint saving and final testing. The global default model remains `baseline`;
that existing model uses all three modalities and serves as a pipeline check.

## Shared data changes

- `(N,1)` scalar targets become `(N,)`, preventing unintended MSE broadcasting.
- Invalid audio shapes, sample counts, lengths or non-finite labels fail clearly.
- Non-finite audio values become zero with a warning.
- **Default:** `audio_length_policy="declared"` preserves supplied lengths.
- **Opt-in:** `--audio-length-policy observed` uses the maximum of the supplied
  length and the last nonzero-frame endpoint, preserving interior zero frames.

The observed policy was used in the reported custom experiments, but is not
MMSA's loader policy. Extra nonzero frames do not establish a confirmed data bug.
No pickle is modified. Keeping this opt-in avoids silently changing teammates'
existing model inputs when the PR is merged.

## Metrics

`metrics.py` keeps the team's existing common metric definitions. Its Acc-5/7
use generic bins and should not be labeled as official MMSA results.
`reference_metrics.py` separately implements the pinned MMSA regression rules:

- MOSI: raw MAE/correlation, signed binary metrics, integer-rounded clipped Acc-5/7.
- CH-SIMS: scores clipped to [-1,1], zero in the negative binary class,
  and dataset-specific three/five-class thresholds.

The exported summaries include both sets of metrics. The shared comparison CSV
uses the MMSA-compatible ones and its existing x100 unit convention.

## Validation and provenance

The study checks checkpoint outputs with an independent loader and metric
implementation; fits 16 real training samples; permutes validation audio; evaluates
zero audio; compares three seeds against AudioMLP; and fits normalization only on
training frames. The original GRU architecture and weights were not changed to
obtain a preferable audit result. Detailed findings are in [validation.md](validation.md).

The original workstation artifacts are preserved under ignored `checkpoints/`.
The committed JSON is a compact review record, not a substitute for archived
weights/predictions. A fresh checkout can rerun the study without those archives.

## Review boundaries

This contribution does not alter the team's MMSA scripts, dependency lockfile,
existing result rows, raw datasets, or shared training-loop behavior. Review the
opt-in length flag, target-shape validation and metric naming especially carefully
before using these results in a combined comparison.
