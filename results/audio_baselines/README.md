# Recorded custom audio experiments

These small English-language metadata files describe the real experiments from
2026-11-11. They contain no feature arrays, raw transcripts, weights or per-example
predictions.

- [`metrics.json`](metrics.json): three-seed raw results, sample standard deviations,
  common metrics, MMSA-compatible metrics, constant references and settings.
- [`validation.json`](validation.json): dataset hashes/counts, checkpoint checks,
  small-subset fits, interventions, normalization diagnostics and duplicate sensitivity.
- [Readable results](../../docs/audio/results.md).
- [Validation conclusions](../../docs/audio/validation.md).

JSON uses raw score units and fractional accuracy/F1/correlation. In contrast,
the repository's shared [`results.csv`](../results.csv) uses MMSA-style x100 values
for all metrics, including MAE and correlation. The four rows with
`source=custom_audio`, `variant=A` follow that convention; their seeds are 42/43/44
and their standard deviations use ddof=1.

The existing MMSA rows use a different seed set and pipeline. Do not infer a
controlled architectural improvement from those cross-source comparisons.

To check or regenerate documentation without downloading datasets:

```sh
uv run python scripts/report_audio_validation.py --check
uv run python scripts/report_audio_validation.py
```

The full real-data reproduction workflow is in the [audio guide](../../docs/audio/README.md).
