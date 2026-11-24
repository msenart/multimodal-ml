# Standalone vision-only baseline

The English CMU-MOSI and Chinese CH-SIMS models are trained independently. Each
dataset has separate training statistics, weights and test results. Only visual
features and their supplied lengths enter the model. The target is overall
sentiment regression, matching the audio-only task.

Read [the results and validation report](results.md) first, followed by the
[label and metric contract](../../results/vision_baselines/README.md).
Audio-only and vision-only work are reviewed together in PR #10, targeting
`main`. The branch name `audio-baseline` is historical; it contains both
contributions. The vision model is trained independently and does not require
an audio checkpoint or audio predictions.

From the project root, using the locked Python 3.13 project environment:

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python scripts/train_vision_baseline.py
```

On Windows with an existing `.venv`, the equivalent commands are:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/train_vision_baseline.py
```

The runner performs one-epoch smoke tests and three fixed-seed runs per dataset,
plus training-median/Ridge references, a small-subset fit, validation permutations,
checkpoint restoration and raw-frame/cache equivalence checks. It does not access
a remote repository or upload anything. Re-running creates a new artifact directory.

To check or regenerate the English report from committed metadata, without data
files, weights, archived experiment folders or retraining:

```sh
uv run python scripts/report_vision_baseline.py --check
uv run python scripts/report_vision_baseline.py
```

To export a newly completed local experiment instead:

```powershell
.\.venv\Scripts\python.exe scripts/report_vision_baseline.py checkpoints/vision_baseline/20261118_standalone
```

The model is a small trainable MLP over masked-mean visual features, with train-only
z-scoring. It exposes `feature_dim=64` and `encode(batch)` for later fusion. These
interfaces do not imply that a multimodal model or gradient-modulation method has
already been trained.

The Chinese and English checkpoints cannot be substituted for each other: their
input dimensions are 709 and 20 respectively. Keep the dataset-specific fitted
normalization buffers when reusing `encode()`.
