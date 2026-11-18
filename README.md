# multimodal-ml

A git repo to work on both the paper presentation and the project.

## Datasets

- Chinese: CH-SIMS — https://thuiar.github.io/sims.github.io/chsims
- English: CMU-MOSI — http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/

Both are consumed as pre-extracted MMSA-format feature pickles (not raw video), see
[`data/README.md`](data/README.md) for download instructions.

For the audio-only GRU and mean-pooling MLP, start with the
[audio experiment guide](docs/audio/README.md), then read the
[three-seed results](docs/audio/results.md) and [validation report](docs/audio/validation.md).
Small, machine-readable results are in [results/audio_baselines](results/audio_baselines/README.md).

For standalone vision-only regression, see the [vision guide](docs/vision/README.md),
[results and validation](docs/vision/results.md), and [explicit label/metric contract](results/vision_baselines/README.md).
MOSI (English) and CH-SIMS (Chinese) are trained independently; both predict overall sentiment.

## Quick start

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) gives everyone the exact same dependency versions
(via `uv.lock`). Install it with `pip install uv` (works on Windows/macOS/Linux, no
separate download needed), then:

```
uv sync
# download the datasets per data/README.md, then:
uv run python -m multimodal_ml --dataset mosi --model baseline --epochs 1
uv run python -m multimodal_ml --dataset sims --model baseline --epochs 1
```

### Without uv (plain pip)

No uv required at all — pip can install this project directly from `pyproject.toml`:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
# download the datasets per data/README.md, then:
python -m multimodal_ml --dataset mosi --model baseline --epochs 1
python -m multimodal_ml --dataset sims --model baseline --epochs 1
```

Note: this resolves dependency versions with pip's own resolver against the ranges
in `pyproject.toml`, not the exact pins in `uv.lock` — fine for this project, but it
means your installed versions may differ slightly from a `uv sync` teammate's.

To add your own model: write a `MultimodalModel` subclass under
`src/multimodal_ml/models/` (see `models/base.py` for the interface and
`models/baseline.py` for an example), register it in `models/__init__.py`'s
`MODEL_REGISTRY`, then run it the same way with `--model your_model`. The
CLI lives in `src/multimodal_ml/__main__.py`; the shared training loop lives
in `training.py`. Override your model's `training_routine` for model-specific behavior.
