# multimodal-ml

A git repo to work on both the paper presentation and the project.

## Datasets

- Chinese: CH-SIMS — https://thuiar.github.io/sims.github.io/chsims
- English: CMU-MOSI — http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/

Both are consumed as pre-extracted MMSA-format feature pickles (not raw video), see
[`data/README.md`](data/README.md) for download instructions.

## Quick start

```
uv sync
# download the datasets per data/README.md, then:
uv run python -m multimodal_ml --dataset mosi --model baseline --epochs 1
uv run python -m multimodal_ml --dataset sims --model baseline --epochs 1
```

To add your own model: write a `MultimodalModel` subclass under
`src/multimodal_ml/models/` (see `models/base.py` for the interface and
`models/baseline.py` for an example), register it in `models/__init__.py`'s
`MODEL_REGISTRY`, then run it the same way with `--model your_model`. The
training routine itself lives in `src/multimodal_ml/__main__.py` — edit it
directly if your model needs something the loop doesn't already do.
