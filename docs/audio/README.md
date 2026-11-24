# Audio-only baselines: team handoff

This contribution supplies the **audio-only comparison** for the group's study of
text/audio/vision contributions and gradient modulation. It covers CMU-MOSI and
the original CH-SIMS, using their **overall sentiment regression labels**.

## Reading order

1. [Results](results.md): three-seed comparison and metric definitions.
2. [Validation](validation.md): what was checked, what failed to improve, and data caveats.
3. [Design and integration](design.md): model interfaces and effects on the shared loader.
4. [Machine-readable results](../../results/audio_baselines/README.md): inspect or regenerate tables.

The deliverable is a pair of defined, tested custom baselines, not a strong audio
model or a reproduction of a published architecture. It does not implement OGM-GE.
The teammates' MMSA models and HPC setup remain in
[`experiments/mmsa_baselines`](../../experiments/mmsa_baselines/README.md).

## Environment and data

Run commands from the repository root using the existing Python 3.13 project environment:

```sh
uv sync --locked
```

This is separate from the Python 3.11 MMSA/HPC environment. Do not install both
sets of dependencies into the same environment.

Download the official pre-extracted features using [data/README.md](../../data/README.md):

```text
data/MOSI/Processed/unaligned_50.pkl
data/SIMS/Processed/unaligned_39.pkl
```

No raw video processing is required. Data, weights, per-example predictions and
virtual environments stay local and are excluded from Git.

## Workflow 1: inspect results without datasets

```sh
uv run python scripts/report_audio_validation.py --check
```

Omit `--check` to regenerate the two English reports from the committed JSON.
This workflow needs no original model files, downloaded datasets, or network requests.

## Workflow 2: code checks without datasets

```sh
uv run python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs these synthetic checks and the report consistency
check. Synthetic results are not evidence of real-world prediction accuracy.

## Workflow 3: audit data and run one epoch

The published experiments used the opt-in `observed` length policy. The shared
loader's default remains `declared` to preserve upstream behavior.

```sh
uv run python scripts/check_audio_data.py --verify-hash --audio-length-policy observed
uv run python scripts/run_audio.py --stage smoke
```

The launcher checks both datasets before training and keeps separate output
directories. It sets OMP/MKL threads to 1 unless explicitly configured otherwise.
On Windows, `run_audio.cmd --stage smoke` uses the existing `.venv` directly;
double-clicking it runs the complete seed-42 workflow, including full training.

## Workflow 4: individual model training

```sh
uv run python -m multimodal_ml --dataset mosi --model audio_gru --audio-length-policy observed --epochs 60 --patience 10 --batch-size 32 --lr 0.001 --hidden-dim 64 --dropout 0.3 --seed 42 --checkpoint-dir checkpoints/audio_gru/mosi/seed42
uv run python -m multimodal_ml --dataset sims --model audio_mlp --audio-length-policy observed --epochs 60 --patience 10 --batch-size 32 --lr 0.001 --hidden-dim 64 --dropout 0.3 --seed 42 --checkpoint-dir checkpoints/audio_mlp/sims/seed42
```

Change the dataset/model/seed and use a new checkpoint directory for each run.
These examples do not overwrite the recorded historical artifacts.

## Workflow 5: reproduce the full validation study

```sh
uv run python scripts/validate_audio_model.py --output checkpoints/audio_validation/reproduction
uv run python scripts/check_audio_duplicates.py checkpoints/audio_validation/reproduction
uv run python scripts/report_audio_validation.py checkpoints/audio_validation/reproduction
```

This trains three seeds for both architectures/datasets, performs small-subset
fits and validation interventions, and runs a seed-42 normalization diagnostic.
It requires the real features and can take substantially longer than a smoke test.
The reference metrics are downloaded from a fixed MMSA commit. Completed comparison
runs are reused when rerunning the same output directory.

A fresh checkout trains seed 42 normally. On the original workstation, an available
archived `checkpoints/audio_results_seed42.json` can reuse the historical weights;
`--original-results PATH` explicitly selects such a manifest. Archived Windows-relative
paths are normalized for other platforms.

Exporting a new study updates committed summaries/reports, so review the diff.
Updating the shared table is a separate explicit action:

```sh
uv run python scripts/report_audio_validation.py --update-shared-table --by YOUR_GITHUB_USERNAME
```

## How this fits the remaining group work

The group still needs matched comparisons of text-only, vision-only and fusion
models, plus the same fusion architecture with/without gradient modulation.
These audio results provide one reference point. Before combining experiments,
agree on label targets, seed protocol, preprocessing, metric definitions and the
CH-SIMS duplicate policy; do not interpret differences between unmatched setups
as an isolated architecture effect.
