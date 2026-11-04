# Dataset setup (manual download required)

Both datasets are pre-extracted feature pickles distributed by the MMSA toolkit
authors (THUIAR) via Google Drive / BaiduYun — see
https://github.com/thuiar/MMSA#datasets for the links. There's no API for this;
you have to download them by hand.

## CMU-MOSI

Download `unaligned_50.pkl` (MOSI row) and place it at:

    data/MOSI/Processed/unaligned_50.pkl

## CH-SIMS

Download `unaligned_39.pkl` (SIMS row) and place it at:

    data/SIMS/Processed/unaligned_39.pkl

## Expected layout

    data/
    ├── MOSI/Processed/unaligned_50.pkl
    └── SIMS/Processed/unaligned_39.pkl

## Verifying your download

After setting up your environment (see the main [README](../README.md) for the
uv or plain-pip setup), run:

    python -m multimodal_ml --dataset mosi --model baseline --epochs 1
    python -m multimodal_ml --dataset sims --model baseline --epochs 1

(prefix with `uv run` if you're using uv). If the file is missing you'll get a
clear error pointing back to this doc instead of a raw traceback.

Note: everything under `data/` except this README is gitignored — never commit the
`.pkl` files (each is tens to hundreds of MB and isn't ours to redistribute).
