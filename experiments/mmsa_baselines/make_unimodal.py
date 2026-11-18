"""Create modality-ablated copies of an MMSA feature file.

Removed modalities are replaced with zeros (text token ids with [PAD]),
so the same models can be trained on e.g. text-only, audio-only, vision-only.

Variant names use the kept modalities: T, A, V, TA, TV, AV.

Usage:
    python make_unimodal.py ~/data/mmsa/unaligned_50.pkl
    python make_unimodal.py ~/data/mmsa/unaligned_50.pkl --variants T,A,V,TA,TV,AV
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

MODALITY = {"T": "text", "A": "audio", "V": "vision"}


def ablate_split(split: dict, keep: set) -> dict:
    """Return a copy of one data split with the removed modalities zeroed."""
    out = dict(split)  # shallow copy; replaced arrays never touch the original

    if "text" not in keep:
        if "text" in out:
            out["text"] = np.zeros_like(out["text"])
        if "text_bert" in out:
            tb = np.array(out["text_bert"], copy=True)
            if tb.ndim == 3 and tb.shape[1] == 3:
                # layout: [input_ids, attention_mask, segment_ids] -> ids become [PAD]
                tb[:, 0, :] = 0
            else:
                tb = np.zeros_like(tb)
            out["text_bert"] = tb

    for name in ("audio", "vision"):
        if name not in keep and name in out:
            out[name] = np.zeros_like(out[name])

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feature_file", help="original MMSA .pkl file")
    p.add_argument("--variants", default="T,A,V", help="comma list of kept modalities")
    p.add_argument("--out-dir", default=None, help="defaults to the input file's folder")
    args = p.parse_args()

    src = Path(args.feature_file).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    with open(src, "rb") as f:
        data = pickle.load(f)

    split = data["train"]
    print("Keys in train split:", sorted(split.keys()))
    for key in ("text", "text_bert", "audio", "vision"):
        if key in split:
            print(f"  {key:10s} shape {np.asarray(split[key]).shape}")

    for variant in [v.strip().upper() for v in args.variants.split(",") if v.strip()]:
        if not variant or any(c not in MODALITY for c in variant):
            raise ValueError(f"Bad variant '{variant}', use letters T, A, V")
        keep = {MODALITY[c] for c in variant}
        ablated = {name: ablate_split(s, keep) if isinstance(s, dict) else s
                   for name, s in data.items()}
        dst = out_dir / f"{src.stem}_{variant}.pkl"
        with open(dst, "wb") as f:
            pickle.dump(ablated, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {dst} (kept: {', '.join(sorted(keep))})")


if __name__ == "__main__":
    main()
