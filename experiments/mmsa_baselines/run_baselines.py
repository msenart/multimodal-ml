"""Run MMSA baselines: models x seeds x modality variants, then summarize.

Each variant gets its own results folder, so MMSA's CSVs don't mix:
    <res-dir>/<dataset>/<variant>/normal/<dataset>.csv
A combined table is written to <res-dir>/<dataset>/summary.csv.

Usage:
    python run_baselines.py --dataset mosi
    python run_baselines.py --dataset mosi --variants T,A,V
    python run_baselines.py --dataset sims --models lf_dnn --seeds 1111
    python run_baselines.py --dataset mosi --summarize-only
"""
import argparse
import inspect
import time
import traceback
from pathlib import Path

import pandas as pd
from MMSA import MMSA_run, get_config_regression

FEATURE_FILE = {"mosi": "unaligned_50.pkl", "sims": "unaligned_39.pkl"}
RUN_ARGS = inspect.signature(MMSA_run).parameters


def feature_path(data_dir: Path, dataset: str, variant: str) -> Path:
    base = Path(FEATURE_FILE[dataset])
    name = base.name if variant == "TAV" else f"{base.stem}_{variant}{base.suffix}"
    return data_dir / name


def summarize(res_root: Path, dataset: str) -> None:
    frames = []
    for csv in sorted(res_root.glob(f"*/normal/{dataset}.csv")):
        df = pd.read_csv(csv)
        df.insert(0, "Variant", csv.parent.parent.name)
        frames.append(df)
    if not frames:
        print("No results found to summarize.")
        return
    table = pd.concat(frames, ignore_index=True)
    out = res_root / "summary.csv"
    table.to_csv(out, index=False)
    print(f"\nSummary written to {out}\n")
    print(table.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mosi", choices=sorted(FEATURE_FILE))
    p.add_argument("--data-dir", default="~/data/mmsa")
    p.add_argument("--models", default="lf_dnn,tfn,mult")
    p.add_argument("--seeds", default="1111,1112,1113")
    p.add_argument("--variants", default="TAV", help="e.g. TAV or T,A,V or TA,TV,AV")
    p.add_argument("--res-dir", default="~/results/mmsa")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    res_root = Path(args.res_dir).expanduser() / args.dataset
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]

    if args.summarize_only:
        summarize(res_root, args.dataset)
        return

    if "res_save_dir" not in RUN_ARGS:
        raise RuntimeError("This MMSA version has no res_save_dir argument; "
                           "variants would overwrite each other's results.")

    # Check all inputs before spending GPU time
    for v in variants:
        path = feature_path(data_dir, args.dataset, v)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run make_unimodal.py first?)")

    failures = []
    for variant in variants:
        var_dir = res_root / variant
        for model in models:
            start = time.time()
            print(f"\n=== {args.dataset} | {variant} | {model} | seeds {seeds} ===", flush=True)
            try:
                config = get_config_regression(model, args.dataset)
                config["featurePath"] = str(feature_path(data_dir, args.dataset, variant))
                kwargs = {
                    "config": config,
                    "seeds": seeds,
                    "gpu_ids": [0],
                    "num_workers": args.num_workers,
                    "res_save_dir": str(var_dir),
                    "model_save_dir": str(var_dir / "models"),
                    "log_dir": str(var_dir / "logs"),
                }
                MMSA_run(model, args.dataset,
                         **{k: v for k, v in kwargs.items() if k in RUN_ARGS})
                print(f"--- done in {(time.time() - start) / 60:.1f} min", flush=True)
            except Exception:
                traceback.print_exc()
                failures.append(f"{variant}/{model}")

    summarize(res_root, args.dataset)
    if failures:
        print("\nFAILED runs:", ", ".join(failures))


if __name__ == "__main__":
    main()
