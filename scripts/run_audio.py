"""Check real data, smoke test each dataset, then optionally train AudioGRU."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "smoke", "train", "all"], default="all")
    parser.add_argument("--dataset", nargs="+", choices=["mosi", "sims"], default=["mosi", "sims"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Small recurrent batches run efficiently without CPU thread oversubscription.
    # Explicit user settings still take precedence.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "checkpoints" / "audio_gru" / f"seed_{args.seed}" / stamp
    try:
        run([
            sys.executable, str(ROOT / "scripts" / "check_audio_data.py"),
            "--dataset", *args.dataset, "--verify-hash",
            "--audio-length-policy", "observed",
            "--output", str(output / "data_audit.json"),
        ])
        if args.stage == "check":
            return 0
        # All requested smoke runs must succeed before any full training starts.
        stages = [("smoke", 1)]
        if args.stage in ("train", "all"):
            stages.append(("train", 60))
        for stage, epochs in stages:
            for dataset in args.dataset:
                run([
                    sys.executable, "-m", "multimodal_ml",
                    "--dataset", dataset, "--model", "audio_gru",
                    "--audio-length-policy", "observed",
                    "--epochs", str(epochs), "--patience", "10", "--batch-size", "32",
                    "--lr", "0.001", "--hidden-dim", "64", "--dropout", "0.3",
                    "--seed", str(args.seed), "--checkpoint-dir", str(output / dataset / stage),
                ])
    except subprocess.CalledProcessError as exc:
        print(f"\nStopped after a failed step (exit {exc.returncode}). See the message above.")
        return exc.returncode
    print(f"\nFinished. Checkpoints and training logs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
