"""CH-SIMS label analysis: does a text-only LLM fail where tone and face change the meaning?

CH-SIMS has an overall label (M) plus labels annotated from text only (T), audio only (A)
and vision only (V). We compare the LLM's predictions (from llm_baseline.py) against each.

Usage (from the repo root):
    uv run python experiments/analysis/sims_label_analysis.py \
        ~/Downloads/unaligned_39.pkl ~/results/llm/sims_*_predictions.csv

Writes results/sims_label_analysis.csv (numbers only, safe to commit) and
~/results/llm/sims_conflict_examples.csv (contains transcripts: keep out of the repo).
"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"M": "regression_labels", "T": "regression_labels_T",
          "A": "regression_labels_A", "V": "regression_labels_V"}


def positive(x):
    return x > 0  # CH-SIMS rule in MMSA: 0 counts as negative


def acc2(pred, label):
    return 100 * np.mean(positive(pred) == positive(label))


def corr(pred, label):
    return 100 * np.corrcoef(pred, label)[0, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feature_file", help="CH-SIMS unaligned_39.pkl")
    p.add_argument("predictions", nargs="+", help="*_predictions.csv from llm_baseline.py")
    p.add_argument("--examples-out", default="~/results/llm/sims_conflict_examples.csv")
    args = p.parse_args()

    with open(Path(args.feature_file).expanduser(), "rb") as f:
        test = pickle.load(f)["test"]
    ids = [str(i) for i in test["id"]]
    labels = {k: np.asarray(test[v], dtype=float).reshape(-1) for k, v in LABELS.items()}
    index = {clip: i for i, clip in enumerate(ids)}

    conflict_all = positive(labels["T"]) != positive(labels["M"])
    print(f"CH-SIMS test: {len(ids)} clips, {conflict_all.sum()} where the text label "
          f"and the overall label disagree in sign ({100 * conflict_all.mean():.1f}%)\n")

    rows, examples = [], []
    for pred_file in args.predictions:
        pred_file = Path(pred_file).expanduser()
        model = pred_file.name.replace("sims_", "").replace("_predictions.csv", "")
        with open(pred_file, newline="", encoding="utf-8") as f:
            preds = list(csv.DictReader(f))
        missing = [r["id"] for r in preds if r["id"] not in index]
        if missing:
            raise SystemExit(f"{pred_file.name}: {len(missing)} ids not in the test set, e.g. {missing[0]}")

        sel = np.array([index[r["id"]] for r in preds])
        pred = np.array([float(r["pred"]) for r in preds])
        lab = {k: v[sel] for k, v in labels.items()}
        conflict = positive(lab["T"]) != positive(lab["M"])

        row = {"model": model, "clips": len(sel)}
        for k in LABELS:
            row[f"acc2_vs_{k}"] = round(acc2(pred, lab[k]), 2)
            row[f"corr_vs_{k}"] = round(corr(pred, lab[k]), 2)
        row["agree_clips"] = int((~conflict).sum())
        row["agree_acc2_vs_M"] = round(acc2(pred[~conflict], lab["M"][~conflict]), 2)
        row["conflict_clips"] = int(conflict.sum())
        row["conflict_acc2_vs_M"] = round(acc2(pred[conflict], lab["M"][conflict]), 2)
        row["conflict_acc2_vs_T"] = round(acc2(pred[conflict], lab["T"][conflict]), 2)
        rows.append(row)

        print(f"== {model} ({len(sel)} clips)")
        print("  vs label        Acc-2    Corr")
        names = {"M": "overall", "T": "text only", "A": "audio only", "V": "vision only"}
        for k in LABELS:
            print(f"  {names[k]:12s}  {row[f'acc2_vs_{k}']:6.2f}  {row[f'corr_vs_{k}']:6.2f}")
        print(f"  Acc-2 vs overall label, text and overall agree:    "
              f"{row['agree_acc2_vs_M']:6.2f}  ({row['agree_clips']} clips)")
        print(f"  Acc-2 vs overall label, text and overall conflict: "
              f"{row['conflict_acc2_vs_M']:6.2f}  ({row['conflict_clips']} clips)")
        print(f"  ...the same conflicting clips, vs text label:      "
              f"{row['conflict_acc2_vs_T']:6.2f}\n")

        for r, m, t, a, v, c in zip(preds, lab["M"], lab["T"], lab["A"], lab["V"], conflict):
            if c and positive(float(r["pred"])) == positive(t):
                examples.append({"model": model, "id": r["id"], "pred": r["pred"], "overall": m,
                                 "text_label": t, "audio_label": a, "vision_label": v,
                                 "text": r["text"]})

    out = ROOT / "results" / "sims_label_analysis.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {out}")

    ex_out = Path(args.examples_out).expanduser()
    ex_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ex_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(examples[0]) if examples else ["model"])
        writer.writeheader()
        writer.writerows(examples)
    print(f"Examples where the LLM follows the words but the clip means the opposite: "
          f"{ex_out} ({len(examples)} rows, contains transcripts, keep out of the repo)")


if __name__ == "__main__":
    main()
