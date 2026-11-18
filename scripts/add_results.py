"""Append MMSA results to the shared table results/results.csv.

Usage (from the repo root):
    python scripts/add_results.py ~/results/mmsa/mosi/summary.csv --dataset mosi --by otto --seeds 1111,1112,1113

Re-adding the same dataset/model/variant/source replaces the old row.
"""
import argparse
import csv
import datetime
from pathlib import Path

TABLE = Path(__file__).resolve().parents[1] / "results" / "results.csv"
COLUMNS = ["dataset", "model", "variant", "source", "acc2", "acc2_std", "f1", "f1_std",
           "mae", "mae_std", "corr", "corr_std", "seeds", "by", "date", "notes"]
# MMSA uses different metric names for MOSI and CH-SIMS
METRICS = {"acc2": ["Has0_acc_2", "Mult_acc_2"], "f1": ["Has0_F1_score", "F1_score"],
           "mae": ["MAE"], "corr": ["Corr"]}


def mean_std(cell):
    """'(75.36, 0.0)' -> ('75.36', '0.0'); '75.36' -> ('75.36', '')"""
    parts = cell.strip("() ").split(",")
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")


p = argparse.ArgumentParser()
p.add_argument("csv_file")
p.add_argument("--dataset", required=True)
p.add_argument("--by", required=True)
p.add_argument("--seeds", default="")
p.add_argument("--source", default="mmsa")
p.add_argument("--notes", default="")
args = p.parse_args()

rows = list(csv.DictReader(open(TABLE))) if TABLE.exists() else []

for r in csv.DictReader(open(Path(args.csv_file).expanduser())):
    new = {"dataset": args.dataset, "model": r["Model"], "variant": r.get("Variant", "TAV"),
           "source": args.source, "seeds": args.seeds, "by": args.by,
           "date": datetime.date.today().isoformat(), "notes": args.notes}
    for short, names in METRICS.items():
        name = next((n for n in names if n in r), None)
        new[short], new[short + "_std"] = mean_std(r[name]) if name else ("", "")
    key = ("dataset", "model", "variant", "source")
    rows = [old for old in rows if any(old[k] != new[k] for k in key)] + [new]
    print(f"{args.dataset} {new['model']} {new['variant']}: Acc-2 {new['acc2']}, Corr {new['corr']}")

TABLE.parent.mkdir(exist_ok=True)
with open(TABLE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {TABLE}")
