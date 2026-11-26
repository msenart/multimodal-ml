"""Uni-modal aids (multimodal_ml/unimodal.py) on CMU-MOSI: do they help where OGM-GE did not?

Same model, training, seeds and probes as scripts/ogm_ge_study.py (Adam, 10e-3). Each run is
saved to results/unimodal/runs/<name>.json and skipped if it exists.

    .venv/bin/python scripts/unimodal_study.py --workers 9
    .venv/bin/python scripts/unimodal_study.py --analyse

Conditions (10 paired seeds each): baseline; auxiliary losses (weight in AUX_WEIGHTS);
text dropout (probability in TEXT_DROPOUTS); uni-modal pretraining (finetune, freeze).
For each method, the setting is selected on the mean validation MAE (early-stopping
criterion); the test differences with the baseline are tested with a Wilcoxon signed-rank
test, Holm-corrected over the 8 conditions, on Acc-2 and F1 (as the rest of the report).
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ogm_ge_analysis import fmt, fmt_diff, holm, mean_sd, paired  # noqa: E402
from ogm_ge_study import ADAM_LR, MODALITIES, SEEDS, SETUP, _build  # noqa: E402

from multimodal_ml import ogm_ge  # noqa: E402
from multimodal_ml.probe import probe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/unimodal"
DEVICE = torch.device("cpu")
AUX_WEIGHTS = (0.3, 1.0, 3.0)
TEXT_DROPOUTS = (0.1, 0.3, 0.5)
PRETRAIN = ("finetune", "freeze")
METRICS = ("acc2_has0", "f1_has0", "corr", "mae")
TESTED = ("acc2_has0", "f1_has0")


def conditions() -> list[dict]:
    out = [{"method": "baseline"}]
    out += [{"method": "aux", "aux_weight": w} for w in AUX_WEIGHTS]
    out += [{"method": "text-dropout", "text_dropout": p} for p in TEXT_DROPOUTS]
    out += [{"method": "pretrain", "unimodal_pretrain": m} for m in PRETRAIN]
    return out


def label(cond: dict) -> str:
    value = cond.get("aux_weight", cond.get("text_dropout", cond.get("unimodal_pretrain")))
    return cond["method"] if value is None else f"{cond['method']}@{value:g}" if isinstance(value, float) \
        else f"{cond['method']}@{value}"


def run(spec: dict) -> str:
    path = OUT / "runs" / f"{label(spec)}_s{spec['seed']}.json"
    if path.exists():
        return f"skip {path.name}"
    config, model, loaders = _build({"data": "mosi", "seed": spec["seed"]}, MODALITIES)
    tr = SETUP["mosi"]["training"]
    args = argparse.Namespace(dataset="mosi", model="concat_fusion", epochs=tr["epochs"], patience=tr["patience"],
                              lr=ADAM_LR, optimizer="adam", ogm_mode="none",
                              aux_weight=spec.get("aux_weight", 0.0), text_dropout=spec.get("text_dropout", 0.0),
                              unimodal_pretrain=spec.get("unimodal_pretrain", "none"))
    test, history = ogm_ge.train(model, loaders, DEVICE, args, verbose=False)
    result = {"spec": spec, "test": test, "history": history,
              "best_val_mae": min(h["val_mae"] for h in history),
              "probe": {m: probe(model.encoders[m], loaders, spec["seed"], DEVICE, config) for m in model.encoders}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result) + "\n")
    return f"done {path.name}"


def analyse() -> None:
    runs = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    by = {}
    for r in runs:
        by.setdefault(label(r["spec"]), {})[r["spec"]["seed"]] = r
    base = by["baseline"]
    summary = {"baseline": {m: mean_sd([r["test"][m] for r in base.values()]) for m in METRICS}, "conditions": {}}
    for cond in conditions()[1:]:
        key, rs = label(cond), by.get(label(cond), {})
        if len(rs) < len(SEEDS):
            continue
        summary["conditions"][key] = {
            "val_mae": mean_sd([r["best_val_mae"] for r in rs.values()]),
            "text_share": mean_sd([r["test"]["text_share"] for r in rs.values()]),
            **{m: mean_sd([r["test"][m] for r in rs.values()]) for m in METRICS},
            "diff": {m: paired({s: r["test"][m] for s, r in base.items()}, {s: r["test"][m] for s, r in rs.items()})
                     for m in METRICS},
            "diff_probe": {mod: {m: paired({s: r["probe"][mod][m] for s, r in base.items()},
                                           {s: r["probe"][mod][m] for s, r in rs.items()}) for m in TESTED}
                           for mod in MODALITIES}}
    keys = list(summary["conditions"])
    for m in TESTED:
        for key, p in zip(keys, holm([summary["conditions"][k]["diff"][m]["p"] for k in keys])):
            summary["conditions"][key]["diff"][m]["p_holm"] = p
    for method in ("aux", "text-dropout", "pretrain"):
        cands = [k for k in keys if k.startswith(method + "@")]
        if cands:
            summary[f"selected_{method}"] = min(cands, key=lambda k: summary["conditions"][k]["val_mae"]["mean"])
    summary["baseline"]["text_share"] = mean_sd([r["test"]["text_share"] for r in base.values()])
    summary["baseline"]["probe"] = {mod: {m: mean_sd([r["probe"][mod][m] for r in base.values()]) for m in TESTED}
                                    for mod in MODALITIES}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    b = summary["baseline"]
    print(f"baseline  Acc2 {fmt(b['acc2_has0'])} | F1 {fmt(b['f1_has0'])} | Corr {fmt(b['corr'])} | "
          f"Tshare {fmt(b['text_share'])}")
    print("selected:", {k: v for k, v in summary.items() if k.startswith("selected_")})
    for key, c in summary["conditions"].items():
        d = c["diff"]
        print(f"{key:18s} valMAE {fmt(c['val_mae'])} | " + " | ".join(
            f"d{m} {fmt_diff(d[m])} pHolm {d[m]['p_holm']:.3f}" for m in TESTED)
            + f" | dCorr {fmt_diff(d['corr'])} | Tshare {fmt(c['text_share'])} | probe F1 "
            + " ".join(f"{mod[0]} {fmt_diff(c['diff_probe'][mod]['f1_has0'])}" for mod in MODALITIES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--analyse", action="store_true", help="Only analyse the saved runs")
    args = parser.parse_args()
    if not args.analyse:
        specs = [{**c, "seed": s} for c in conditions() for s in SEEDS]
        with Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
            for i, msg in enumerate(pool.imap_unordered(run, specs), 1):
                print(f"[{i}/{len(specs)}] {msg}", flush=True)
    analyse()


if __name__ == "__main__":
    main()
