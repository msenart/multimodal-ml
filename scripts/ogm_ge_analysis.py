"""Analyse the runs of scripts/ogm_ge_study.py; writes results/ogm_ge/summary.json and prints the report tables.

    .venv/bin/python scripts/ogm_ge_analysis.py
    .venv/bin/python scripts/ogm_ge_analysis.py --runs results/ogm_ge/runs_ratio_mean \
        --summary results/ogm_ge/summary_ratio_mean.json   # sensitivity: --ogm-ratio mean

Statistics (n = 10 paired seeds): difference OGM - baseline per seed (same seed, so same
initialisation and same data order); mean with a 95% bootstrap CI over seeds (10,000
resamples); two-sided Wilcoxon signed-rank test, Holm-corrected over the 8 OGM conditions
of an optimizer; minimum detectable effect (MDE) of a paired t-test, two-sided 5% level,
80% power: (t_0.975 + t_0.80) * sd(diff) / sqrt(n). Primary metric: test MAE (the
early-stopping and selection criterion). alpha is selected on the mean validation MAE.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ogm_ge"
MODALITIES = ("text", "audio", "vision")
REPORTED = ("mae", "corr", "acc2_has0", "f1_has0")


RUNS = OUT / "runs"


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def select(runs, **spec) -> dict[int, dict]:
    """Runs matching `spec`, keyed by seed."""
    out = {r["spec"]["seed"]: r for r in runs if all(r["spec"].get(k) == v for k, v in spec.items())}
    return dict(sorted(out.items()))


def mean_sd(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def paired(base: dict[int, float], other: dict[int, float]) -> dict:
    seeds = sorted(set(base) & set(other))
    d = np.array([other[s] - base[s] for s in seeds])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, d.size, (10_000, d.size))].mean(axis=1)
    nonzero = d[d != 0]
    p = float(stats.wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
    sd = float(d.std(ddof=1))
    mde = (stats.t.ppf(0.975, d.size - 1) + stats.t.ppf(0.80, d.size - 1)) * sd / np.sqrt(d.size)
    return {"mean": float(d.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": p, "sd": sd, "mde": float(mde), "n": int(d.size)}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def test_metric(runs: dict[int, dict], key: str) -> dict[int, float]:
    return {s: r["test"][key] for s, r in runs.items()}


def probe_metric(runs: dict[int, dict], modality: str) -> dict[int, float]:
    return {s: r["probe"][modality]["corr"] for s, r in runs.items()}


def condition_grid(runs, data: str, optimizer: str, lr: float) -> dict:
    """Baseline and every OGM condition of one optimizer, with paired comparisons."""
    common = {"data": data, "kind": "fusion", "optimizer": optimizer, "lr": lr}
    base = select(runs, **common, mode="none")
    conditions = sorted({(r["spec"]["mode"], r["spec"]["alpha"]) for r in runs
                         if all(r["spec"].get(k) == v for k, v in common.items()) and r["spec"]["mode"] != "none"})

    def describe(rs):
        return {"val_mae": mean_sd([r["best_val_mae"] for r in rs.values()]),
                "text_share": mean_sd([r["test"]["text_share"] for r in rs.values()]),
                "k_text": mean_sd([np.mean([h["k_text"] for h in r["history"]]) for r in rs.values()]),
                **{m: mean_sd(list(test_metric(rs, m).values())) for m in REPORTED},
                "probe": {m: mean_sd(list(probe_metric(rs, m).values())) for m in MODALITIES}}

    grid = {"baseline": describe(base), "conditions": {}}
    for mode, alpha in conditions:
        rs = select(runs, **common, mode=mode, alpha=alpha)
        entry = describe(rs)
        entry["diff"] = {m: paired(test_metric(base, m), test_metric(rs, m)) for m in REPORTED}
        entry["diff_probe"] = {m: paired(probe_metric(base, m), probe_metric(rs, m)) for m in MODALITIES}
        grid["conditions"][f"{mode}@{alpha:g}"] = entry
    keys = list(grid["conditions"])
    for key, p in zip(keys, holm([grid["conditions"][k]["diff"]["mae"]["p"] for k in keys])):
        grid["conditions"][key]["diff"]["mae"]["p_holm"] = p
    for mode in sorted({k.split("@")[0] for k in keys}):
        candidates = [k for k in keys if k.startswith(mode + "@")]
        grid[f"selected_{mode}"] = min(candidates, key=lambda k: grid["conditions"][k]["val_mae"]["mean"])
    return grid


def mechanism(runs, data: str, optimizer: str, lr: float, epochs: int = 5) -> dict:
    """OM-only runs vs the baseline over the first `epochs` epochs (all runs last at least 6):
    mean k of each encoder, and ratio of the distances its parameters travelled (sum of the update norms)."""
    common = {"data": data, "kind": "fusion", "optimizer": optimizer, "lr": lr}
    base = select(runs, **common, mode="none")

    def path(r, m):
        return sum(h[f"update_norm_{m}"] for h in r["history"][:epochs])

    out = {}
    for alpha in sorted({r["spec"]["alpha"] for r in runs if r["spec"].get("mode") == "ogm"
                         and all(r["spec"].get(k) == v for k, v in common.items())}):
        rs = select(runs, **common, mode="ogm", alpha=alpha)
        out[f"{alpha:g}"] = {
            m: {"k": mean_sd([np.mean([h[f"k_{m}"] for h in r["history"][:epochs]]) for r in rs.values()]),
                "path_ratio": mean_sd([path(rs[s], m) / path(base[s], m) for s in rs if s in base])}
            for m in MODALITIES}
    return out


def stacking(runs, data: str = "mosi") -> dict:
    """Linear regression on the validation predictions of the uni-modal models, evaluated on test."""
    uni = {m: select(runs, data=data, kind="unimodal", modality=m, optimizer="adam") for m in MODALITIES}
    seeds = sorted(set.intersection(*(set(u) for u in uni.values())))
    scores = defaultdict(dict)
    for s in seeds:
        pred = {split: np.column_stack([uni[m][s]["predictions"][split]["pred"] for m in MODALITIES])
                for split in ("valid", "test")}
        label = {split: np.asarray(uni["text"][s]["predictions"][split]["label"]) for split in ("valid", "test")}
        for name, cols in (("T", [0]), ("TAV", [0, 1, 2])):
            fit = LinearRegression().fit(pred["valid"][:, cols], label["valid"])
            p = fit.predict(pred["test"][:, cols])
            scores[name][s] = {"mae": float(np.mean(np.abs(p - label["test"]))),
                               "corr": float(np.corrcoef(p, label["test"])[0, 1]),
                               "weights": fit.coef_.tolist()}
    out = {}
    for name in ("T", "TAV"):
        out[name] = {k: mean_sd([scores[name][s][k] for s in seeds]) for k in ("mae", "corr")}
        out[name]["weights"] = np.mean([scores[name][s]["weights"] for s in seeds], axis=0).tolist()
    out["TAV-T"] = {k: paired({s: scores["T"][s][k] for s in seeds}, {s: scores["TAV"][s][k] for s in seeds})
                    for k in ("mae", "corr")}
    out["alone"] = {m: {k: mean_sd(list(test_metric(uni[m], k).values())) for k in ("mae", "corr")}
                    for m in MODALITIES}
    return out


def probes_alone(runs, data: str) -> dict:
    out = {"random": {}, "alone": {}}
    rand = select(runs, data=data, kind="random")
    for m in MODALITIES:
        out["random"][m] = mean_sd(list(probe_metric(rand, m).values()))
        alone = {}
        for r in runs:
            sp = r["spec"]
            if sp["data"] == data and sp["kind"] == "unimodal" and sp["modality"] == m and sp["optimizer"] == "adam":
                alone[sp["seed"]] = r["probe"][m]["corr"]
        if alone:
            out["alone"][m] = mean_sd(list(alone.values()))
    return out


def fmt(x: dict, scale: float = 100, digits: int = 2) -> str:
    return f"{x['mean'] * scale:.{digits}f} ± {x['sd'] * scale:.{digits}f}"


def fmt_diff(d: dict, scale: float = 100) -> str:
    lo, hi = (v * scale for v in d["ci"])
    return f"{d['mean'] * scale:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--summary", type=Path, default=OUT / "summary.json")
    args = parser.parse_args()
    RUNS = args.runs
    runs = [r for r in load() if "error" not in r]
    diverged = [p.name for p in RUNS.glob("*.json") if "error" in json.loads(p.read_text())]
    lr_sgd = json.loads((OUT / "sgd_lr_selection.json").read_text())
    summary = {"n_runs": len(runs), "diverged_runs": diverged, "sgd_lr_selection": lr_sgd, "mosi": {}, "synthetic": {}}
    optim = {"adam": 1e-3, "sgd": lr_sgd["selected"]}
    for opt, lr in optim.items():
        summary["mosi"][opt] = {"grid": condition_grid(runs, "mosi", opt, lr), "mechanism": mechanism(runs, "mosi", opt, lr)}
    summary["mosi"]["stacking"] = stacking(runs)
    summary["mosi"]["probes"] = probes_alone(runs, "mosi")
    for scenario in ("independent", "redundant"):
        summary["synthetic"][scenario] = {"probes": probes_alone(runs, scenario)}
        for opt, lr in {"adam": 1e-3, "sgd": 1e-2}.items():
            uni = select(runs, data=scenario, kind="unimodal", modality="vision", optimizer=opt)
            summary["synthetic"][scenario][opt] = {
                "grid": condition_grid(runs, scenario, opt, lr), "mechanism": mechanism(runs, scenario, opt, lr),
                "vision_alone": {k: mean_sd(list(test_metric(uni, k).values())) for k in ("mae", "corr")},
                "vision_alone_probe": mean_sd([r["probe"]["vision"]["corr"] for r in uni.values()])}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(runs)} runs, diverged: {diverged}")
    print("SGD learning rate (mean best val MAE):", lr_sgd)
    for data, block in [("mosi", summary["mosi"])] + [(f"synthetic-{s}", summary["synthetic"][s])
                                                      for s in ("independent", "redundant")]:
        for opt in ("adam", "sgd"):
            g = block[opt]["grid"]
            print(f"\n=== {data} / {opt} (x100) ===  selected: ogm={g.get('selected_ogm')} ogm-ge={g.get('selected_ogm-ge')}")
            b = g["baseline"]
            print(f"{'baseline':10s} valMAE {fmt(b['val_mae'])} | MAE {fmt(b['mae'])} | Corr {fmt(b['corr'])} | "
                  f"Acc2 {fmt(b['acc2_has0'])} | F1 {fmt(b['f1_has0'])} | Tshare {fmt(b['text_share'])}")
            for key, c in g["conditions"].items():
                d = c["diff"]
                print(f"{key:10s} valMAE {fmt(c['val_mae'])} | k_T {c['k_text']['mean']:.3f} | "
                      f"dMAE {fmt_diff(d['mae'])} pHolm {d['mae']['p_holm']:.3f} MDE {d['mae']['mde'] * 100:.2f} | "
                      f"dCorr {fmt_diff(d['corr'])} | dAcc2 {fmt_diff(d['acc2_has0'])} | "
                      f"dProbeV {fmt_diff(c['diff_probe']['vision'])} | Tshare {fmt(c['text_share'])}")
            print("mechanism (OM only, first 5 epochs): alpha -> mean k / path ratio per modality")
            for alpha, mech in block[opt]["mechanism"].items():
                print(f"  {alpha:>4s}: " + "  ".join(
                    f"{m} k {mech[m]['k']['mean']:.3f} path {mech[m]['path_ratio']['mean']:.3f}±"
                    f"{mech[m]['path_ratio']['sd']:.3f}" for m in MODALITIES))
    pr = summary["mosi"]["probes"]
    print("\n=== MOSI probes (corr x100) ===")
    for m in MODALITIES:
        print(f"{m:7s} random {fmt(pr['random'][m])} | alone {fmt(pr['alone'][m])} | "
              + " | ".join(f"joint {o} base {fmt(summary['mosi'][o]['grid']['baseline']['probe'][m])}" for o in optim))
    st = summary["mosi"]["stacking"]
    print("\n=== MOSI stacking (x100) ===")
    for k in ("T", "TAV"):
        print(f"{k:4s} MAE {fmt(st[k]['mae'])} Corr {fmt(st[k]['corr'])} weights {np.round(st[k]['weights'], 3)}")
    print(f"TAV-T dMAE {fmt_diff(st['TAV-T']['mae'])} dCorr {fmt_diff(st['TAV-T']['corr'])}")
    print("alone:", {m: (fmt(st['alone'][m]['mae']), fmt(st['alone'][m]['corr'])) for m in MODALITIES})
    for s in ("independent", "redundant"):
        sy = summary["synthetic"][s]
        print(f"\n=== synthetic {s}: vision probe random {fmt(sy['probes']['random']['vision'])}")
        for opt in ("adam", "sgd"):
            print(f"  {opt}: vision alone corr {fmt(sy[opt]['vision_alone']['corr'])}, "
                  f"alone probe {fmt(sy[opt]['vision_alone_probe'])}, "
                  f"joint base probe {fmt(sy[opt]['grid']['baseline']['probe']['vision'])}")


if __name__ == "__main__":
    main()
