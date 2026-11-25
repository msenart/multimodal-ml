"""Aggregate the OGM-GE study runs (results/ogm_ge/runs/*.json) into tables and a figure.

    .venv/bin/python experiments/ogm_ge/aggregate.py

Writes to results/ogm_ge/:
  lr_selection.csv  mean validation MAE per (variant, lr)
  alpha.csv         alpha ablation (validation and test)
  final.csv         selected configuration of each variant, test metrics
  paired_tests.csv  paired t-test / Wilcoxon over seeds, OGM-GE vs each other variant
  masked.csv        test metrics with modalities zeroed at inference
  probes.csv        linear probes on the frozen encoders
  dynamics.csv      mean k and rho per epoch
and the figure report/figures/ogm_ge_dynamics.svg. Metrics are given x100 (as MMSA).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_study import (  # noqa: E402
    ALPHAS, DEFAULT_ALPHA, LR_GRID, OGM_VARIANTS, OUT_DIR, ROOT, SEEDS, VARIANTS, load_runs, selected_lr,
)

MODALITIES = ("text", "audio", "vision")
METRICS = ("acc2_has0", "f1_has0", "mae", "corr")
FIG = ROOT / "report" / "figures" / "ogm_ge_dynamics.svg"


def ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float) * 100
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(values) -> str:
    m, s = ms(values)
    return f"{m:.2f} ± {s:.2f}"


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n== {name}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for r in [header, *rows]:
        print("  ".join(str(x).ljust(wd) for x, wd in zip(r, widths)))


def by_seed(runs: list[dict]) -> dict[int, dict]:
    out = {r["seed"]: r for r in runs}
    assert sorted(out) == SEEDS, f"missing seeds: {sorted(set(SEEDS) - set(out))}"
    return out


def main() -> None:
    runs = load_runs()

    # 1. lr selection
    rows = []
    for variant, (opt, _, _) in VARIANTS.items():
        for lr in LR_GRID[opt]:
            sel = [r for r in runs if r["variant"] == variant and r["lr"] == lr
                   and (variant not in OGM_VARIANTS or r["alpha"] == DEFAULT_ALPHA)]
            ok = [r for r in sel if not r.get("diverged")]
            n_div = len(sel) - len(ok)
            val = fmt([r["val"]["mae"] for r in ok]) if not n_div else "diverged"
            ep = f"{np.mean([r['best_epoch'] for r in ok]):.1f}" if ok else "-"
            rows.append([variant, f"{lr:g}", val, ep, len(sel), n_div])
    write_csv("lr_selection.csv", ["variant", "lr", "val_mae", "best_epoch", "n", "n_diverged"], rows)
    lrs = {v: selected_lr(runs, v) for v in VARIANTS}

    # 2. alpha ablation (at the selected lr); final alpha = lowest mean val MAE
    rows, best_alpha = [], {}
    for variant in OGM_VARIANTS:
        scores = {}
        for alpha in ALPHAS:
            sel = [r for r in runs if r["variant"] == variant and r["lr"] == lrs[variant] and r["alpha"] == alpha]
            if len(sel) != len(SEEDS):
                continue
            scores[alpha] = np.mean([r["val"]["mae"] for r in sel])
            rows.append([variant, alpha, fmt([r["val"]["mae"] for r in sel]),
                         *(fmt([r["test"][m] for r in sel]) for m in METRICS),
                         f"{np.mean([r['history'][r['best_epoch'] - 1]['k']['text'] for r in sel]):.3f}"])
        best_alpha[variant] = min(scores, key=scores.get) if scores else DEFAULT_ALPHA
    write_csv("alpha.csv", ["variant", "alpha", "val_mae", *(f"test_{m}" for m in METRICS), "k_text"], rows)

    # 3. final configuration of each variant
    final = {}
    for variant in VARIANTS:
        sel = [r for r in runs if r["variant"] == variant and r["lr"] == lrs[variant]
               and (variant not in OGM_VARIANTS or r["alpha"] == best_alpha[variant])]
        final[variant] = by_seed(sel)
    rows = []
    for variant, rs in final.items():
        sel = list(rs.values())
        rows.append([variant, f"{lrs[variant]:g}", best_alpha.get(variant, ""),
                     *(fmt([r["test"][m] for r in sel]) for m in METRICS),
                     fmt([r["test"]["acc2_non0"] for r in sel]), fmt([r["val"]["mae"] for r in sel])])
    write_csv("final.csv", ["variant", "lr", "alpha", *METRICS, "acc2_non0", "val_mae"], rows)

    # 4. paired tests (same seed = same initialisation and same first shuffle)
    rows = []
    ref = final["ogm_ge"]
    for other in [v for v in VARIANTS if v != "ogm_ge"]:
        for m in ("acc2_has0", "mae", "corr"):
            a = np.array([ref[s]["test"][m] for s in SEEDS]) * 100
            b = np.array([final[other][s]["test"][m] for s in SEEDS]) * 100
            d = a - b
            p_t = ttest_rel(a, b).pvalue if d.std() > 0 else 1.0
            p_w = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
            rows.append([f"ogm_ge - {other}", m, f"{d.mean():+.2f}", f"{d.std(ddof=1):.2f}",
                         f"{p_t:.3f}", f"{p_w:.3f}", int((d > 0).sum())])
    write_csv("paired_tests.csv", ["comparison", "metric", "mean_diff", "std_diff", "p_ttest",
                                   "p_wilcoxon", "n_seeds_ogm_higher"], rows)

    # 5. masked evaluation and 6. linear probes (final configurations)
    rows = []
    for variant, rs in final.items():
        sel = list(rs.values())
        for mask in ("TAV", "T", "A", "V", "AV"):
            rows.append([variant, mask, fmt([r["masked_test"][mask]["acc2_has0"] for r in sel]),
                         fmt([r["masked_test"][mask]["corr"] for r in sel]),
                         fmt([r["masked_test"][mask]["mae"] for r in sel])])
    write_csv("masked.csv", ["variant", "kept", "acc2_has0", "corr", "mae"], rows)

    rows = []
    for variant, rs in final.items():
        sel = list(rs.values())
        for m in MODALITIES:
            rows.append([variant, m, fmt([r["probe_test"][m]["acc2_has0"] for r in sel]),
                         fmt([r["probe_test"][m]["corr"] for r in sel]),
                         fmt([r["probe_test"][m]["mae"] for r in sel])])
    write_csv("probes.csv", ["variant", "modality", "acc2_has0", "corr", "mae"], rows)

    # 7. training dynamics (mean over the seeds still training at each epoch)
    dyn = {}
    rows = []
    for variant in ("linear_sgd", "ogm_ge"):
        hist = [r["history"] for r in final[variant].values()]
        n_ep = max(len(h) for h in hist)
        series = {f"{q}_{m}": [] for q in ("k", "rho") for m in MODALITIES}
        for e in range(n_ep):
            alive = [h[e] for h in hist if len(h) > e]
            if len(alive) < len(hist) / 2:  # stop when most runs have early-stopped
                break
            for q in ("k", "rho"):
                for m in MODALITIES:
                    series[f"{q}_{m}"].append(float(np.mean([a[q][m] for a in alive])))
            rows.append([variant, e + 1, len(alive), *(f"{series[k][-1]:.4f}" for k in series)])
        dyn[variant] = series
    write_csv("dynamics.csv", ["variant", "epoch", "n_runs", *dyn["ogm_ge"].keys()], rows)
    write_figure(dyn)


def write_figure(dyn: dict[str, dict[str, list[float]]]) -> None:
    """Two-panel line chart as plain SVG (no plotting dependency)."""
    colors = {"text": "#2a6fdb", "audio": "#d9822b", "vision": "#3a9a5b"}
    W, H, pad_l, pad_r, pad_t, pad_b, gap = 760, 300, 52, 16, 34, 44, 60
    pw = (W - pad_l - pad_r - gap) / 2
    ph = H - pad_t - pad_b
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']

    def panel(x0: float, title: str, lines: list[tuple[list[float], str, bool]], ymin: float, ymax: float,
              yticks: list[float], ref: float | None) -> None:
        n = max(len(v) for v, _, _ in lines)
        sx = lambda i: x0 + i / max(n - 1, 1) * pw  # noqa: E731
        sy = lambda y: pad_t + (1 - (y - ymin) / (ymax - ymin)) * ph  # noqa: E731
        out.append(f'<text x="{x0}" y="{pad_t - 12}" font-weight="bold">{title}</text>')
        out.append(f'<rect x="{x0}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>')
        for t in yticks:
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{x0 - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
        if ref is not None:
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" '
                       f'stroke="#555" stroke-dasharray="2,3"/>')
        step = 10 if n > 30 else 5
        for e in range(0, n, step):
            out.append(f'<text x="{sx(e):.1f}" y="{pad_t + ph + 16}" text-anchor="middle">{e + 1}</text>')
        out.append(f'<text x="{x0 + pw / 2}" y="{H - 8}" text-anchor="middle">epoch</text>')
        for values, color, dashed in lines:
            pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
            dash = ' stroke-dasharray="6,4"' if dashed else ""
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')

    rho_all = [v for d in dyn.values() for m in MODALITIES for v in d[f"rho_{m}"]]
    lo, hi = np.floor(min(rho_all) * 10) / 10, np.ceil(max(rho_all) * 10) / 10
    panel(pad_l, "(a) discrepancy ratio ρ", [
        *((dyn["linear_sgd"][f"rho_{m}"], colors[m], False) for m in MODALITIES),
        *((dyn["ogm_ge"][f"rho_{m}"], colors[m], True) for m in MODALITIES),
    ], lo, hi, [round(t, 1) for t in np.arange(lo, hi + 1e-9, 0.2)], 1.0)
    panel(pad_l + pw + gap, "(b) modulation coefficient k (OGM-GE)", [
        (dyn["ogm_ge"][f"k_{m}"], colors[m], True) for m in MODALITIES
    ], 0.0, 1.05, [0, 0.25, 0.5, 0.75, 1.0], None)

    # legend
    lx, ly = pad_l + 8, pad_t + 14
    for i, m in enumerate(MODALITIES):
        out.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly + i * 15}" y2="{ly + i * 15}" '
                   f'stroke="{colors[m]}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 23}" y="{ly + i * 15 + 4}">{m}</text>')
    out.append(f'<text x="{lx + 80}" y="{ly + 4}">solid: no modulation</text>')
    out.append(f'<text x="{lx + 80}" y="{ly + 19}">dashed: OGM-GE</text>')
    out.append("</svg>")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    FIG.write_text("\n".join(out))
    print(f"\nfigure: {FIG}")


if __name__ == "__main__":
    main()
