"""MMSA-compatible regression metrics, kept separate from team-wide metrics.

Definitions checked against thuiar/MMSA commit
a94e65d07fa1ae0d44e552390074b29b0898edfd, src/MMSA/utils/metricsTop.py.
In particular, SIMS clips scores and puts zero in the negative binary class;
MOSI Acc-5 clips to [-2,2] and rounds, rather than dividing [-3,3] equally.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_mmsa_metrics(predictions, labels, dataset: str) -> dict[str, float]:
    p, y = np.asarray(predictions).reshape(-1), np.asarray(labels).reshape(-1)
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Expected matching nonempty finite prediction and label arrays")
    if dataset not in ("mosi", "sims"):
        raise ValueError("Supported datasets: mosi, sims")
    if dataset == "sims":
        p, y = np.clip(p, -1, 1), np.clip(y, -1, 1)
    metrics = {"MAE": float(np.abs(p - y).mean()), "Corr": float(np.corrcoef(p, y)[0, 1])}
    if dataset == "mosi":
        for bins, limit in ((5, 2), (7, 3)):
            metrics[f"Mult_acc_{bins}"] = float((np.round(np.clip(p, -limit, limit)) == np.round(np.clip(y, -limit, limit))).mean())
        for prefix, mask, inclusive in (("Has0", np.ones(len(y), dtype=bool), True), ("Non0", y != 0, False)):
            truth = y[mask] >= 0 if inclusive else y[mask] > 0
            guessed = p[mask] >= 0 if inclusive else p[mask] > 0
            metrics[f"{prefix}_acc_2"] = float(accuracy_score(truth, guessed))
            metrics[f"{prefix}_F1_score"] = float(f1_score(truth, guessed, average="weighted", zero_division=0))
    else:
        def bins(values, thresholds):
            indices = np.zeros(len(values), dtype=np.int64)
            for threshold in thresholds:
                indices += values > threshold
            return indices

        for count, thresholds in ((2, [0.0]), (3, [-0.1, 0.1]), (5, [-0.7, -0.1, 0.1, 0.7])):
            metrics[f"Mult_acc_{count}"] = float(accuracy_score(bins(y, thresholds), bins(p, thresholds)))
        metrics["F1_score"] = float(f1_score(y > 0, p > 0, average="weighted", zero_division=0))
    return {key: round(value, 4) for key, value in metrics.items()}
