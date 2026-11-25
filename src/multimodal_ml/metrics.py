from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from multimodal_ml.config import DatasetConfig


def _bin_index(values: np.ndarray, low: float, high: float, num_bins: int) -> np.ndarray:
    """Map values in [low, high] to the nearest of `num_bins` evenly spaced classes.

    Generic equal-width binning parameterized by the dataset's label range — not
    bit-identical to any one paper's eval script (those vary slightly across papers
    too), but gives a consistent, dataset-agnostic Acc-5/Acc-7 definition.
    """
    clipped = np.clip(values, low, high)
    scaled = (clipped - low) / (high - low) * (num_bins - 1)
    return np.round(scaled).astype(int)


def _multiclass_acc(preds: np.ndarray, labels: np.ndarray, low: float, high: float, num_bins: int) -> float:
    pred_bins = _bin_index(preds, low, high, num_bins)
    label_bins = _bin_index(labels, low, high, num_bins)
    return float(np.mean(pred_bins == label_bins))


def compute_regression_metrics(
    preds: np.ndarray, labels: np.ndarray, config: DatasetConfig
) -> dict[str, float]:
    """Shared project metrics, with generic rather than MMSA-specific binning.

    MAE, Pearson Corr, Acc-2 (has0: >=0 vs <0) and Acc-2 (non0: excludes exact-zero
    labels, >0 vs <0) with matching weighted F1, Acc-5, and Acc-7 (only when
    config.has_acc7 — MOSI-style datasets with a wide enough label range).
    """
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)

    # Pearson is undefined when either side is constant (zero std): report 0.0 instead of nan.
    if preds.size < 2 or preds.std() == 0 or labels.std() == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds, labels)[0, 1])

    metrics = {
        "mae": float(np.mean(np.abs(preds - labels))),
        "corr": corr,
        "acc5": _multiclass_acc(preds, labels, config.label_low, config.label_high, 5),
    }
    if config.has_acc7:
        metrics["acc7"] = _multiclass_acc(preds, labels, config.label_low, config.label_high, 7)

    has0_truth = labels >= 0
    has0_pred = preds >= 0
    metrics["acc2_has0"] = accuracy_score(has0_truth, has0_pred)
    metrics["f1_has0"] = f1_score(has0_truth, has0_pred, average="weighted")

    non_zero = labels != 0
    if non_zero.any():
        non0_truth = labels[non_zero] > 0
        non0_pred = preds[non_zero] > 0
        metrics["acc2_non0"] = accuracy_score(non0_truth, non0_pred)
        metrics["f1_non0"] = f1_score(non0_truth, non0_pred, average="weighted")

    return metrics


def compute_contribution_metrics(
    contributions: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, float]:
    """Per-modality dominance metrics for an additive fusion (prediction = sum of parts).

    For each modality m with contribution c_m over a whole split:
      - `{m}_share`: Var(c_m) / sum_k Var(c_k), the part of the prediction's spread
        coming from m. Shares sum to 1 (0 for all if every part is constant).
        Covariance between contributions is ignored, so this is a simple guide,
        not an exact variance decomposition.
      - `{m}_corr`: Pearson corr(c_m, labels), how well m's part alone tracks the
        labels (0 when either side is constant).
      - `{m}_mae`: MAE of c_m + sum_{k != m} mean(c_k), i.e. the prediction from m
        alone with the other modalities held at their average contribution.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    parts = {m: np.asarray(c, dtype=np.float64).reshape(-1) for m, c in contributions.items()}
    variances = {m: c.var() for m, c in parts.items()}
    total_var = sum(variances.values())
    means = {m: c.mean() for m, c in parts.items()}
    metrics = {}
    for m, c in parts.items():
        metrics[f"{m}_share"] = float(variances[m] / total_var) if total_var > 0 else 0.0
        constant = c.size < 2 or c.std() == 0 or labels.std() == 0
        metrics[f"{m}_corr"] = 0.0 if constant else float(np.corrcoef(c, labels)[0, 1])
        others = sum(means[k] for k in parts if k != m)
        metrics[f"{m}_mae"] = float(np.mean(np.abs(c + others - labels)))
    return metrics
