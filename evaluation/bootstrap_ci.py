"""Bootstrap confidence intervals for per-class recall/F1, required for
classes with test-sample counts below `low_sample_threshold` -- point
estimates on a handful of rare-class test samples are not precise enough to
present without a CI, and doing so silently overstates confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    class_name: str
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_samples: int
    is_low_sample: bool


def _recall(y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:
    mask = y_true == class_idx
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == class_idx).mean())


def _f1(y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:
    tp = int(((y_true == class_idx) & (y_pred == class_idx)).sum())
    fp = int(((y_true != class_idx) & (y_pred == class_idx)).sum())
    fn = int(((y_true == class_idx) & (y_pred != class_idx)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_METRIC_FNS = {"recall": _recall, "f1": _f1}


def bootstrap_per_class_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    low_sample_threshold: int = 200,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    metrics: tuple[str, ...] = ("recall", "f1"),
) -> list[BootstrapCI]:
    rng = np.random.default_rng(seed)
    results: list[BootstrapCI] = []

    for class_idx, class_name in enumerate(class_names):
        class_mask = y_true == class_idx
        n_samples = int(class_mask.sum())
        is_low = n_samples < low_sample_threshold

        for metric in metrics:
            fn = _METRIC_FNS[metric]
            point = fn(y_true, y_pred, class_idx)
            if n_samples == 0:
                results.append(BootstrapCI(class_name, metric, point, float("nan"), float("nan"), 0, is_low))
                continue
            if not is_low:
                # still reported, but flagged False so callers can render point estimates
                # as sufficiently precise without a CI if they choose.
                results.append(BootstrapCI(class_name, metric, point, point, point, n_samples, False))
                continue

            class_idxs = np.flatnonzero(class_mask)
            other_idxs = np.flatnonzero(~class_mask)
            boot_vals = []
            for _ in range(n_bootstrap):
                resampled_class = rng.choice(class_idxs, size=len(class_idxs), replace=True)
                resampled_other = rng.choice(other_idxs, size=len(other_idxs), replace=True) if len(other_idxs) else np.array([], dtype=int)
                idx = np.concatenate([resampled_class, resampled_other])
                boot_vals.append(fn(y_true[idx], y_pred[idx], class_idx))
            boot_vals = np.array(boot_vals)
            lo, hi = np.quantile(boot_vals, [alpha / 2, 1 - alpha / 2])
            results.append(BootstrapCI(class_name, metric, point, float(lo), float(hi), n_samples, True))

    return results
