"""Classification metrics shared by every in-memory training variant.

ROC-AUC is computed one-vs-rest from per-class probabilities.  A class whose
evaluation slice contains only positives or only negatives has an undefined
AUC and is reported as NaN; macro/weighted aggregates use the remaining
evaluable classes.  Passing scores is optional for backwards compatibility,
but all training entry points pass them.

Macro-F1 and per-class precision/recall/F1 remain the primary metrics
everywhere in this repo (ported unchanged from moe_nids/evaluation/metrics.py).
Weighted F1 is computed only for reference logging and must never be used
for model selection or headline reporting.

NEW here: `evaluate_per_dataset`, since this project's whole point is
measuring which specific datasets get rescued or hurt relative to
`plain_pooled`/`no_fusion` -- a pooled macro-F1 alone can't show that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


@dataclass
class ClassMetrics:
    class_name: str
    support: int
    precision: float
    recall: float
    f1: float
    roc_auc_ovr: float
    pr_auc_ovr: float


@dataclass
class EvaluationResult:
    class_names: list[str]
    per_class: list[ClassMetrics]
    macro_f1: float
    weighted_f1: float  # reference-only, see module docstring
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    weighted_precision: float
    weighted_recall: float
    roc_auc_ovr_macro: float
    roc_auc_ovr_weighted: float
    roc_auc_ovr_micro: float
    pr_auc_ovr_macro: float
    pr_auc_ovr_weighted: float
    pr_auc_ovr_micro: float
    confusion: np.ndarray = field(repr=False)


def _roc_auc_ovr(
    y_true: np.ndarray, y_score: np.ndarray | None, num_classes: int
) -> tuple[np.ndarray, float, float, float]:
    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    if y_score is None:
        return per_class, float("nan"), float("nan"), float("nan")
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.shape != (len(y_true), num_classes):
        raise ValueError(
            f"y_score must have shape ({len(y_true)}, {num_classes}), got {y_score.shape}"
        )
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values")

    support = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    for class_idx in range(num_classes):
        binary_truth = y_true == class_idx
        if binary_truth.any() and (~binary_truth).any():
            per_class[class_idx] = roc_auc_score(binary_truth, y_score[:, class_idx])
    valid = np.isfinite(per_class)
    macro = float(per_class[valid].mean()) if valid.any() else float("nan")
    weighted = (
        float(np.average(per_class[valid], weights=support[valid]))
        if valid.any() and support[valid].sum() > 0
        else float("nan")
    )
    one_hot = np.eye(num_classes, dtype=np.uint8)[y_true]
    flat_truth = one_hot.ravel()
    micro = (
        float(roc_auc_score(flat_truth, y_score.ravel()))
        if flat_truth.any() and (~flat_truth.astype(bool)).any()
        else float("nan")
    )
    return per_class, macro, weighted, micro


def _pr_auc_ovr(
    y_true: np.ndarray, y_score: np.ndarray | None, num_classes: int
) -> tuple[np.ndarray, float, float, float]:
    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    if y_score is None:
        return per_class, float("nan"), float("nan"), float("nan")
    support = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    for class_idx in range(num_classes):
        binary_truth = y_true == class_idx
        if binary_truth.any() and (~binary_truth).any():
            per_class[class_idx] = average_precision_score(binary_truth, y_score[:, class_idx])
    valid = np.isfinite(per_class)
    macro = float(per_class[valid].mean()) if valid.any() else float("nan")
    weighted = (
        float(np.average(per_class[valid], weights=support[valid]))
        if valid.any() and support[valid].sum() else float("nan")
    )
    one_hot = np.eye(num_classes, dtype=np.uint8)[y_true]
    micro = float(average_precision_score(one_hot.ravel(), y_score.ravel())) if len(y_true) else float("nan")
    return per_class, macro, weighted, micro


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray | None = None,
) -> EvaluationResult:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shape")
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_auc, roc_macro, roc_weighted, roc_micro = _roc_auc_ovr(
        y_true, y_score, len(class_names)
    )
    per_class_pr, pr_macro, pr_weighted, pr_micro = _pr_auc_ovr(
        y_true, y_score, len(class_names)
    )
    per_class = [
        ClassMetrics(
            class_names[i], int(support[i]), float(precision[i]), float(recall[i]),
            float(f1[i]), float(per_class_auc[i]), float(per_class_pr[i]),
        )
        for i in labels
    ]
    macro_f1 = float(np.mean(f1))
    weighted_precision, weighted_recall, weighted_f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = support > 0
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return EvaluationResult(
        class_names=class_names,
        per_class=per_class,
        macro_f1=macro_f1,
        weighted_f1=float(weighted_f1_arr),
        accuracy=accuracy,
        balanced_accuracy=float(recall[present].mean()) if present.any() else float("nan"),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        micro_precision=float(micro_precision),
        micro_recall=float(micro_recall),
        micro_f1=float(micro_f1),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        roc_auc_ovr_macro=roc_macro,
        roc_auc_ovr_weighted=roc_weighted,
        roc_auc_ovr_micro=roc_micro,
        pr_auc_ovr_macro=pr_macro,
        pr_auc_ovr_weighted=pr_weighted,
        pr_auc_ovr_micro=pr_micro,
        confusion=cm,
    )


def evaluate_per_dataset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray | None = None,
) -> dict[str, EvaluationResult]:
    """Per-dataset macro-F1 + per-class metrics, broken out by the source
    dataset of each TEST row (bookkeeping-only metadata, never a model
    input -- see training/dataset.py::PreparedSplit.dataset_name). This is
    the primary way to measure whether a given architecture rescues or
    hurts a specific dataset relative to plain_pooled/no_fusion, the core
    empirical question this project exists to answer.
    """
    results: dict[str, EvaluationResult] = {}
    for name in sorted(set(dataset_name.tolist())):
        mask = dataset_name == name
        if mask.sum() == 0:
            continue
        scores = y_score[mask] if y_score is not None else None
        results[name] = evaluate_predictions(y_true[mask], y_pred[mask], class_names, scores)
    return results


def per_class_recall_delta(before: EvaluationResult, after: EvaluationResult) -> dict[str, float]:
    """Per-class recall before/after, required for every experiment."""
    before_by_name = {c.class_name: c.recall for c in before.per_class}
    after_by_name = {c.class_name: c.recall for c in after.per_class}
    return {name: after_by_name[name] - before_by_name.get(name, 0.0) for name in after_by_name}
