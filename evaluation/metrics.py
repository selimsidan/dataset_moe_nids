"""Macro-F1 and per-class precision/recall/F1 -- the primary metrics
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass
class ClassMetrics:
    class_name: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    class_names: list[str]
    per_class: list[ClassMetrics]
    macro_f1: float
    weighted_f1: float  # reference-only, see module docstring
    confusion: np.ndarray = field(repr=False)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> EvaluationResult:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = [
        ClassMetrics(class_names[i], int(support[i]), float(precision[i]), float(recall[i]), float(f1[i]))
        for i in labels
    ]
    macro_f1 = float(np.mean(f1))
    _, _, weighted_f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return EvaluationResult(
        class_names=class_names,
        per_class=per_class,
        macro_f1=macro_f1,
        weighted_f1=float(weighted_f1_arr),
        confusion=cm,
    )


def evaluate_per_dataset(
    y_true: np.ndarray, y_pred: np.ndarray, dataset_name: np.ndarray, class_names: list[str]
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
        results[name] = evaluate_predictions(y_true[mask], y_pred[mask], class_names)
    return results


def per_class_recall_delta(before: EvaluationResult, after: EvaluationResult) -> dict[str, float]:
    """Per-class recall before/after, required for every experiment."""
    before_by_name = {c.class_name: c.recall for c in before.per_class}
    after_by_name = {c.class_name: c.recall for c in after.per_class}
    return {name: after_by_name[name] - before_by_name.get(name, 0.0) for name in after_by_name}
