"""Cross-dataset diagnostic utilities. Any code here that compares
distributions/metrics across datasets must assert it is comparing the same
attack class -- never mixing classes silently. This is the one required
enforcement point; add new cross-dataset comparisons through
`assert_same_class` rather than reimplementing the check ad hoc.
"""
from __future__ import annotations

import numpy as np


class SameClassComparisonError(ValueError):
    pass


def assert_same_class(class_label_by_dataset: dict[str, str]) -> str:
    """Raises unless every dataset in the comparison names the same class."""
    labels = set(class_label_by_dataset.values())
    if len(labels) != 1:
        raise SameClassComparisonError(
            f"Cross-dataset comparison must compare the same class everywhere; got {class_label_by_dataset}"
        )
    return next(iter(labels))


def compare_class_distribution(
    values_by_dataset: dict[str, np.ndarray],
    class_label_by_dataset: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-dataset summary stats (mean/std/count) for ONE class, verified to
    be the same class across every dataset before any values are touched."""
    assert_same_class(class_label_by_dataset)
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": int(len(values))}
        for name, values in values_by_dataset.items()
    }


def compare_class_recall(
    recall_by_dataset: dict[str, float],
    class_label_by_dataset: dict[str, str],
) -> dict[str, float]:
    """Per-dataset recall for ONE class, same-class-verified."""
    assert_same_class(class_label_by_dataset)
    return dict(recall_by_dataset)
