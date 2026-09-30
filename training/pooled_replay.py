"""Deterministic, bounded-memory helpers for Stage-B pooled replay."""
from __future__ import annotations

import numpy as np


def replay_rows_for_owned_batch(owned_rows: int, replay_fraction: float) -> int:
    """Rows needed so replay is ``replay_fraction`` of the combined batch."""
    fraction = float(replay_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("Stage-B replay_fraction must be in [0, 1)")
    if owned_rows <= 0 or fraction == 0.0:
        return 0
    return max(1, int(round(owned_rows * fraction / (1.0 - fraction))))


def build_stratified_replay_reservoir(
    labels,
    dataset_ids,
    *,
    max_per_class_dataset: int,
    seed: int,
    chunk_rows: int = 1_000_000,
) -> dict[int, np.ndarray]:
    """Uniformly reservoir-sample each present class/dataset stratum.

    Random priorities make the bounded reservoir independent of disk order.
    The returned pools are grouped by class; equal per-domain caps prevent a
    large source dataset from completely dominating a class's replay pool.
    """
    cap = int(max_per_class_dataset)
    if cap <= 0:
        raise ValueError("replay_pool_per_class_dataset must be positive")
    if len(labels) != len(dataset_ids):
        raise ValueError("labels and dataset_ids must have equal length")
    rng = np.random.default_rng(int(seed))
    reservoirs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(labels), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(labels))
        local_labels = np.asarray(labels[start:stop], dtype=np.int64)
        local_datasets = np.asarray(dataset_ids[start:stop], dtype=np.int64)
        pairs = np.unique(np.stack([local_labels, local_datasets], axis=1), axis=0)
        for class_id, dataset_id in pairs:
            local = np.flatnonzero(
                (local_labels == class_id) & (local_datasets == dataset_id)
            )
            row_ids = local.astype(np.int64, copy=False) + start
            priorities = rng.random(len(row_ids))
            key = (int(class_id), int(dataset_id))
            if key in reservoirs:
                old_rows, old_priorities = reservoirs[key]
                row_ids = np.concatenate([old_rows, row_ids])
                priorities = np.concatenate([old_priorities, priorities])
            if len(row_ids) > cap:
                keep = np.argpartition(priorities, cap - 1)[:cap]
                row_ids = row_ids[keep]
                priorities = priorities[keep]
            reservoirs[key] = row_ids, priorities
    by_class: dict[int, list[np.ndarray]] = {}
    for (class_id, _dataset_id), (row_ids, _priorities) in reservoirs.items():
        by_class.setdefault(class_id, []).append(row_ids)
    return {
        class_id: np.concatenate(parts).astype(np.int64, copy=False)
        for class_id, parts in sorted(by_class.items())
        if parts
    }


def draw_class_balanced_replay_rows(
    pools_by_class: dict[int, np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a near-uniform class allocation from a bounded replay reservoir."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    classes = np.asarray(
        [class_id for class_id, rows in sorted(pools_by_class.items()) if len(rows)],
        dtype=np.int64,
    )
    if not len(classes):
        raise ValueError("pooled replay reservoir contains no rows")
    allocations = np.full(len(classes), count // len(classes), dtype=np.int64)
    remainder = count % len(classes)
    if remainder:
        allocations[rng.permutation(len(classes))[:remainder]] += 1
    selected = []
    for class_id, amount in zip(classes, allocations):
        if amount == 0:
            continue
        pool = pools_by_class[int(class_id)]
        selected.append(rng.choice(pool, size=int(amount), replace=len(pool) < amount))
    output = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(output)
    return output
