"""Class-balanced batch sampler: guarantees at least `min_per_class` samples
of every class in every batch, built at the batch-construction level (not
via raw duplication of the underlying dataset).
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, batch_size: int, min_per_class: int, seed: int = 0, num_batches: int | None = None) -> None:
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.rng = np.random.default_rng(seed)
        self.classes = np.unique(self.labels)
        self.indices_by_class = {c: np.flatnonzero(self.labels == c) for c in self.classes}
        guaranteed = len(self.classes) * min_per_class
        if guaranteed > batch_size:
            raise ValueError(
                f"min_per_class ({min_per_class}) * num_classes ({len(self.classes)}) = {guaranteed} "
                f"exceeds batch_size ({batch_size}); lower min_per_class, raise batch_size, or reduce active_classes."
            )
        self.num_batches = num_batches or max(1, len(self.labels) // batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            guaranteed_idx = []
            for c in self.classes:
                pool = self.indices_by_class[c]
                replace = len(pool) < self.min_per_class
                guaranteed_idx.append(self.rng.choice(pool, size=self.min_per_class, replace=replace))
            guaranteed_idx = np.concatenate(guaranteed_idx)

            remaining = self.batch_size - len(guaranteed_idx)
            if remaining > 0:
                fill_idx = self.rng.choice(len(self.labels), size=remaining, replace=len(self.labels) < remaining)
                batch = np.concatenate([guaranteed_idx, fill_idx])
            else:
                batch = guaranteed_idx
            self.rng.shuffle(batch)
            yield batch.tolist()
