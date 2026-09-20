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


class ClassDomainBalancedBatchSampler(Sampler):
    """Guarantee class positives and cross-domain positives when available.

    The remainder of each batch follows the empirical row distribution, so
    balancing does not turn every batch into a fully uniform class sample.
    """

    def __init__(
        self,
        labels: np.ndarray,
        dataset_ids: np.ndarray,
        batch_size: int,
        min_per_class: int,
        seed: int = 0,
        num_batches: int | None = None,
    ) -> None:
        self.labels = np.asarray(labels)
        self.dataset_ids = np.asarray(dataset_ids)
        if self.labels.shape != self.dataset_ids.shape or self.labels.ndim != 1:
            raise ValueError("labels and dataset_ids must be one-dimensional arrays with equal shape")
        if min_per_class < 2:
            raise ValueError("class/domain-balanced contrastive batches require min_per_class >= 2")
        self.batch_size = int(batch_size)
        self.min_per_class = int(min_per_class)
        self.seed = int(seed)
        self.classes = np.unique(self.labels)
        guaranteed = len(self.classes) * self.min_per_class
        if guaranteed > self.batch_size:
            raise ValueError(
                f"min_per_class ({min_per_class}) * present classes ({len(self.classes)}) "
                f"exceeds batch_size ({batch_size})"
            )
        self.num_batches = num_batches or max(1, len(self.labels) // self.batch_size)
        self.pools = {}
        self.domains_by_class = {}
        for class_id in self.classes:
            class_rows = np.flatnonzero(self.labels == class_id)
            local_domains = self.dataset_ids[class_rows]
            domains = np.unique(local_domains)
            self.domains_by_class[class_id] = domains
            for domain_id in domains:
                self.pools[(class_id, domain_id)] = class_rows[local_domains == domain_id]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_batches):
            guaranteed = []
            for class_id in self.classes:
                domains = self.domains_by_class[class_id]
                first = rng.permutation(domains)[: min(len(domains), self.min_per_class)]
                extra_count = self.min_per_class - len(first)
                extra = rng.choice(domains, size=extra_count, replace=True) if extra_count else np.empty(0, dtype=domains.dtype)
                selected_domains = np.concatenate([first, extra])
                for domain_id in selected_domains:
                    pool = self.pools[(class_id, domain_id)]
                    guaranteed.append(int(rng.choice(pool)))
            remaining = self.batch_size - len(guaranteed)
            fill = rng.choice(len(self.labels), size=remaining, replace=len(self.labels) < remaining)
            batch = np.concatenate([np.asarray(guaranteed, dtype=np.int64), fill])
            rng.shuffle(batch)
            yield batch.tolist()


def class_domain_balanced_row_batches(
    labels,
    dataset_ids,
    batch_size: int,
    min_per_class: int,
    seed: int,
    num_batches: int | None = None,
):
    """Out-of-core-friendly wrapper yielding NumPy row-index arrays."""
    def materialize(values):
        if isinstance(values, np.ndarray):
            return np.asarray(values)
        if hasattr(values, "materialize"):
            return np.asarray(values.materialize())
        output = np.empty(len(values), dtype=getattr(values, "dtype", np.int64))
        for start in range(0, len(values), 1_000_000):
            output[start : start + 1_000_000] = values[start : start + 1_000_000]
        return output

    sampler = ClassDomainBalancedBatchSampler(
        materialize(labels), materialize(dataset_ids), batch_size, min_per_class,
        seed=seed, num_batches=num_batches,
    )
    for batch in sampler:
        yield np.asarray(batch, dtype=np.int64)
