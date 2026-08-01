"""The direct test of the soft-mixing hypothesis: does `hard_two_stage`
(argmax dataset routing, no blending) degrade sharply on dataset-ambiguous
or out-of-distribution traffic relative to a soft-gated dataset-MoE
(`moe_dataset_soft`), which blends experts instead of committing to one?

Two legs, either or both may run depending on config:

1. **Cross-dataset near-neighbor leg** (always runs): within the pooled
   test set, find rows whose canonical class also appears in at least one
   OTHER active dataset, and whose harmonized feature vector's nearest
   neighbor of the same class lives in a different dataset than its own
   (via sklearn NearestNeighbors) -- these are exactly the samples where a
   hard dataset classifier is most likely to guess wrong, since they sit
   close to another dataset's version of the same behavior.
2. **Genuinely held-out dataset leg** (runs only if
   `evaluation.ood_holdout_dataset` is set): a dataset NOT in
   `active_datasets` at all. Since it was never in `active_datasets`, no
   per-dataset scaler was fit for it during `prepare_datasets`; this module
   fits ONE separate harmonizer over active_datasets' train splits plus the
   held-out dataset's own train split (purely for feature scaling -- this
   never touches model training/supervision) so its rows can be transformed
   into the same harmonized feature space the models were trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from data.harmonization import Harmonizer, TrainSplit
from data.loaders import load_dataset, stratified_split
from data.registry import get_spec
from training.dataset import BENIGN_LABEL, PreparedData


@dataclass
class OODEvalSplit:
    features: np.ndarray
    class_idx: np.ndarray
    dataset_name: np.ndarray
    description: str


def build_cross_dataset_ambiguous_split(data: PreparedData, k_neighbors: int = 5) -> OODEvalSplit:
    """Rows from the pooled TEST split whose same-canonical-class nearest
    neighbor (in harmonized feature space) belongs to a DIFFERENT dataset
    than the row itself -- i.e. exactly the traffic a hard dataset
    classifier is most likely to misroute.
    """
    features = data.test.features
    class_idx = data.test.class_idx
    dataset_name = data.test.dataset_name

    ambiguous_mask = np.zeros(len(features), dtype=bool)
    for cls in np.unique(class_idx):
        cls_mask = class_idx == cls
        cls_datasets = np.unique(dataset_name[cls_mask])
        if len(cls_datasets) < 2:
            continue  # class only appears in one dataset here -- no cross-dataset ambiguity possible
        cls_idx_global = np.flatnonzero(cls_mask)
        cls_features = features[cls_idx_global]
        cls_dataset_names = dataset_name[cls_idx_global]

        n_neighbors = min(k_neighbors + 1, len(cls_idx_global))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(cls_features)
        _, neighbor_idx = nn.kneighbors(cls_features)

        for row_i in range(len(cls_idx_global)):
            own_dataset = cls_dataset_names[row_i]
            neighbor_datasets = cls_dataset_names[neighbor_idx[row_i, 1:]]  # exclude self (col 0)
            if np.any(neighbor_datasets != own_dataset):
                ambiguous_mask[cls_idx_global[row_i]] = True

    return OODEvalSplit(
        features=features[ambiguous_mask],
        class_idx=class_idx[ambiguous_mask],
        dataset_name=dataset_name[ambiguous_mask],
        description=f"cross_dataset_ambiguous (n={int(ambiguous_mask.sum())}, k_neighbors={k_neighbors})",
    )


def build_held_out_dataset_split(config: dict, data: PreparedData, holdout_dataset_name: str) -> OODEvalSplit:
    """Genuinely out-of-distribution leg: `holdout_dataset_name` must NOT be
    in `data.active_datasets` (never seen during training at all -- neither
    by any expert nor the gate). Fits a scratch harmonizer (active datasets'
    train splits + the holdout's own train split) purely to place its rows
    in the shared harmonized feature space; this scratch harmonizer is used
    for feature scaling ONLY and is never involved in model training.
    """
    if holdout_dataset_name in data.active_datasets:
        raise ValueError(
            f"'{holdout_dataset_name}' is in data.active_datasets -- the held-out-dataset OOD leg "
            "requires a dataset the model never saw during training at all."
        )

    data_cfg = config["data"]
    spec = get_spec(holdout_dataset_name)
    raw = load_dataset(
        spec,
        chunksize=data_cfg.get("chunksize", 200_000),
        mapping_override=data_cfg.get("label_mapping", {}).get(holdout_dataset_name),
        strict=False,  # OOD leg only needs *a* label for reporting, not full taxonomy coverage
        max_rows=data_cfg.get("max_rows_per_dataset"),
        seed=config.get("seed", 0),
    )
    split = stratified_split(raw, seed=config.get("seed", 0), min_class_count=data_cfg["split"].get("min_class_count", 3))

    all_names = [*data.active_datasets, holdout_dataset_name]
    scratch_harmonizer = Harmonizer(all_names, divergent_features=frozenset(data_cfg.get("divergent_features", [])))
    # Re-fit requires every active dataset's own train frame too; the caller
    # is expected to have `data` built from the same config, so recompute
    # nothing else here besides adding the holdout dataset's fit.
    scratch_harmonizer.fit([TrainSplit(holdout_dataset_name, split.train)])
    for name in data.active_datasets:
        # Reuse the ALREADY-fitted per-dataset scaler state from `data.harmonizer`
        # instead of re-loading/re-splitting every active dataset's raw data again.
        scratch_harmonizer._fit_state[name] = data.harmonizer._fit_state[name]  # noqa: SLF001

    class_to_idx = {c: i for i, c in enumerate(data.class_names)}
    features = scratch_harmonizer.transform(split.test, holdout_dataset_name)
    mapped = split.test["canonical_label"].map(class_to_idx)
    keep = mapped.notna()
    class_idx = mapped[keep].to_numpy().astype(np.int64)
    features = features[keep.to_numpy()]

    return OODEvalSplit(
        features=features,
        class_idx=class_idx,
        dataset_name=np.full(len(class_idx), holdout_dataset_name),
        description=f"held_out_dataset={holdout_dataset_name} (n={len(class_idx)}, rows whose label falls outside "
        f"the trained class space were dropped from this comparison)",
    )


def summarize_ood_comparison(
    split: OODEvalSplit, predictions_by_variant: dict[str, np.ndarray]
) -> pd.DataFrame:
    """One row per variant: accuracy + macro-recall on `split`. Callers pass
    predictions already computed via each variant's own predict() over
    `split.features` (hard_two_stage vs moe_dataset_soft, primarily).
    """
    rows = []
    for variant, preds in predictions_by_variant.items():
        correct = preds == split.class_idx
        per_class_recall = []
        for cls in np.unique(split.class_idx):
            mask = split.class_idx == cls
            per_class_recall.append((preds[mask] == cls).mean())
        rows.append(
            {
                "variant": variant,
                "split": split.description,
                "n": len(split.class_idx),
                "accuracy": float(correct.mean()) if len(correct) else float("nan"),
                "macro_recall": float(np.mean(per_class_recall)) if per_class_recall else float("nan"),
            }
        )
    return pd.DataFrame(rows)
