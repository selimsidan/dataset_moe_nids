"""Harmonization layer: turns per-dataset raw rows into one fixed-width
harmonized feature vector (+ presence mask), independent of source dataset.

Feature taxonomy (per the project spec):
  (a) common/aligned    -- canonical feature present (via alias) in every
                            active dataset; scaled per-dataset, fit on that
                            dataset's TRAIN split only.
  (b) dataset-unique     -- canonical feature present in only some active
                            datasets; included with a companion presence
                            mask (1 if the source dataset computes it, 0
                            otherwise) and a fixed placeholder (0, applied
                            post-scaling) where absent.
  (c) divergent          -- same name across datasets but not measured the
                            same way; excluded from the common bucket by
                            default (pass via `divergent_features`).

Hard requirement enforced structurally here (not just by convention): you
cannot call `Harmonizer.fit()` on an arbitrary DataFrame. It only accepts
`TrainSplit`-wrapped frames, so fitting on anything but a designated
training split is a type error, not a silent bug.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .registry import IDENTITY_LIKE_HINTS, DatasetSpec, get_spec

PLACEHOLDER_VALUE = 0.0


@dataclass(frozen=True)
class TrainSplit:
    """Marks a per-dataset DataFrame as the designated training split.

    Wrap a frame in this before passing it to `Harmonizer.fit()`. There is no
    other way to fit a scaler in this codebase -- this is the mechanism that
    turns "never fit on non-train data" from a convention into something
    that fails loudly if violated.
    """

    dataset_name: str
    frame: pd.DataFrame


class DatasetIdentityLeakageError(RuntimeError):
    pass


def _assert_no_identity_columns(columns: list[str], context: str) -> None:
    for col in columns:
        lowered = col.lower()
        if any(hint in lowered for hint in IDENTITY_LIKE_HINTS):
            raise DatasetIdentityLeakageError(
                f"Identity-like column '{col}' reached {context}. Dataset-identity / "
                "endpoint-identity columns must never reach the harmonized tensor."
            )


class Harmonizer:
    def __init__(
        self,
        active_datasets: list[str],
        divergent_features: frozenset[str] = frozenset(),
        specs: dict[str, DatasetSpec] | None = None,
    ) -> None:
        if len(active_datasets) == 0:
            raise ValueError("active_datasets must be non-empty")
        self.active_datasets = list(active_datasets)
        self.specs: dict[str, DatasetSpec] = specs or {name: get_spec(name) for name in active_datasets}

        alias_sets = {name: set(spec.feature_alias) for name, spec in self.specs.items()}
        common = set.intersection(*alias_sets.values()) - set(divergent_features)
        all_features = set.union(*alias_sets.values())
        unique = sorted(all_features - common)

        self.common_features: list[str] = sorted(common)
        self.unique_features: list[str] = unique
        self.divergent_features = set(divergent_features)

        # presence_mask[dataset][j] == 1 iff unique_features[j] is aliased for that dataset
        self.presence_mask: dict[str, np.ndarray] = {
            name: np.array(
                [1.0 if feat in self.specs[name].feature_alias else 0.0 for feat in self.unique_features],
                dtype=np.float32,
            )
            for name in active_datasets
        }

        # populated by fit(): per-dataset {"common": (scaler, median), "unique": (scaler, median)}
        self._fit_state: dict[str, dict] = {}

    @property
    def output_width(self) -> int:
        return len(self.common_features) + 2 * len(self.unique_features)

    @property
    def output_columns(self) -> list[str]:
        return (
            [f"common__{c}" for c in self.common_features]
            + [f"unique__{c}" for c in self.unique_features]
            + [f"presence__{c}" for c in self.unique_features]
        )

    def _raw_to_canonical(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        spec = self.specs[dataset_name]
        out: dict[str, pd.Series] = {}
        for canon, raw in spec.feature_alias.items():
            if raw not in df.columns:
                continue
            # Network-flow datasets can legitimately contain divisions by
            # zero (for example, bytes/second for a zero-duration flow), and
            # some CSV exporters encode overflows as Infinity. Scikit-learn
            # scalers reject those values. Treat every non-finite value like
            # any other missing numeric observation so it is imputed using
            # statistics learned from this dataset's training split.
            numeric = pd.to_numeric(df[raw], errors="coerce").astype(np.float64)
            out[canon] = numeric.where(np.isfinite(numeric), np.nan)
        canon_df = pd.DataFrame(out, index=df.index)
        _assert_no_identity_columns(list(spec.feature_alias.values()), context="raw_to_canonical alias map")
        return canon_df

    def fit(self, train_splits: list[TrainSplit]) -> None:
        for split in train_splits:
            if not isinstance(split, TrainSplit):
                raise TypeError(
                    "Harmonizer.fit() only accepts TrainSplit-wrapped frames -- "
                    f"got {type(split)}. This is intentional: scalers must never be "
                    "fit on anything but the designated training split."
                )
            name = split.dataset_name
            if name not in self.specs:
                raise KeyError(f"'{name}' is not one of this Harmonizer's active_datasets: {self.active_datasets}")

            canon_df = self._raw_to_canonical(split.frame, name)

            common_vals = canon_df.reindex(columns=self.common_features)
            common_median = common_vals.median(numeric_only=True).fillna(0.0)
            common_filled = common_vals.fillna(common_median)
            common_scaler = StandardScaler().fit(common_filled.values)

            unique_cols = [c for c in self.unique_features if c in canon_df.columns]
            unique_state = None
            if unique_cols:
                unique_vals = canon_df[unique_cols]
                unique_median = unique_vals.median(numeric_only=True).fillna(0.0)
                unique_filled = unique_vals.fillna(unique_median)
                unique_scaler = StandardScaler().fit(unique_filled.values)
                unique_state = (unique_cols, unique_scaler, unique_median)

            self._fit_state[name] = {"common": (common_scaler, common_median), "unique": unique_state}

    def transform(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        """Returns a (n_rows, output_width) float32 array: [common | unique | presence_mask]."""
        if dataset_name not in self._fit_state:
            raise RuntimeError(
                f"No fitted scaler for dataset '{dataset_name}'. Call Harmonizer.fit() with a "
                "TrainSplit for this dataset before transforming any of its rows (train, val, or test)."
            )
        canon_df = self._raw_to_canonical(df, dataset_name)
        n = len(df)

        common_scaler, common_median = self._fit_state[dataset_name]["common"]
        common_vals = canon_df.reindex(columns=self.common_features)
        common_filled = common_vals.fillna(common_median)
        common_scaled = (
            common_scaler.transform(common_filled.values) if self.common_features else np.zeros((n, 0), dtype=np.float32)
        )

        unique_scaled = np.zeros((n, len(self.unique_features)), dtype=np.float32)
        unique_state = self._fit_state[dataset_name]["unique"]
        if unique_state is not None:
            unique_cols, unique_scaler, unique_median = unique_state
            vals = canon_df[unique_cols].fillna(unique_median)
            scaled = unique_scaler.transform(vals.values)
            idx = [self.unique_features.index(c) for c in unique_cols]
            unique_scaled[:, idx] = scaled
        # placeholder for dataset-absent unique features is exactly 0 post-scaling, already satisfied
        # by initializing unique_scaled with zeros and only overwriting present columns.

        presence = np.tile(self.presence_mask[dataset_name], (n, 1))

        harmonized = np.concatenate(
            [common_scaled.astype(np.float32), unique_scaled.astype(np.float32), presence.astype(np.float32)],
            axis=1,
        )
        _assert_no_identity_columns(self.output_columns, context="harmonized output tensor")
        assert harmonized.shape == (n, self.output_width), (
            f"harmonized width mismatch: got {harmonized.shape[1]}, expected {self.output_width}"
        )
        return harmonized

    def fitted_datasets(self) -> set[str]:
        return set(self._fit_state)
