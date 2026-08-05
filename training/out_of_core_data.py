"""Disk-backed full-data assembly for configurable NF-v3 MoE combinations."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from data import paths
from data.out_of_core import (
    ConcatenatedRows,
    PreparedDataset,
    RemappedLabels,
    fit_preprocessor_from_arrays,
    prepare_dataset,
    stage_dataset,
    standardized_batch,
)
from data.registry import get_spec

from .dataset import BENIGN_LABEL, PreparedData


class StandardizedRows:
    """Array-like standardized view; only requested rows are materialized."""

    def __init__(self, rows, stats: dict[str, np.ndarray]):
        self.rows = rows
        self.shape = rows.shape
        self.dtype = np.dtype(np.float32)
        self.stats = stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return standardized_batch(
            self.rows[index], self.stats["finite_medians"],
            self.stats["mean"], self.stats["scale"],
        )


class ConstantRows:
    def __init__(self, value: int, length: int, dtype=np.int16):
        self.value = value
        self.shape = (int(length),)
        self.dtype = np.dtype(dtype)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if index < -len(self) or index >= len(self):
                raise IndexError(index)
            return self.dtype.type(self.value)
        if isinstance(index, slice):
            length = len(range(*index.indices(len(self))))
            return np.full(length, self.value, dtype=self.dtype)
        indices = np.asarray(index)
        return np.full(indices.shape, self.value, dtype=self.dtype)


@dataclass
class OutOfCoreSplit:
    features: StandardizedRows
    class_idx: RemappedLabels
    dataset_idx: ConcatenatedRows
    dataset_name: None
    dataset_slices: dict[str, slice]

    def __len__(self):
        return len(self.class_idx)


@dataclass
class OutOfCoreContext:
    data: PreparedData
    prepared_by_dataset: dict[str, PreparedDataset]
    split_signatures: dict[str, str]
    preprocessing_signature: str
    preprocessing_path: str
    feature_columns: list[str]


def _slices(parts: list[np.ndarray], names: list[str]) -> dict[str, slice]:
    result = {}
    start = 0
    for name, part in zip(names, parts):
        result[name] = slice(start, start + len(part))
        start += len(part)
    return result


def _build_split(
    split: str,
    names: list[str],
    prepared: dict[str, PreparedDataset],
    remaps: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
) -> OutOfCoreSplit:
    feature_parts = [prepared[name].features(split) for name in names]
    label_parts = [prepared[name].labels(split) for name in names]
    raw_rows = ConcatenatedRows(feature_parts)
    labels = RemappedLabels(label_parts, [remaps[name] for name in names])
    dataset_ids = ConcatenatedRows([
        ConstantRows(index, len(part)) for index, part in enumerate(feature_parts)
    ])
    return OutOfCoreSplit(
        features=StandardizedRows(raw_rows, stats),
        class_idx=labels,
        dataset_idx=dataset_ids,
        dataset_name=None,
        dataset_slices=_slices(feature_parts, names),
    )


def prepare_out_of_core_data(config: dict) -> OutOfCoreContext:
    data_cfg = config["data"]
    names = list(data_cfg["active_datasets"])
    if not 2 <= len(names) <= 4:
        raise ValueError("out_of_core_full expects a 2-way, 3-way, or 4-way dataset combination")

    specs = {name: get_spec(name) for name in names}
    first_alias = specs[names[0]].feature_alias
    for name, spec in specs.items():
        if spec.kind != "file" or spec.feature_alias != first_alias:
            raise ValueError(
                "out_of_core_full currently requires schema-compatible, single-file NF-v3 datasets; "
                f"{name!r} is not compatible"
            )
    feature_columns = list(first_alias.values())
    if len(feature_columns) != 47:
        raise ValueError(f"Expected the confirmed 47-feature NF-v3 schema, found {len(feature_columns)}")

    output_root = paths.ensure_output_dir()
    cache_root = os.path.join(output_root, "out_of_core_splits")
    scratch_root = os.path.join(paths.SCRATCH_DIR, "out_of_core_splits")
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)
    chunk_rows = int(data_cfg.get("out_of_core_chunk_rows", data_cfg.get("chunksize", 200_000)))
    split_seed = int(data_cfg.get("split_seed", config.get("seed", 0)))
    mapping_catalog = data_cfg.get("label_mapping", {})

    prepared_by_dataset: dict[str, PreparedDataset] = {}
    for name in names:
        print(f"[ooc-data] preparing or reusing {name}", flush=True)
        persistent = prepare_dataset(
            specs[name], mapping_catalog[name], feature_columns, data_cfg["split"],
            split_seed, cache_root, chunk_rows, scratch_root,
        )
        if data_cfg.get("stage_out_of_core_to_local", True):
            print(f"[ooc-data] staging {name} on local Colab disk", flush=True)
            prepared_by_dataset[name] = stage_dataset(persistent, scratch_root)
        else:
            prepared_by_dataset[name] = persistent

    split_signatures = {name: prepared_by_dataset[name].signature for name in names}
    preprocessing_payload = {
        "split_signatures": split_signatures,
        "feature_columns": feature_columns,
        "active_datasets": names,
        "preprocessing": "pooled-train-standard-scaler-v1",
    }
    preprocessing_signature = hashlib.sha256(
        json.dumps(preprocessing_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    preprocessing_dir = os.path.join(output_root, "out_of_core_preprocessors")
    os.makedirs(preprocessing_dir, exist_ok=True)
    preprocessing_path = os.path.join(preprocessing_dir, f"pooled-{preprocessing_signature}.npz")
    if os.path.isfile(preprocessing_path):
        with np.load(preprocessing_path) as saved:
            stats = {key: saved[key].copy() for key in saved.files}
    else:
        stats = fit_preprocessor_from_arrays(
            [prepared_by_dataset[name].features("train") for name in names], chunk_rows
        )
        temporary = preprocessing_path + ".tmp.npz"
        np.savez(temporary, **stats)
        os.replace(temporary, preprocessing_path)

    observed_classes = {
        label for prepared in prepared_by_dataset.values() for label in prepared.class_names
    }
    allowed_classes = {BENIGN_LABEL, *data_cfg["active_classes"]}
    unknown_observed = sorted(observed_classes - allowed_classes)
    if unknown_observed:
        raise ValueError(
            f"Mapped classes are absent from [Benign, *data.active_classes]: {unknown_observed}"
        )
    # Match the successful diagnostic combination path: the classifier's
    # task space is the union actually represented by the selected datasets,
    # not repository-global classes that can never be a training target.
    class_names = [BENIGN_LABEL, *sorted(observed_classes - {BENIGN_LABEL})]
    class_to_idx = {label: index for index, label in enumerate(class_names)}
    remaps = {}
    for name in names:
        remaps[name] = np.asarray(
            [class_to_idx[label] for label in prepared_by_dataset[name].class_names], dtype=np.int16
        )

    data = PreparedData(
        harmonizer=None,
        class_names=class_names,
        active_datasets=names,
        train=_build_split("train", names, prepared_by_dataset, remaps, stats),
        val=_build_split("val", names, prepared_by_dataset, remaps, stats),
        test=_build_split("test", names, prepared_by_dataset, remaps, stats),
    )
    return OutOfCoreContext(
        data=data,
        prepared_by_dataset=prepared_by_dataset,
        split_signatures=split_signatures,
        preprocessing_signature=preprocessing_signature,
        preprocessing_path=preprocessing_path,
        feature_columns=feature_columns,
    )


def class_counts(labels, num_classes: int, chunk_rows: int = 1_000_000) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for start in range(0, len(labels), chunk_rows):
        values = np.asarray(labels[start : start + chunk_rows], dtype=np.int64)
        counts += np.bincount(values, minlength=num_classes)
    return counts
