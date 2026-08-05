"""Deterministic, full-data, disk-backed preparation for NF-v3 experiments.

The ordinary pandas path is convenient for sampled/debug runs but creates
several full-size copies while splitting and preprocessing.  This module uses
two streaming CSV passes and ``.npy`` memory maps instead:

1. count and strictly validate harmonized labels;
2. assign every mapped row exactly once to deterministic stratified
   train/validation/test splits and write float32 features directly to disk;
3. fit invalid-value replacements and scaling statistics from training rows
   only, in bounded-memory chunks.

The resulting signed artifact is reusable by separate and pooled runs.  A
persisted artifact, rather than merely a seed, is the strongest guarantee that
all conditions evaluate the exact same test rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .loaders import EXCLUDE_LABEL
from .registry import DatasetSpec

FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


def _json_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _source_identity(path: str) -> dict[str, int | str]:
    stat = os.stat(path)
    return {
        # Do not sign the mount prefix: the same Drive file appears below
        # /content/drive in Colab and below Library/CloudStorage locally.
        "filename": os.path.basename(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_path(spec: DatasetSpec) -> str:
    """Resolve the one physical CSV used by the full-data NF-v3 path.

    The general in-memory loader continues to support multi-file and
    directory datasets.  This out-of-core implementation is intentionally
    strict: all datasets in one MoE combination must be single-file NF-v3
    datasets with the same confirmed schema.
    """
    if spec.kind != "file":
        raise ValueError(
            f"out_of_core_full currently supports single-file datasets; "
            f"{spec.name!r} has kind={spec.kind!r}"
        )
    candidates = [path for path in spec.paths if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(f"No file found for dataset {spec.name!r}: {spec.paths}")
    return candidates[0]


def split_contract(
    spec: DatasetSpec,
    mapping: dict[str, str | None],
    feature_columns: list[str],
    split_cfg: dict,
    seed: int,
) -> dict:
    """Return everything that defines row membership and feature semantics."""
    return {
        "format_version": FORMAT_VERSION,
        "dataset": spec.name,
        "source": _source_identity(_source_path(spec)),
        "label_column": spec.label_col,
        "benign_label": spec.benign_label,
        "label_harmonization": mapping,
        "feature_columns": list(feature_columns),
        "split": {
            "train_frac": float(split_cfg["train_frac"]),
            "val_frac": float(split_cfg["val_frac"]),
            "test_frac": float(split_cfg["test_frac"]),
            "min_class_count": int(split_cfg["min_class_count"]),
        },
        "split_seed": int(seed),
        "assignment_algorithm": "streamed-classwise-multivariate-hypergeometric-v1",
    }


def split_signature(
    spec: DatasetSpec,
    mapping: dict[str, str | None],
    feature_columns: list[str],
    split_cfg: dict,
    seed: int,
) -> str:
    return _json_hash(split_contract(spec, mapping, feature_columns, split_cfg, seed))


@dataclass(frozen=True)
class PreparedDataset:
    dataset: str
    directory: str
    signature: str
    class_names: tuple[str, ...]
    feature_columns: tuple[str, ...]
    split_rows: dict[str, int]
    class_split_counts: dict[str, dict[str, int]]

    def features(self, split: str, mode: str = "r") -> np.memmap:
        _validate_split_name(split)
        return np.load(os.path.join(self.directory, f"{split}_X.npy"), mmap_mode=mode)

    def labels(self, split: str, mode: str = "r") -> np.memmap:
        _validate_split_name(split)
        return np.load(os.path.join(self.directory, f"{split}_y.npy"), mmap_mode=mode)

    def preprocessor(self) -> dict[str, np.ndarray]:
        with np.load(os.path.join(self.directory, "preprocessor.npz")) as values:
            return {name: values[name].copy() for name in values.files}


class ConcatenatedRows:
    """Array-like row concatenation without constructing a pooled array."""

    def __init__(self, arrays: list[np.ndarray]):
        if not arrays:
            raise ValueError("arrays cannot be empty")
        self.arrays = tuple(arrays)
        self.offsets = np.cumsum([0, *[len(array) for array in arrays]], dtype=np.int64)
        tail_shape = tuple(arrays[0].shape[1:])
        if any(tuple(array.shape[1:]) != tail_shape for array in arrays):
            raise ValueError("all arrays must have the same non-row shape")
        self.shape = (int(self.offsets[-1]), *tail_shape)
        self.dtype = np.result_type(*[array.dtype for array in arrays])

    def __len__(self) -> int:
        return self.shape[0]

    def _one(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        owner = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.arrays[owner][index - self.offsets[owner]]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._one(int(index))
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return self[np.arange(start, stop, step, dtype=np.int64)]
            if start >= stop:
                return np.empty((0, *self.shape[1:]), dtype=self.dtype)
            pieces = []
            for array_index, array in enumerate(self.arrays):
                local_start = max(start, int(self.offsets[array_index])) - int(self.offsets[array_index])
                local_stop = min(stop, int(self.offsets[array_index + 1])) - int(self.offsets[array_index])
                if local_start < local_stop:
                    pieces.append(np.asarray(array[local_start:local_stop]))
            return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        indices = np.asarray(index, dtype=np.int64)
        flat = indices.ravel().copy()
        flat[flat < 0] += len(self)
        if flat.size and (flat.min() < 0 or flat.max() >= len(self)):
            raise IndexError("concatenated row index out of bounds")
        output = np.empty((len(flat), *self.shape[1:]), dtype=self.dtype)
        owners = np.searchsorted(self.offsets, flat, side="right") - 1
        for owner in np.unique(owners):
            mask = owners == owner
            output[mask] = self.arrays[int(owner)][flat[mask] - self.offsets[int(owner)]]
        return output.reshape((*indices.shape, *self.shape[1:]))


class RemappedLabels:
    """Concatenate dataset-local encoded labels in one shared label space."""

    def __init__(self, labels: list[np.ndarray], remaps: list[np.ndarray]):
        if len(labels) != len(remaps):
            raise ValueError("labels and remaps must have the same length")
        mapped = [_MappedLabelArray(label, remap) for label, remap in zip(labels, remaps)]
        self._rows = ConcatenatedRows(mapped)
        self.shape = self._rows.shape
        self.dtype = np.dtype(np.int16)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def materialize(self, chunk_rows: int = 1_000_000) -> np.ndarray:
        output = np.empty(len(self), dtype=np.int16)
        for section in iter_slices(len(self), chunk_rows):
            output[section] = self[section]
        return output


class _MappedLabelArray:
    def __init__(self, labels: np.ndarray, remap: np.ndarray):
        self.labels = labels
        self.remap = np.asarray(remap, dtype=np.int16)
        self.shape = labels.shape
        self.dtype = np.dtype(np.int16)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.remap[np.asarray(self.labels[index], dtype=np.int64)]


def _validate_split_name(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")


def _load_prepared(directory: str) -> PreparedDataset:
    metadata_path = os.path.join(directory, "metadata.json")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    for split in SPLIT_NAMES:
        x = np.load(os.path.join(directory, f"{split}_X.npy"), mmap_mode="r")
        y = np.load(os.path.join(directory, f"{split}_y.npy"), mmap_mode="r")
        expected = int(metadata["split_rows"][split])
        if x.shape != (expected, len(metadata["feature_columns"])) or y.shape != (expected,):
            raise ValueError(f"corrupt out-of-core artifact for {metadata['dataset']} {split}")
    return PreparedDataset(
        dataset=metadata["dataset"],
        directory=directory,
        signature=metadata["signature"],
        class_names=tuple(metadata["class_names"]),
        feature_columns=tuple(metadata["feature_columns"]),
        split_rows={key: int(value) for key, value in metadata["split_rows"].items()},
        class_split_counts={
            label: {key: int(value) for key, value in counts.items()}
            for label, counts in metadata["class_split_counts"].items()
        },
    )


def _harmonize_series(
    raw: pd.Series,
    spec: DatasetSpec,
    mapping: dict[str, str | None],
) -> tuple[np.ndarray, dict[str, int]]:
    cleaned = raw.astype("string").fillna("").str.strip()
    benign = cleaned.str.lower().eq(spec.benign_label.lower())
    mapped = cleaned.isin(mapping)
    explicit = (benign | mapped).to_numpy(dtype=bool)
    unmapped = cleaned[~explicit].value_counts()
    canonical = cleaned.to_numpy(dtype=object)
    benign_mask = benign.to_numpy(dtype=bool, na_value=False)
    mapped_mask = mapped.to_numpy(dtype=bool, na_value=False)
    canonical[benign_mask] = "Benign"
    if mapped.any():
        resolved = cleaned[mapped].map(mapping).fillna(EXCLUDE_LABEL).to_numpy(dtype=object)
        canonical[mapped_mask] = resolved
    return canonical, {str(label): int(count) for label, count in unmapped.items()}


def _scan_class_counts(
    spec: DatasetSpec,
    mapping: dict[str, str | None],
    chunk_rows: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    unmapped: dict[str, int] = {}
    seen = 0
    for chunk in pd.read_csv(
        _source_path(spec),
        usecols=[spec.label_col],
        chunksize=chunk_rows,
        low_memory=False,
    ):
        canonical, bad = _harmonize_series(chunk[spec.label_col], spec, mapping)
        for label, count in bad.items():
            unmapped[label] = unmapped.get(label, 0) + count
        values, value_counts = np.unique(canonical[canonical != EXCLUDE_LABEL], return_counts=True)
        for label, count in zip(values, value_counts):
            counts[str(label)] = counts.get(str(label), 0) + int(count)
        seen += len(chunk)
        if seen % (chunk_rows * 10) == 0:
            print(f"[out-of-core:{spec.name}] label scan: {seen:,} raw row(s)")
    if unmapped:
        details = ", ".join(f"'{label}' ({count:,} rows)" for label, count in sorted(unmapped.items(), key=lambda item: -item[1]))
        raise ValueError(f"Dataset '{spec.name}' has unmapped Attack values: {details}")
    if not counts:
        raise ValueError(f"Dataset '{spec.name}' has no mapped rows")
    return counts


def _largest_remainder_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * total
    result = np.floor(raw).astype(np.int64)
    remainder = total - int(result.sum())
    if remainder:
        order = np.argsort(-(raw - result), kind="stable")
        result[order[:remainder]] += 1
    return result


def class_split_quotas(class_counts: dict[str, int], split_cfg: dict) -> dict[str, dict[str, int]]:
    fractions = np.array(
        [split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"]],
        dtype=np.float64,
    )
    if not np.isclose(fractions.sum(), 1.0) or np.any(fractions < 0):
        raise ValueError("split fractions must be non-negative and sum to 1")
    minimum = int(split_cfg["min_class_count"])
    quotas: dict[str, dict[str, int]] = {}
    for label, total in sorted(class_counts.items()):
        values = np.array([total, 0, 0]) if total < minimum else _largest_remainder_counts(total, fractions)
        quotas[label] = {name: int(values[i]) for i, name in enumerate(SPLIT_NAMES)}
    return quotas


def _feature_matrix(chunk: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    matrix = np.empty((len(chunk), len(feature_columns)), dtype=np.float32)
    invalid: dict[str, int] = {}
    float32_max = np.finfo(np.float32).max
    for column_index, column in enumerate(feature_columns):
        values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(values) & (np.abs(values) <= float32_max)
        count = int((~valid).sum())
        if count:
            invalid[column] = count
        matrix[:, column_index] = np.where(valid, values, np.nan).astype(np.float32, copy=False)
    return matrix, invalid


def _seed_for_class(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _seed_for_split_shuffle(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:slots:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _write_streamed_splits(
    directory: str,
    spec: DatasetSpec,
    mapping: dict[str, str | None],
    feature_columns: list[str],
    quotas: dict[str, dict[str, int]],
    chunk_rows: int,
    seed: int,
) -> dict[str, dict[str, int]]:
    """Stream rows from the source CSV into disk-backed split arrays.

    Source rows arrive grouped into large contiguous same-class runs (NIDS
    captures record one attack scenario at a time). Appending in encounter
    order would leave the persisted arrays effectively sorted by class, which
    silently breaks any consumer that shuffles training batches with a
    bounded-size window (an MLP's block-shuffle window is far smaller than a
    multi-million-row same-class run). Each row is instead written directly
    to a pre-shuffled destination slot so the arrays are globally randomized
    on disk, independent of downstream shuffling strategy.
    """
    split_rows = {name: sum(values[name] for values in quotas.values()) for name in SPLIT_NAMES}
    x_maps = {
        name: np.lib.format.open_memmap(
            os.path.join(directory, f"{name}_X.npy"),
            mode="w+", dtype=np.float32, shape=(split_rows[name], len(feature_columns)),
        )
        for name in SPLIT_NAMES
    }
    y_maps = {
        name: np.lib.format.open_memmap(
            os.path.join(directory, f"{name}_y.npy"),
            mode="w+", dtype=np.int16, shape=(split_rows[name],),
        )
        for name in SPLIT_NAMES
    }
    class_names = sorted(quotas)
    class_to_id = {label: index for index, label in enumerate(class_names)}
    remaining = {label: np.array([values[name] for name in SPLIT_NAMES], dtype=np.int64) for label, values in quotas.items()}
    generators = {label: np.random.default_rng(_seed_for_class(seed, label)) for label in class_names}
    invalid_by_split = {name: np.zeros(len(feature_columns), dtype=np.int64) for name in SPLIT_NAMES}

    # Pre-assign every (class, split) row a random destination slot so
    # written arrays are globally shuffled rather than class-contiguous.
    slots_by_class_split: dict[tuple[str, str], np.ndarray] = {}
    for name in SPLIT_NAMES:
        slot_permutation = np.random.default_rng(_seed_for_split_shuffle(seed, name)).permutation(split_rows[name])
        cursor = 0
        for label in class_names:
            amount = quotas[label][name]
            slots_by_class_split[(label, name)] = slot_permutation[cursor : cursor + amount]
            cursor += amount
    slot_cursor = {key: 0 for key in slots_by_class_split}
    seen = 0

    usecols = list(dict.fromkeys([*feature_columns, spec.label_col]))
    for chunk in pd.read_csv(_source_path(spec), usecols=usecols, chunksize=chunk_rows, low_memory=False):
        chunk.columns = [column.strip() for column in chunk.columns]
        canonical, bad = _harmonize_series(chunk[spec.label_col], spec, mapping)
        if bad:
            raise ValueError(f"mapping changed between out-of-core passes: {bad}")
        keep = canonical != EXCLUDE_LABEL
        if not np.any(keep):
            continue
        chunk = chunk.loc[keep]
        canonical = canonical[keep]
        matrix, _ = _feature_matrix(chunk, feature_columns)
        for label in np.unique(canonical):
            label = str(label)
            positions = np.flatnonzero(canonical == label)
            rng = generators[label]
            draw = rng.multivariate_hypergeometric(remaining[label], len(positions))
            shuffled = positions[rng.permutation(len(positions))]
            cursor = 0
            for split_index, amount in enumerate(draw):
                amount = int(amount)
                if not amount:
                    continue
                row_positions = shuffled[cursor : cursor + amount]
                cursor += amount
                name = SPLIT_NAMES[split_index]
                key = (label, name)
                start = slot_cursor[key]
                target_slots = slots_by_class_split[key][start : start + amount]
                slot_cursor[key] = start + amount
                invalid_by_split[name][:] += (~np.isfinite(matrix[row_positions])).sum(axis=0)
                x_maps[name][target_slots] = matrix[row_positions]
                y_maps[name][target_slots] = class_to_id[label]
            remaining[label] -= draw
        seen += len(chunk)
        if seen % (chunk_rows * 10) < chunk_rows:
            print(f"[out-of-core:{spec.name}] feature write: {seen:,} mapped row(s)")

    offsets_complete = all(
        slot_cursor[(label, name)] == quotas[label][name] for label in class_names for name in SPLIT_NAMES
    )
    if any(np.any(values) for values in remaining.values()) or not offsets_complete:
        raise RuntimeError(f"streamed split accounting failed: remaining={remaining}, slot_cursor={slot_cursor}")
    for array in [*x_maps.values(), *y_maps.values()]:
        array.flush()
    invalid_counts = {
        split: {feature_columns[index]: int(count) for index, count in enumerate(values) if count}
        for split, values in invalid_by_split.items()
    }
    totals = {
        column: sum(invalid_counts[split].get(column, 0) for split in SPLIT_NAMES)
        for column in feature_columns
    }
    totals = {column: count for column, count in totals.items() if count}
    if totals:
        details = ", ".join(f"{name}={count:,}" for name, count in totals.items())
        print(f"[out-of-core:{spec.name}] stored invalid values as NaN pending train-only replacement: {details}")
    return invalid_counts


def _fit_preprocessor(
    directory: str,
    feature_columns: list[str],
    train_invalid_counts: dict[str, int],
    chunk_rows: int,
) -> None:
    train = np.load(os.path.join(directory, "train_X.npy"), mmap_mode="r")
    medians = np.zeros(len(feature_columns), dtype=np.float64)
    invalid_columns = [feature_columns.index(name) for name in train_invalid_counts]
    for column in invalid_columns:
        medians[column] = float(np.nanmedian(train[:, column]))
        if not np.isfinite(medians[column]):
            raise ValueError(f"feature column {column} has no finite training values")
    del train

    scaler = StandardScaler()
    values = np.load(os.path.join(directory, "train_X.npy"), mmap_mode="r")
    for start in range(0, len(values), chunk_rows):
        batch = np.asarray(values[start : start + chunk_rows], dtype=np.float32).copy()
        invalid = ~np.isfinite(batch)
        if invalid.any():
            rows, columns = np.nonzero(invalid)
            batch[rows, columns] = medians[columns]
        scaler.partial_fit(batch)
    del values
    no_train_invalid = np.ones(len(feature_columns), dtype=bool)
    no_train_invalid[invalid_columns] = False
    # A missing value first encountered in validation/test is replaced with
    # the training mean. This is leakage-free and maps to zero after scaling.
    medians[no_train_invalid] = scaler.mean_[no_train_invalid]
    np.savez(
        os.path.join(directory, "preprocessor.npz"),
        finite_medians=medians,
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        variance=scaler.var_.astype(np.float64),
        samples_seen=np.asarray(scaler.n_samples_seen_),
    )


def prepare_dataset(
    spec: DatasetSpec,
    mapping: dict[str, str | None],
    feature_columns: list[str],
    split_cfg: dict,
    seed: int,
    cache_root: str,
    chunk_rows: int = 200_000,
    work_root: str | None = None,
) -> PreparedDataset:
    """Create or validate a persistent full-data split artifact.

    When ``work_root`` is supplied, random-slot writes and preprocessing are
    performed on fast local disk. The completed artifact is then copied to
    ``cache_root`` sequentially and retained locally for immediate staging.
    This is critical on Colab: row-random writes through Drive FUSE are both
    slow and prone to timeouts.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    contract = split_contract(spec, mapping, feature_columns, split_cfg, seed)
    signature = _json_hash(contract)
    final_directory = os.path.join(cache_root, spec.name, f"split-{signature}")
    metadata_path = os.path.join(final_directory, "metadata.json")
    if os.path.isfile(metadata_path):
        prepared = _load_prepared(final_directory)
        if prepared.signature != signature:
            raise ValueError(f"signature mismatch in {metadata_path}")
        print(f"[out-of-core:{spec.name}] reusing prepared split: {final_directory}")
        return prepared

    persistent_dataset_cache = os.path.dirname(final_directory)
    os.makedirs(persistent_dataset_cache, exist_ok=True)
    persisting_prefix = f".split-{signature}-persisting-"
    for name in os.listdir(persistent_dataset_cache):
        stale = os.path.join(persistent_dataset_cache, name)
        if name.startswith(persisting_prefix) and os.path.isdir(stale):
            print(f"[out-of-core:{spec.name}] removing incomplete prior persistence copy: {stale}")
            shutil.rmtree(stale)
    build_dataset_cache = (
        os.path.join(work_root, spec.name) if work_root is not None else persistent_dataset_cache
    )
    os.makedirs(build_dataset_cache, exist_ok=True)
    local_final_directory = os.path.join(build_dataset_cache, f"split-{signature}")
    building_prefix = f".split-{signature}-building-"
    for name in os.listdir(build_dataset_cache):
        stale = os.path.join(build_dataset_cache, name)
        if name.startswith(building_prefix) and os.path.isdir(stale):
            print(f"[out-of-core:{spec.name}] removing incomplete prior build: {stale}")
            shutil.rmtree(stale)
    temporary = tempfile.mkdtemp(prefix=building_prefix, dir=build_dataset_cache)
    try:
        counts = _scan_class_counts(spec, mapping, chunk_rows)
        quotas = class_split_quotas(counts, split_cfg)
        split_rows = {name: sum(values[name] for values in quotas.values()) for name in SPLIT_NAMES}
        estimated = sum(split_rows.values()) * (4 * len(feature_columns) + 2)
        free = shutil.disk_usage(temporary).free
        if estimated * 1.05 > free:
            raise OSError(
                f"insufficient disk for {spec.name}: need about {estimated / 2**30:.1f} GiB "
                f"plus overhead, have {free / 2**30:.1f} GiB"
            )
        invalid = _write_streamed_splits(
            temporary, spec, mapping, feature_columns, quotas, chunk_rows, seed
        )
        _fit_preprocessor(temporary, feature_columns, invalid["train"], chunk_rows)
        metadata = {
            "signature": signature,
            "contract": contract,
            "dataset": spec.name,
            "class_names": sorted(counts),
            "feature_columns": feature_columns,
            "class_counts": counts,
            "class_split_counts": quotas,
            "split_rows": split_rows,
            "invalid_value_counts_by_split": invalid,
        }
        with open(os.path.join(temporary, "metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        if work_root is None:
            os.replace(temporary, final_directory)
        else:
            required = sum(
                os.path.getsize(os.path.join(temporary, name))
                for name in os.listdir(temporary)
                if os.path.isfile(os.path.join(temporary, name))
            )
            persistent_free = shutil.disk_usage(persistent_dataset_cache).free
            if required * 1.02 > persistent_free:
                raise OSError(
                    f"insufficient persistent storage for {spec.name}: need {required / 2**30:.1f} GiB, "
                    f"have {persistent_free / 2**30:.1f} GiB"
                )
            persistent_temporary = tempfile.mkdtemp(
                prefix=persisting_prefix, dir=persistent_dataset_cache
            )
            try:
                for name in os.listdir(temporary):
                    source = os.path.join(temporary, name)
                    if os.path.isfile(source):
                        shutil.copy2(source, os.path.join(persistent_temporary, name))
                _load_prepared(persistent_temporary)
                os.replace(persistent_temporary, final_directory)
            except Exception:
                shutil.rmtree(persistent_temporary, ignore_errors=True)
                raise
            if os.path.isdir(local_final_directory):
                shutil.rmtree(local_final_directory)
            os.replace(temporary, local_final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[out-of-core:{spec.name}] prepared full split: {final_directory}")
    return _load_prepared(final_directory)


def stage_dataset(prepared: PreparedDataset, scratch_root: str) -> PreparedDataset:
    """Copy a persistent Drive artifact to fast local disk, with validation."""
    destination = os.path.join(scratch_root, prepared.dataset, f"split-{prepared.signature}")
    if os.path.isfile(os.path.join(destination, "metadata.json")):
        try:
            return _load_prepared(destination)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".staging-", dir=os.path.dirname(destination))
    try:
        required = sum(
            os.path.getsize(os.path.join(prepared.directory, name))
            for name in os.listdir(prepared.directory)
            if os.path.isfile(os.path.join(prepared.directory, name))
        )
        free = shutil.disk_usage(temporary).free
        if required * 1.02 > free:
            raise OSError(
                f"insufficient local scratch for {prepared.dataset}: need {required / 2**30:.1f} GiB, "
                f"have {free / 2**30:.1f} GiB; set STAGE_OOC_TO_LOCAL_DISK=False or free disk"
            )
        for name in os.listdir(prepared.directory):
            source = os.path.join(prepared.directory, name)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(temporary, name))
        _load_prepared(temporary)
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[out-of-core:{prepared.dataset}] staged split on local disk: {destination}")
    return _load_prepared(destination)


def iter_slices(length: int, chunk_rows: int):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    for start in range(0, length, chunk_rows):
        yield slice(start, min(start + chunk_rows, length))


def fit_preprocessor_from_arrays(
    arrays: list[np.ndarray],
    chunk_rows: int = 200_000,
) -> dict[str, np.ndarray]:
    """Fit exact pooled replacements/scaling using only supplied train rows."""
    if not arrays:
        raise ValueError("arrays cannot be empty")
    feature_count = arrays[0].shape[1]
    if any(array.ndim != 2 or array.shape[1] != feature_count for array in arrays):
        raise ValueError("all arrays must be 2-D with the same feature count")
    invalid_columns = np.zeros(feature_count, dtype=bool)
    for array in arrays:
        for section in iter_slices(len(array), chunk_rows):
            invalid_columns |= ~np.isfinite(array[section]).all(axis=0)
    medians = np.zeros(feature_count, dtype=np.float64)
    for column in np.flatnonzero(invalid_columns):
        finite_parts = []
        for array in arrays:
            values = np.asarray(array[:, column])
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_parts.append(finite)
        if not finite_parts:
            raise ValueError(f"feature column {column} has no finite training values")
        combined = finite_parts[0] if len(finite_parts) == 1 else np.concatenate(finite_parts)
        medians[column] = float(np.median(combined))
        del finite_parts, combined

    scaler = StandardScaler()
    for array in arrays:
        for section in iter_slices(len(array), chunk_rows):
            batch = np.asarray(array[section], dtype=np.float32).copy()
            invalid = ~np.isfinite(batch)
            if invalid.any():
                rows, columns = np.nonzero(invalid)
                batch[rows, columns] = medians[columns]
            scaler.partial_fit(batch)
    no_train_invalid = ~invalid_columns
    medians[no_train_invalid] = scaler.mean_[no_train_invalid]
    return {
        "finite_medians": medians,
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "variance": scaler.var_.astype(np.float64),
        "samples_seen": np.asarray(scaler.n_samples_seen_),
    }


def standardized_batch(
    values: np.ndarray,
    finite_medians: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Return one bounded-memory standardized float32 batch."""
    output = np.asarray(values, dtype=np.float32).copy()
    invalid = ~np.isfinite(output)
    if invalid.any():
        rows, columns = np.nonzero(invalid)
        output[rows, columns] = finite_medians.astype(np.float32, copy=False)[columns]
    output -= mean.astype(np.float32, copy=False)
    output /= scale.astype(np.float32, copy=False)
    return output
