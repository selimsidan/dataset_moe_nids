"""Chunked CSV streaming, folder-based label derivation, label harmonization,
and stratified per-class splitting.

Split-before-fit is a hard pipeline ordering in this codebase: `load_dataset`
only ever returns raw rows + harmonized labels, `stratified_split` partitions
those rows into train/val/test, and only `TrainSplit`-wrapped train rows
(harmonization.py) may be used to fit a scaler. There is no code path that
lets you fit on val/test data.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .registry import DatasetSpec

DEFAULT_CHUNKSIZE = 200_000


def _clean_header(col: str) -> str:
    return col.strip()


def ciciot2023_label_from_filename(path: str) -> str:
    """Derive a CICIoT2023 label from its class directory or flat filename.

    The project's real release is nested as
    ``CSV/<class-name>/<capture-number>.pcap.csv``; some distributions are
    flat and use the class as the filename. Prefer the parent directory when
    it is below ``CSV`` so numbered captures do not become fake classes.
    """
    parent = os.path.basename(os.path.dirname(path))
    if parent.lower() != "csv":
        raw_label = parent
    else:
        raw_label = os.path.splitext(os.path.basename(path))[0]
    if raw_label.lower().startswith("benign"):
        return "Benign"
    return raw_label


_FOLDER_LABEL_FNS = {
    "ciciot2023_label_from_filename": ciciot2023_label_from_filename,
}


EXCLUDE_LABEL = "__EXCLUDE__"


def harmonize_label(
    spec: DatasetSpec, raw_value: str, mapping_override: dict[str, str | None] | None = None
) -> str:
    """Maps one dataset's raw label string to a canonical class name.

    Lookup order: (1) `mapping_override` (from `data.label_mapping.<dataset>`
    in config -- the primary, easily-edited place this should live), (2)
    `spec.label_alias` (a low-priority default baked into the registry), (3)
    exact/case-insensitive match against `spec.benign_label` -> `"Benign"`.
    A mapped value of `None` means "exclude this row" (returns the
    `EXCLUDE_LABEL` sentinel, which `load_dataset` filters out).

    There is deliberately NO passthrough fallback: an unmapped raw value is
    returned as-is here (so `load_dataset`'s strict check can find and
    report it), but callers should treat "unmapped" as a bug to fix in
    config, not a class to silently start training on -- see
    `load_dataset(strict=True)`.
    """
    if pd.isna(raw_value):
        cleaned = ""
    else:
        cleaned = str(raw_value).strip()

    override = mapping_override or {}
    if cleaned in override:
        mapped = override[cleaned]
        return EXCLUDE_LABEL if mapped is None else mapped

    if cleaned.lower() == spec.benign_label.lower():
        return "Benign"
    if cleaned in spec.label_alias:
        mapped = spec.label_alias[cleaned]
        return EXCLUDE_LABEL if mapped is None else mapped
    if cleaned.lower() in spec.label_alias:
        mapped = spec.label_alias[cleaned.lower()]
        return EXCLUDE_LABEL if mapped is None else mapped

    return cleaned if cleaned else "Benign"


def is_explicitly_mapped(spec: DatasetSpec, raw_value: str, mapping_override: dict[str, str | None] | None = None) -> bool:
    """True iff `raw_value` resolves via an explicit mapping entry (config
    override, registry default, or a benign-label match) -- i.e. NOT the
    passthrough fallback in `harmonize_label`. Used by `load_dataset`'s
    strict-mode check to find raw labels nobody has told this codebase how
    to canonicalize yet.
    """
    if pd.isna(raw_value):
        cleaned = ""
    else:
        cleaned = str(raw_value).strip()
    override = mapping_override or {}
    if cleaned in override:
        return True
    if cleaned.lower() == spec.benign_label.lower():
        return True
    if cleaned in spec.label_alias or cleaned.lower() in spec.label_alias:
        return True
    return False


def _iter_file_chunks(path: str, chunksize: int):
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = [_clean_header(c) for c in chunk.columns]
        yield chunk


def _check_unmapped(spec: DatasetSpec, raw_values: pd.Series, mapping_override: dict[str, str | None] | None) -> dict[str, int]:
    counts = raw_values.value_counts()
    return {
        str(raw): int(n)
        for raw, n in counts.items()
        if not is_explicitly_mapped(spec, raw, mapping_override)
    }


def load_dataset(
    spec: DatasetSpec,
    chunksize: int = DEFAULT_CHUNKSIZE,
    mapping_override: dict[str, str | None] | None = None,
    strict: bool = True,
    max_rows: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Loads one dataset's raw rows plus a harmonized 'canonical_label' column.

    `mapping_override` is `data.label_mapping.<dataset>` from config -- see
    `harmonize_label`. If `strict=True` (default), raises `ValueError` when
    any raw label value in the data has no explicit mapping (config
    override, registry default, or benign match), listing the exact values
    and counts so you can add them to config rather than silently starting
    to train on an unintended new class. Rows explicitly mapped to `None`
    (excluded) are dropped from the returned frame; the drop count is logged.

    With `max_rows=None`, reads via chunked streaming and returns one
    concatenated frame. With a positive `max_rows`, maintains a bounded
    per-class random-key reservoir during ingestion, tracks true class counts
    across the full source, and allocates the final sample proportionally
    while preserving every class. This is the smoke-run path: a multi-GB CSV
    is fully validated but never fully retained in RAM.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer or None")

    frames: list[pd.DataFrame] = []
    unmapped: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    sample_key = "__moe_nids_random_sample_key__"
    reservoirs: dict[str, pd.DataFrame] = {}
    class_counts: dict[str, int] = {}

    def _tag_by_column(chunk: pd.DataFrame) -> None:
        raw_col = chunk[spec.label_col]
        if strict:
            for raw, n in _check_unmapped(spec, raw_col, mapping_override).items():
                unmapped[raw] = unmapped.get(raw, 0) + n
        chunk["canonical_label"] = [harmonize_label(spec, v, mapping_override) for v in raw_col]

    def _retain(chunk: pd.DataFrame) -> None:
        if max_rows is None:
            frames.append(chunk)
            return

        valid = chunk[chunk["canonical_label"] != EXCLUDE_LABEL]
        for label, group in valid.groupby("canonical_label", sort=False):
            label = str(label)
            class_counts[label] = class_counts.get(label, 0) + len(group)
            candidates = group.copy()
            if sample_key in candidates.columns:
                raise ValueError(f"Input dataset unexpectedly contains reserved column '{sample_key}'")
            candidates[sample_key] = rng.random(len(candidates))
            if label in reservoirs:
                candidates = pd.concat([reservoirs[label], candidates], ignore_index=True)
            if len(candidates) > max_rows:
                candidates = candidates.nlargest(max_rows, sample_key)
            reservoirs[label] = candidates

    if spec.kind == "file":
        candidates = [p for p in spec.paths if os.path.isfile(p)]
        if not candidates:
            raise FileNotFoundError(f"No file found for dataset '{spec.name}' among candidates: {spec.paths}")
        for chunk in _iter_file_chunks(candidates[0], chunksize):
            _tag_by_column(chunk)
            _retain(chunk)

    elif spec.kind == "files":
        found_any = False
        for path in spec.paths:
            if not os.path.isfile(path):
                continue
            found_any = True
            for chunk in _iter_file_chunks(path, chunksize):
                _tag_by_column(chunk)
                _retain(chunk)
        if not found_any:
            raise FileNotFoundError(f"No files found for dataset '{spec.name}' among candidates: {spec.paths}")

    elif spec.kind == "directory":
        directory = next((p for p in spec.paths if os.path.isdir(p)), None)
        if directory is None:
            raise FileNotFoundError(f"No directory found for dataset '{spec.name}' among candidates: {spec.paths}")
        csv_files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under directory: {directory}")
        label_fn = _FOLDER_LABEL_FNS[spec.folder_label_fn] if spec.folder_label_fn else None
        for path in csv_files:
            file_label = label_fn(path) if label_fn else None
            for chunk in _iter_file_chunks(path, chunksize):
                if file_label is not None:
                    # The folder/filename-derived string IS the raw label for
                    # this file -- it still goes through the same mapping/
                    # strict-check machinery as a column value would.
                    if strict and not is_explicitly_mapped(spec, file_label, mapping_override):
                        unmapped[file_label] = unmapped.get(file_label, 0) + len(chunk)
                    chunk["canonical_label"] = harmonize_label(spec, file_label, mapping_override)
                else:
                    _tag_by_column(chunk)
                _retain(chunk)
    else:
        raise ValueError(f"Unknown DatasetSpec.kind '{spec.kind}' for '{spec.name}'")

    if strict and unmapped:
        details = ", ".join(f"'{raw}' ({n} rows)" for raw, n in sorted(unmapped.items(), key=lambda kv: -kv[1]))
        raise ValueError(
            f"Dataset '{spec.name}' has raw label value(s) with no explicit mapping: {details}. "
            f"Add each to config `data.label_mapping.{spec.name}` (map to a canonical class name, "
            f"or to `null` to exclude those rows), or set `data.strict_label_mapping: false` to "
            f"allow passthrough (not recommended -- see data/loaders.py::harmonize_label)."
        )

    if max_rows is not None:
        if not reservoirs:
            raise ValueError(f"Dataset '{spec.name}' has no rows after label mapping/exclusion")
        quotas = _proportional_class_quotas(class_counts, max_rows)
        sampled = [
            reservoirs[label].nlargest(quota, sample_key)
            for label, quota in quotas.items()
            if quota > 0
        ]
        df = (
            pd.concat(sampled, ignore_index=True)
            .drop(columns=[sample_key])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        print(
            f"[load_dataset:{spec.name}] retained {len(df):,} stratified row(s) "
            f"from {sum(class_counts.values()):,} mapped row(s)"
        )
        return df

    if not frames:
        raise ValueError(f"Dataset '{spec.name}' has no readable rows")
    df = pd.concat(frames, ignore_index=True)
    excluded = int((df["canonical_label"] == EXCLUDE_LABEL).sum())
    if excluded:
        print(f"[load_dataset:{spec.name}] dropping {excluded} row(s) explicitly excluded via label_mapping")
        df = df[df["canonical_label"] != EXCLUDE_LABEL].reset_index(drop=True)
    return df


def _proportional_class_quotas(class_counts: dict[str, int], max_rows: int) -> dict[str, int]:
    """Allocate at most `max_rows` slots proportionally, with one slot for
    every class when the budget permits. Largest remainders receive leftover
    slots, and no class receives more rows than it actually contains.
    """
    total = sum(class_counts.values())
    if total <= max_rows:
        return dict(class_counts)

    labels = sorted(class_counts)
    if max_rows < len(labels):
        # This only arises with an unusually tiny caller-provided cap. Keep a
        # deterministic subset instead of silently exceeding the requested cap.
        return {label: int(i < max_rows) for i, label in enumerate(labels)}

    exact = {label: max_rows * class_counts[label] / total for label in labels}
    quotas = {
        label: min(class_counts[label], max(1, int(np.floor(exact[label]))))
        for label in labels
    }

    while sum(quotas.values()) > max_rows:
        candidates = [label for label in labels if quotas[label] > 1]
        label = min(candidates, key=lambda item: (exact[item] - np.floor(exact[item]), -quotas[item], item))
        quotas[label] -= 1

    while sum(quotas.values()) < max_rows:
        candidates = [label for label in labels if quotas[label] < class_counts[label]]
        if not candidates:
            break
        label = max(candidates, key=lambda item: (exact[item] - np.floor(exact[item]), class_counts[item], item))
        quotas[label] += 1
        # Once a class receives its largest-remainder slot, lower its priority
        # for any additional pass through the loop.
        exact[label] = float(np.floor(exact[label]))

    return quotas


def cap_dataset_rows(
    df: pd.DataFrame, max_rows: int | None, label_col: str = "canonical_label", seed: int = 0
) -> pd.DataFrame:
    """Stratified subsample down to (approximately) `max_rows` total rows,
    preserving each class's relative share (every present class keeps at
    least 1 row) -- an opt-in knob for fast local iteration on a laptop
    before a full run on the real data. `max_rows=None` (default) is a
    no-op, so behavior is unchanged unless a run explicitly sets
    `data.max_rows_per_dataset`.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [group.sample(n=max(1, min(len(group), round(len(group) * frac))), random_state=seed) for _, group in df.groupby(label_col)]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "canonical_label",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    min_class_count: int = 3,
) -> Split:
    """Stratified per-class train/val/test split, executed BEFORE any
    scaler/encoder fitting -- callers must not fit anything on `df` itself,
    only on `.train` via a `TrainSplit` wrapper.

    Classes with fewer than `min_class_count` rows go entirely to train
    (too few samples to stratify safely); this is logged, not silent.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_class_count].index.tolist()
    if rare_classes:
        print(
            f"[stratified_split] {len(rare_classes)} class(es) below min_class_count={min_class_count} "
            f"routed entirely to train: {rare_classes}"
        )

    rare_mask = df[label_col].isin(rare_classes)
    rare_df = df[rare_mask]
    usable_df = df[~rare_mask]

    train_val, test = train_test_split(
        usable_df, test_size=test_frac, stratify=usable_df[label_col], random_state=seed
    )
    val_relative = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val[label_col], random_state=seed
    )
    train = pd.concat([train, rare_df], ignore_index=True) if len(rare_df) else train
    return Split(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def stratified_sample_indices(
    labels: np.ndarray, per_class: int, seed: int = 0, replace_if_short: bool = True
) -> np.ndarray:
    """Returns row indices with (approximately) `per_class` samples per
    distinct label -- the building block for class-balanced batches/splits.
    Classes with fewer than `per_class` rows are sampled with replacement
    (if `replace_if_short`) so every class still contributes a full slice.
    """
    rng = np.random.default_rng(seed)
    out = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if len(idx) >= per_class:
            chosen = rng.choice(idx, size=per_class, replace=False)
        elif replace_if_short:
            chosen = rng.choice(idx, size=per_class, replace=True)
        else:
            chosen = idx
        out.append(chosen)
    return np.concatenate(out)
