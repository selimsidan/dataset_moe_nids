"""Prints raw label value counts for a registered dataset, without any
harmonization/mapping applied -- the tool for building or extending
`data.label_mapping.<dataset>` in config/default.yaml from real data,
instead of guessing at raw category strings.

    python -m data.inspect_labels NF-BoT-IoT-v3
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter

from .loaders import DEFAULT_CHUNKSIZE, _FOLDER_LABEL_FNS, _iter_file_chunks
from .registry import get_spec


def raw_label_value_counts(name: str, chunksize: int = DEFAULT_CHUNKSIZE) -> Counter:
    spec = get_spec(name)
    counts: Counter = Counter()

    if spec.kind in ("file", "files"):
        paths = [p for p in spec.paths if os.path.isfile(p)]
        if not paths:
            raise FileNotFoundError(f"No file(s) found for dataset '{name}' among candidates: {spec.paths}")
        for path in paths:
            for chunk in _iter_file_chunks(path, chunksize):
                counts.update(str(v).strip() if not _is_na(v) else "<NA>" for v in chunk[spec.label_col])

    elif spec.kind == "directory":
        directory = next((p for p in spec.paths if os.path.isdir(p)), None)
        if directory is None:
            raise FileNotFoundError(f"No directory found for dataset '{name}' among candidates: {spec.paths}")
        csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under directory: {directory}")
        label_fn = _FOLDER_LABEL_FNS[spec.folder_label_fn] if spec.folder_label_fn else None
        for path in csv_files:
            if label_fn is not None:
                n_rows = sum(len(chunk) for chunk in _iter_file_chunks(path, chunksize))
                counts[label_fn(path)] += n_rows
            else:
                for chunk in _iter_file_chunks(path, chunksize):
                    counts.update(str(v).strip() for v in chunk[spec.label_col])
    else:
        raise ValueError(f"Unknown DatasetSpec.kind '{spec.kind}' for '{name}'")

    return counts


def _is_na(value) -> bool:
    try:
        return value != value  # NaN != NaN
    except Exception:
        return False


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m data.inspect_labels <dataset_name>")
        sys.exit(1)
    name = sys.argv[1]
    counts = raw_label_value_counts(name)
    print(f"Raw label value counts for '{name}':")
    for raw, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {raw!r:50s} {n:>12d}")


if __name__ == "__main__":
    main()
