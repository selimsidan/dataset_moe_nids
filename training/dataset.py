"""Assembles the harmonized, split, tensorized pool of data every training
stage reads from. Adapted from moe_nids/training/dataset.py: the class
vocabulary here (`class_names`) is the FIXED canonical taxonomy from config
(`[Benign, *active_classes]`), not a data-derived vocabulary -- every
dataset-expert predicts over this entire fixed space directly, there's no
per-expert relabeling step to reconcile against a separately-tracked
"combined" space the way moe_nids needs. `dataset_idx` (index into
`active_datasets`) is new here: it's the gate's structural target space and
the routing key for Stage B's independent per-dataset training and for the
`no_fusion`/`hard_two_stage` baselines -- always bookkeeping alongside the
feature tensor, never concatenated into it (data/harmonization.py already
guarantees no dataset-identity signal reaches the feature vector itself).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.harmonization import Harmonizer, TrainSplit
from data.loaders import Split, cap_dataset_rows, load_dataset, stratified_split
from data.registry import get_spec

BENIGN_LABEL = "Benign"


def assert_active_datasets_consistent(config: dict) -> None:
    """Fail-loud startup guard (mirrors moe_nids' strict_label_mapping
    philosophy): every dataset referenced anywhere in `data.label_mapping`
    must be a subset of `data.active_datasets`. Catches a stale mapping
    entry left over after narrowing `active_datasets` for a fast-iteration
    subset run.
    """
    active = set(config["data"]["active_datasets"])
    mapped = set(config["data"].get("label_mapping", {}) or {})
    stale = mapped - active
    if stale:
        raise ValueError(
            f"data.label_mapping references dataset(s) not in data.active_datasets: {sorted(stale)}. "
            "Either add them to active_datasets or remove their label_mapping entries."
        )


@dataclass
class PreparedSplit:
    features: np.ndarray  # (n, D) harmonized
    class_idx: np.ndarray  # (n,) int64, index into class_names (fixed [Benign, *active_classes] space)
    dataset_idx: np.ndarray  # (n,) int64, index into active_datasets -- bookkeeping/gate-aux target only
    dataset_name: np.ndarray  # (n,) str, metadata only -- never fed to the model


@dataclass
class PreparedData:
    harmonizer: Harmonizer
    class_names: list[str]  # [Benign, *active_classes], the FULL fixed task vocabulary every expert predicts over
    active_datasets: list[str]
    train: PreparedSplit
    val: PreparedSplit
    test: PreparedSplit


def _vectorize(
    frame_by_dataset: dict[str, "pandas.DataFrame"],
    harmonizer: Harmonizer,
    class_to_idx: dict[str, int],
    dataset_to_idx: dict[str, int],
) -> PreparedSplit:
    feats, class_idx, dataset_idx, names = [], [], [], []
    unmapped: dict[str, int] = {}
    for name, df in frame_by_dataset.items():
        if len(df) == 0:
            continue
        feats.append(harmonizer.transform(df, name))
        mapped = df["canonical_label"].map(class_to_idx)
        bad = df.loc[mapped.isna(), "canonical_label"].value_counts()
        for label, n in bad.items():
            unmapped[str(label)] = unmapped.get(str(label), 0) + int(n)
        class_idx.append(mapped.fillna(-1).to_numpy())
        dataset_idx.append(np.full(len(df), dataset_to_idx[name], dtype=np.int64))
        names.append(np.full(len(df), name))

    if unmapped:
        details = ", ".join(f"'{label}' ({n} rows)" for label, n in sorted(unmapped.items(), key=lambda kv: -kv[1]))
        raise ValueError(
            f"canonical_label value(s) present in data but not in the fixed class space "
            f"[Benign, *data.active_classes]: {details}. Every raw label mapped via "
            f"data.label_mapping must resolve to Benign or an entry in data.active_classes."
        )

    return PreparedSplit(
        features=np.concatenate(feats, axis=0).astype(np.float32),
        class_idx=np.concatenate(class_idx).astype(np.int64),
        dataset_idx=np.concatenate(dataset_idx),
        dataset_name=np.concatenate(names),
    )


def _print_class_counts(active_datasets: list[str], splits: dict[str, "Split"], class_names: list[str]) -> None:
    header = f"{'class':<20s}" + "".join(f"{name:>18s}" for name in active_datasets)
    print("[prepare_datasets] per-dataset class counts (train split):")
    print(header)
    for cls in class_names:
        row = f"{cls:<20s}"
        for name in active_datasets:
            count = int((splits[name].train["canonical_label"] == cls).sum())
            row += f"{count:>18d}"
        print(row)


def prepare_datasets(config: dict) -> PreparedData:
    assert_active_datasets_consistent(config)

    data_cfg = config["data"]
    active_datasets = list(data_cfg["active_datasets"])
    active_classes = list(data_cfg["active_classes"])
    seed = config.get("seed", 0)

    label_mapping = data_cfg.get("label_mapping", {})
    strict = data_cfg.get("strict_label_mapping", True)
    max_rows = data_cfg.get("max_rows_per_dataset")

    splits: dict[str, Split] = {}
    for name in active_datasets:
        spec = get_spec(name)
        raw = load_dataset(
            spec,
            chunksize=data_cfg.get("chunksize", 200_000),
            mapping_override=label_mapping.get(name),
            strict=strict,
            max_rows=max_rows,
            seed=seed,
        )
        raw = cap_dataset_rows(raw, max_rows, seed=seed)
        splits[name] = stratified_split(
            raw,
            train_frac=data_cfg["split"]["train_frac"],
            val_frac=data_cfg["split"]["val_frac"],
            test_frac=data_cfg["split"]["test_frac"],
            seed=seed,
            min_class_count=data_cfg["split"].get("min_class_count", 3),
        )

    harmonizer = Harmonizer(active_datasets, divergent_features=frozenset(data_cfg.get("divergent_features", [])))
    harmonizer.fit([TrainSplit(dataset_name=name, frame=splits[name].train) for name in active_datasets])

    class_names = [BENIGN_LABEL, *active_classes]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    dataset_to_idx = {name: i for i, name in enumerate(active_datasets)}

    _print_class_counts(active_datasets, splits, class_names)

    train = _vectorize({n: splits[n].train for n in active_datasets}, harmonizer, class_to_idx, dataset_to_idx)
    val = _vectorize({n: splits[n].val for n in active_datasets}, harmonizer, class_to_idx, dataset_to_idx)
    test = _vectorize({n: splits[n].test for n in active_datasets}, harmonizer, class_to_idx, dataset_to_idx)

    return PreparedData(
        harmonizer=harmonizer,
        class_names=class_names,
        active_datasets=active_datasets,
        train=train,
        val=val,
        test=test,
    )


class HarmonizedTensorDataset(Dataset):
    """Feeds (features, class_idx, dataset_idx) triples. `dataset_idx` is
    exposed here (unlike moe_nids' HarmonizedTensorDataset, which never
    exposes dataset identity at all) because Stage B/C training and the
    `dataset_aux_loss` regularizer need it as a training-time TARGET, not a
    model input -- it is never concatenated onto `features`, and
    `MoEDatasetNIDS.forward` never receives it (see models/moe.py).
    """

    def __init__(self, split: PreparedSplit) -> None:
        self.features = torch.from_numpy(split.features)
        self.class_idx = torch.from_numpy(split.class_idx)
        self.dataset_idx = torch.from_numpy(split.dataset_idx)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.class_idx[idx], self.dataset_idx[idx]
