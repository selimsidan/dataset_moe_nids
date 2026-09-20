"""Checkpoint-aware latent-space evaluation for the private-encoder MoE.

The report compares the pooled Stage-A encoder, the Stage-B gate/experts, and
the Stage-C gate/experts on identical stratified rows.  All quantitative
geometry is computed in the original latent space; PCA/UMAP are visual aids.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    recall_score,
    silhouette_samples,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from models.encoder import SharedEncoder
from models.private_encoder_experts import PrivateEncoderExpertBank
from training.checkpoint import (
    STAGE_B_FILE,
    STAGE_C_FILE,
    load_stage_b,
    load_stage_c,
    resolve_stage_a_path,
)
from training.model_utils import build_expert_bank
from training.out_of_core_train import build_ooc_model


@dataclass(frozen=True)
class LatentSnapshot:
    stage: str
    encoder: str
    module: torch.nn.Module | None
    representation: str
    equivalent_to: str | None = None

    @property
    def snapshot_id(self) -> str:
        return f"{self.stage}__{self.encoder}"


def _new_encoder(config: dict, input_dim: int, device: torch.device) -> SharedEncoder:
    model_cfg = config["model"]
    return SharedEncoder(
        input_dim=input_dim,
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)


def _normalize_stages(stages: Iterable[str] | None) -> tuple[str, ...]:
    source = ("A", "B", "C") if stages is None else stages
    requested = tuple(dict.fromkeys(str(stage).upper() for stage in source))
    invalid = sorted(set(requested) - {"A", "B", "C"})
    if invalid:
        raise ValueError(f"unknown latent stages: {invalid}; expected A, B, and/or C")
    if not requested:
        raise ValueError("at least one latent stage must be requested")
    return tuple(stage for stage in ("A", "B", "C") if stage in requested)


def load_private_encoder_snapshots(
    config: dict,
    context,
    device: torch.device,
    *,
    stages: Iterable[str] | None = None,
) -> list[LatentSnapshot]:
    """Load only the requested meaningful encoder snapshots."""
    if config["architecture"] != "moe_dataset_private_encoders":
        raise ValueError("private latent reporting requires architecture=moe_dataset_private_encoders")
    stages = _normalize_stages(stages)
    input_dim = context.data.train.features.shape[1]
    snapshots: list[LatentSnapshot] = []
    if "A" in stages:
        stage_a_checkpoint = torch.load(resolve_stage_a_path(config), map_location="cpu")
        stage_a = _new_encoder(config, input_dim, device)
        stage_a.load_state_dict(stage_a_checkpoint["encoder_state"])
        snapshots.append(LatentSnapshot("A", "shared_initialization", stage_a, "shared"))

    if "B" in stages:
        stage_b_checkpoint = load_stage_b(config["training"]["checkpoint_dir"])
        bank = build_expert_bank(
            "private_encoder",
            context.data.active_datasets,
            config["model"]["latent_dim"],
            len(context.data.class_names),
            config["model"],
            input_dim=input_dim,
        ).to(device)
        bank.load_state_dict(stage_b_checkpoint["expert_bank_state"])
        if not isinstance(bank, PrivateEncoderExpertBank):
            raise TypeError("expected a PrivateEncoderExpertBank")
        # Stage B never updates the dedicated gate encoder. Keep a metadata-only
        # alias; evaluation reuses the Stage-A embeddings and metric rows.
        snapshots.append(LatentSnapshot(
            "B", "gate", None, "gate", equivalent_to="A__shared_initialization"
        ))
        for name, expert in zip(context.data.active_datasets, bank.experts):
            snapshots.append(LatentSnapshot("B", f"expert::{name}", expert.encoder, "private_expert"))

    if "C" in stages:
        model = build_ooc_model(config, context, device)
        model.load_state_dict(load_stage_c(config["training"]["checkpoint_dir"])["model_state"])
        snapshots.append(LatentSnapshot("C", "gate", model.encoder, "gate"))
        if not isinstance(model.expert_bank, PrivateEncoderExpertBank):
            raise TypeError("Stage-C checkpoint does not contain private encoders")
        for name, expert in zip(context.data.active_datasets, model.expert_bank.experts):
            snapshots.append(LatentSnapshot("C", f"expert::{name}", expert.encoder, "private_expert"))
    for snapshot in snapshots:
        if snapshot.module is not None:
            snapshot.module.eval()
    return snapshots


def stratified_row_sample(
    labels,
    dataset_ids,
    max_rows: int,
    max_per_class_dataset: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels if isinstance(labels, np.ndarray) else labels[:], dtype=np.int64)
    dataset_ids = np.asarray(
        dataset_ids if isinstance(dataset_ids, np.ndarray) else dataset_ids[:], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        for dataset_id in np.unique(dataset_ids[labels == class_id]):
            pool = np.flatnonzero((labels == class_id) & (dataset_ids == dataset_id))
            amount = min(len(pool), max_per_class_dataset)
            selected.extend(rng.choice(pool, size=amount, replace=False).tolist())
    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    if len(selected) > max_rows:
        selected = np.sort(rng.choice(selected, size=max_rows, replace=False))
    return selected


def _extract(module, features, rows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    output = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            ids = rows[start : start + batch_size]
            values = np.asarray(features[ids], dtype=np.float32)
            output.append(module(torch.from_numpy(values).to(device)).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)


def _random_pair_metrics(z: np.ndarray, labels: np.ndarray, seed: int, pairs: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by_class = {value: np.flatnonzero(labels == value) for value in np.unique(labels)}
    valid_classes = [value for value, rows in by_class.items() if len(rows) >= 2]
    alignment_values = []
    if valid_classes:
        for _ in range(min(pairs, len(z) * 4)):
            class_id = rng.choice(valid_classes)
            left, right = rng.choice(by_class[class_id], size=2, replace=False)
            alignment_values.append(float(np.square(z[left] - z[right]).sum()))
    if len(z) >= 2:
        left = rng.integers(0, len(z), size=min(pairs, len(z) * 4))
        right = rng.integers(0, len(z), size=len(left))
        different = left != right
        squared = np.square(z[left[different]] - z[right[different]]).sum(axis=1)
        uniformity = float(np.log(np.exp(-2.0 * squared).mean())) if len(squared) else float("nan")
    else:
        uniformity = float("nan")
    return (
        float(np.mean(alignment_values)) if alignment_values else float("nan"),
        uniformity,
    )


def _effective_rank(z: np.ndarray) -> float:
    if len(z) < 2:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(np.cov(z, rowvar=False)).clip(min=0)
    if eigenvalues.sum() == 0:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def _geometry_rows(
    snapshot: LatentSnapshot,
    z: np.ndarray,
    labels: np.ndarray,
    dataset_ids: np.ndarray,
    class_names: list[str],
    dataset_names: list[str],
    silhouette_cap: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    raw_norm_mean = float(np.linalg.norm(z, axis=1).mean()) if len(z) else float("nan")
    z = normalize(z)
    rng = np.random.default_rng(seed)
    metric_rows = np.arange(len(z))
    if len(metric_rows) > silhouette_cap:
        metric_rows = np.sort(rng.choice(metric_rows, silhouette_cap, replace=False))
    sample_z, sample_labels = z[metric_rows], labels[metric_rows]
    present = np.unique(sample_labels)
    silhouette = np.full(len(sample_z), np.nan)
    if 1 < len(present) < len(sample_z):
        silhouette = silhouette_samples(sample_z, sample_labels, metric="cosine")
    centroids = {
        class_id: normalize(z[labels == class_id].mean(axis=0, keepdims=True))[0]
        for class_id in np.unique(labels)
    }
    class_rows = []
    for class_id, centroid in centroids.items():
        mask = labels == class_id
        own_distance = 1.0 - z[mask] @ centroid
        rivals = [1.0 - float(centroid @ other) for other_id, other in centroids.items() if other_id != class_id]
        nearest_rival = min(rivals) if rivals else float("nan")
        local_silhouette = silhouette[sample_labels == class_id]
        domain_centroids = []
        for dataset_id in np.unique(dataset_ids[mask]):
            local = mask & (dataset_ids == dataset_id)
            domain_centroids.append(normalize(z[local].mean(axis=0, keepdims=True))[0])
        domain_distances = [
            1.0 - float(domain_centroids[i] @ domain_centroids[j])
            for i in range(len(domain_centroids)) for j in range(i + 1, len(domain_centroids))
        ]
        class_rows.append({
            "snapshot": snapshot.snapshot_id,
            "stage": snapshot.stage,
            "encoder": snapshot.encoder,
            "class": class_names[int(class_id)],
            "support": int(mask.sum()),
            "dataset_support": int(len(domain_centroids)),
            "mean_own_centroid_distance": float(own_distance.mean()),
            "p90_own_centroid_distance": float(np.quantile(own_distance, 0.9)),
            "nearest_rival_centroid_distance": nearest_rival,
            "centroid_margin": nearest_rival - float(own_distance.mean()),
            "silhouette": float(np.nanmean(local_silhouette)) if len(local_silhouette) else float("nan"),
            "cross_dataset_centroid_dispersion": (
                float(np.mean(domain_distances)) if domain_distances else float("nan")
            ),
        })
    alignment, uniformity = _random_pair_metrics(z, labels, seed)
    global_row = {
        "snapshot": snapshot.snapshot_id,
        "stage": snapshot.stage,
        "encoder": snapshot.encoder,
        "representation": snapshot.representation,
        "equivalent_to": snapshot.equivalent_to,
        "rows": len(z),
        "classes_with_support": len(np.unique(labels)),
        "silhouette": float(np.nanmean(silhouette)),
        "davies_bouldin": float(davies_bouldin_score(sample_z, sample_labels)) if len(present) > 1 else float("nan"),
        "calinski_harabasz": float(calinski_harabasz_score(sample_z, sample_labels)) if len(present) > 1 else float("nan"),
        "alignment": alignment,
        "uniformity": uniformity,
        "effective_rank": _effective_rank(z),
        "mean_norm_before_normalization": raw_norm_mean,
    }
    return global_row, class_rows


def _probe_rows(
    snapshot: LatentSnapshot,
    train_z: np.ndarray,
    train_y: np.ndarray,
    train_dataset: np.ndarray,
    eval_z: np.ndarray,
    eval_y: np.ndarray,
    eval_dataset: np.ndarray,
    class_names: list[str],
    neighbors: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    train_z, eval_z = normalize(train_z), normalize(eval_z)
    models = {
        "knn": KNeighborsClassifier(
            n_neighbors=min(neighbors, len(train_z)), metric="cosine", weights="distance", n_jobs=-1
        ),
        "linear": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, random_state=seed
        ),
    }
    overall, per_class = [], []
    for name, model in models.items():
        model.fit(train_z, train_y)
        prediction = model.predict(eval_z)
        labels = np.unique(eval_y)
        overall.append({
            "snapshot": snapshot.snapshot_id,
            "stage": snapshot.stage,
            "encoder": snapshot.encoder,
            "probe": name,
            "target": "class",
            "accuracy": accuracy_score(eval_y, prediction),
            "balanced_accuracy": balanced_accuracy_score(eval_y, prediction),
            "macro_f1_present": f1_score(eval_y, prediction, labels=labels, average="macro", zero_division=0),
        })
        recalls = recall_score(eval_y, prediction, labels=labels, average=None, zero_division=0)
        for class_id, recall in zip(labels, recalls):
            per_class.append({
                "snapshot": snapshot.snapshot_id,
                "stage": snapshot.stage,
                "encoder": snapshot.encoder,
                "probe": name,
                "class": class_names[int(class_id)],
                "recall": float(recall),
                "support": int((eval_y == class_id).sum()),
            })
    domain_probe = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=seed
    )
    domain_probe.fit(train_z, train_dataset)
    domain_prediction = domain_probe.predict(eval_z)
    overall.append({
        "snapshot": snapshot.snapshot_id,
        "stage": snapshot.stage,
        "encoder": snapshot.encoder,
        "probe": "linear",
        "target": "dataset",
        "accuracy": accuracy_score(eval_dataset, domain_prediction),
        "balanced_accuracy": balanced_accuracy_score(eval_dataset, domain_prediction),
        "macro_f1_present": f1_score(eval_dataset, domain_prediction, average="macro", zero_division=0),
    })
    return overall, per_class


_REPORT_FILES = {
    "snapshots": "Latent_Snapshot_Metrics.csv",
    "per_class": "Latent_Per_Class.csv",
    "probes": "Latent_Probe_Metrics.csv",
    "probe_per_class": "Latent_Probe_Per_Class.csv",
}
_EVAL_MANIFEST_FILE = "Latent_Sample_Manifest.csv"
_TRAIN_MANIFEST_FILE = "Latent_Train_Sample_Manifest.csv"
_EVAL_EMBEDDINGS_FILE = "Latent_Embeddings.npz"
_TRAIN_EMBEDDINGS_FILE = "Latent_Train_Embeddings.npz"
_REPORT_CONFIG_FILE = "Latent_Report_Config.json"


def _atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_npz(values: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _rows_digest(*row_groups: np.ndarray) -> str:
    digest = hashlib.sha256()
    for rows in row_groups:
        digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _checkpoint_fingerprints(config: dict, stages: tuple[str, ...]) -> dict[str, dict[str, int | str]]:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    paths = {}
    if "A" in stages or "B" in stages:
        paths["A"] = resolve_stage_a_path(config)
    if "B" in stages:
        paths["B"] = os.path.join(checkpoint_dir, STAGE_B_FILE)
    if "C" in stages:
        paths["C"] = os.path.join(checkpoint_dir, STAGE_C_FILE)
    fingerprints = {}
    for stage, path in paths.items():
        stat = os.stat(path)
        fingerprints[stage] = {
            "path": os.path.abspath(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return fingerprints


def _sample_manifest(rows, split, split_name: str, context) -> pd.DataFrame:
    labels = np.asarray(split.class_idx[rows], dtype=np.int64)
    datasets = np.asarray(split.dataset_idx[rows], dtype=np.int64)
    return pd.DataFrame({
        "row_index": np.asarray(rows, dtype=np.int64),
        "class_idx": labels,
        "class": [context.data.class_names[index] for index in labels],
        "dataset_idx": datasets,
        "dataset": [context.data.active_datasets[index] for index in datasets],
        "split": split_name,
    })


def _validate_manifest(frame: pd.DataFrame, split, split_name: str, context) -> np.ndarray:
    required = {"row_index", "class_idx", "class", "dataset_idx", "dataset", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"latent sample manifest is missing columns: {missing}")
    if set(frame["split"].astype(str)) != {split_name}:
        raise ValueError(f"latent sample manifest does not describe split={split_name!r}")
    rows = frame["row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(split.class_idx)):
        raise ValueError("latent sample manifest contains duplicate or out-of-range row indices")
    labels = np.asarray(split.class_idx[rows], dtype=np.int64)
    datasets = np.asarray(split.dataset_idx[rows], dtype=np.int64)
    expected_classes = np.asarray([context.data.class_names[index] for index in labels], dtype=str)
    expected_datasets = np.asarray([context.data.active_datasets[index] for index in datasets], dtype=str)
    if not np.array_equal(labels, frame["class_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("latent sample manifest class indices do not match the prepared split")
    if not np.array_equal(datasets, frame["dataset_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("latent sample manifest dataset indices do not match the prepared split")
    if not np.array_equal(expected_classes, frame["class"].astype(str).to_numpy()):
        raise ValueError("latent sample manifest class names do not match the prepared split")
    if not np.array_equal(expected_datasets, frame["dataset"].astype(str).to_numpy()):
        raise ValueError("latent sample manifest dataset names do not match the prepared split")
    return rows


def _resolve_sample_manifests(
    context,
    split_name: str,
    latent_cfg: dict,
    seed: int,
    sample_manifest: str | os.PathLike | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train, evaluation = context.data.train, getattr(context.data, split_name)
    train_manifest = eval_manifest = None
    if sample_manifest is not None:
        source = Path(sample_manifest)
        if source.is_dir():
            train_path = source / _TRAIN_MANIFEST_FILE
            eval_path = source / _EVAL_MANIFEST_FILE
        else:
            train_path = source.with_name(_TRAIN_MANIFEST_FILE)
            eval_path = source
        if not eval_path.is_file():
            raise FileNotFoundError(f"no reusable latent manifest at {eval_path}")
        eval_manifest = pd.read_csv(eval_path)
        if train_path.is_file():
            train_manifest = pd.read_csv(train_path)

    if train_manifest is None:
        train_rows = stratified_row_sample(
            train.class_idx, train.dataset_idx,
            int(latent_cfg.get("max_train_rows", 20000)),
            int(latent_cfg.get("max_per_class_dataset", 250)), seed,
        )
        train_manifest = _sample_manifest(train_rows, train, "train", context)
    else:
        train_rows = _validate_manifest(train_manifest, train, "train", context)
    if eval_manifest is None:
        eval_rows = stratified_row_sample(
            evaluation.class_idx, evaluation.dataset_idx,
            int(latent_cfg.get("max_eval_rows", 5000)),
            int(latent_cfg.get("max_per_class_dataset", 250)), seed + 1,
        )
        eval_manifest = _sample_manifest(eval_rows, evaluation, split_name, context)
    else:
        eval_rows = _validate_manifest(eval_manifest, evaluation, split_name, context)
    return train_rows, eval_rows, train_manifest, eval_manifest


def load_latent_report(output_dir: str | os.PathLike) -> dict[str, object]:
    """Load a completed latent report, including embeddings used by plots."""
    output_dir = os.fspath(output_dir)
    with open(os.path.join(output_dir, _REPORT_CONFIG_FILE)) as handle:
        report_config = json.load(handle)
    frames = {
        key: pd.read_csv(os.path.join(output_dir, filename))
        for key, filename in _REPORT_FILES.items()
    }
    with np.load(os.path.join(output_dir, _EVAL_EMBEDDINGS_FILE)) as stored:
        embeddings = {key: stored[key] for key in stored.files}
    with np.load(os.path.join(output_dir, _TRAIN_EMBEDDINGS_FILE)) as stored:
        train_embeddings = {key: stored[key] for key in stored.files}
    return {
        **frames,
        "manifest": pd.read_csv(os.path.join(output_dir, _EVAL_MANIFEST_FILE)),
        "train_manifest": pd.read_csv(os.path.join(output_dir, _TRAIN_MANIFEST_FILE)),
        "embeddings": embeddings,
        "train_embeddings": train_embeddings,
        "config": report_config,
    }


def _write_latent_report(report: dict[str, object], output_dir: str, report_config: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, filename in _REPORT_FILES.items():
        _atomic_csv(report[key], os.path.join(output_dir, filename))
    _atomic_csv(report["manifest"], os.path.join(output_dir, _EVAL_MANIFEST_FILE))
    _atomic_csv(report["train_manifest"], os.path.join(output_dir, _TRAIN_MANIFEST_FILE))
    _atomic_npz(report["embeddings"], os.path.join(output_dir, _EVAL_EMBEDDINGS_FILE))
    _atomic_npz(report["train_embeddings"], os.path.join(output_dir, _TRAIN_EMBEDDINGS_FILE))
    # This is deliberately last: its presence marks a complete, loadable bundle.
    _atomic_json(report_config, os.path.join(output_dir, _REPORT_CONFIG_FILE))


def _aliased_rows(rows, source_id: str, snapshot: LatentSnapshot) -> list[dict]:
    copied = []
    if isinstance(rows, pd.DataFrame):
        source_rows = rows.loc[rows["snapshot"] == source_id].to_dict("records")
    else:
        source_rows = [row for row in rows if row["snapshot"] == source_id]
    for row in source_rows:
        row = dict(row)
        row.update({"snapshot": snapshot.snapshot_id, "stage": snapshot.stage, "encoder": snapshot.encoder})
        if "representation" in row:
            row.update({"representation": snapshot.representation, "equivalent_to": source_id})
        copied.append(row)
    if not copied:
        raise ValueError(f"alias source {source_id!r} is absent from the supplied Stage-A report")
    return copied


def evaluate_private_latent_checkpoints(
    config: dict,
    context,
    *,
    split_name: str = "val",
    output_dir: str | None = None,
    device: str | torch.device | None = None,
    stages: Iterable[str] | None = None,
    sample_manifest: str | os.PathLike | None = None,
    reference_report: dict[str, object] | None = None,
    reuse_completed: bool = True,
) -> dict[str, object]:
    """Create an atomic, stage-selective latent report and sampled embeddings."""
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be val or test")
    stages = _normalize_stages(stages)
    latent_cfg = config.get("evaluation", {}).get("latent", {})
    seed = int(latent_cfg.get("random_seed", config.get("seed", 0)))
    device = torch.device(device or config["training"].get("device", "cpu"))
    train, evaluation = context.data.train, getattr(context.data, split_name)
    train_rows, eval_rows, train_manifest, manifest = _resolve_sample_manifests(
        context, split_name, latent_cfg, seed, sample_manifest,
    )
    train_y = np.asarray(train.class_idx[train_rows], dtype=np.int64)
    train_d = np.asarray(train.dataset_idx[train_rows], dtype=np.int64)
    eval_y = np.asarray(evaluation.class_idx[eval_rows], dtype=np.int64)
    eval_d = np.asarray(evaluation.dataset_idx[eval_rows], dtype=np.int64)
    batch_size = int(latent_cfg.get("batch_size", 4096))
    snapshots = load_private_encoder_snapshots(config, context, device, stages=stages)
    output_dir = output_dir or config["evaluation"]["output_dir"]
    report_config = {
        "format_version": 2,
        "split": split_name,
        "seed": seed,
        "stages": list(stages),
        "snapshots": [snapshot.snapshot_id for snapshot in snapshots],
        "sample_rows_sha256": _rows_digest(train_rows, eval_rows),
        "class_names": list(context.data.class_names),
        "active_datasets": list(context.data.active_datasets),
        "latent_config": latent_cfg,
        "checkpoint_fingerprints": _checkpoint_fingerprints(config, stages),
    }
    config_path = os.path.join(output_dir, _REPORT_CONFIG_FILE)
    if reuse_completed and os.path.isfile(config_path):
        try:
            cached = load_latent_report(output_dir)
        except (FileNotFoundError, KeyError, ValueError, OSError):
            cached = None
        if cached is not None and cached["config"] == report_config:
            print(f"[latent] reusing complete report at {output_dir}")
            return cached

    global_rows, class_rows, probe_rows, probe_class_rows = [], [], [], []
    embeddings, train_embeddings = {}, {}
    reference_report = reference_report or {}
    for snapshot in snapshots:
        if snapshot.equivalent_to:
            source_id = snapshot.equivalent_to
            source_embeddings = embeddings if source_id in embeddings else reference_report.get("embeddings", {})
            source_train_embeddings = (
                train_embeddings if source_id in train_embeddings else reference_report.get("train_embeddings", {})
            )
            if source_id not in source_embeddings or source_id not in source_train_embeddings:
                raise ValueError(
                    "Stage-B gate is an alias of Stage A. Evaluate Stage A first and pass its "
                    "report as reference_report, or request stages=('A', 'B') together."
                )
            eval_z = np.asarray(source_embeddings[source_id])
            train_z = np.asarray(source_train_embeddings[source_id])
            source_frames = {
                "snapshots": global_rows or reference_report.get("snapshots", pd.DataFrame()),
                "per_class": class_rows or reference_report.get("per_class", pd.DataFrame()),
                "probes": probe_rows or reference_report.get("probes", pd.DataFrame()),
                "probe_per_class": probe_class_rows or reference_report.get("probe_per_class", pd.DataFrame()),
            }
            global_rows.extend(_aliased_rows(source_frames["snapshots"], source_id, snapshot))
            class_rows.extend(_aliased_rows(source_frames["per_class"], source_id, snapshot))
            probe_rows.extend(_aliased_rows(source_frames["probes"], source_id, snapshot))
            probe_class_rows.extend(_aliased_rows(source_frames["probe_per_class"], source_id, snapshot))
            embeddings[snapshot.snapshot_id] = eval_z
            train_embeddings[snapshot.snapshot_id] = train_z
            print(f"[latent:{snapshot.snapshot_id}] alias={source_id} rows={len(eval_z):,}")
            continue
        if snapshot.module is None:
            raise RuntimeError(f"snapshot {snapshot.snapshot_id} has no encoder module")
        train_z = _extract(snapshot.module, train.features, train_rows, device, batch_size)
        eval_z = _extract(snapshot.module, evaluation.features, eval_rows, device, batch_size)
        embeddings[snapshot.snapshot_id] = eval_z
        train_embeddings[snapshot.snapshot_id] = train_z
        global_row, local_class_rows = _geometry_rows(
            snapshot, eval_z, eval_y, eval_d,
            context.data.class_names, context.data.active_datasets,
            int(latent_cfg.get("max_silhouette_rows", 5000)), seed,
        )
        global_rows.append(global_row); class_rows.extend(local_class_rows)
        local_probes, local_probe_classes = _probe_rows(
            snapshot, train_z, train_y, train_d, eval_z, eval_y, eval_d,
            context.data.class_names, int(latent_cfg.get("knn_neighbors", 5)), seed,
        )
        probe_rows.extend(local_probes); probe_class_rows.extend(local_probe_classes)
        print(f"[latent:{snapshot.snapshot_id}] silhouette={global_row['silhouette']:.4f} rows={len(eval_z):,}")

    frames = {
        "snapshots": pd.DataFrame(global_rows),
        "per_class": pd.DataFrame(class_rows),
        "probes": pd.DataFrame(probe_rows),
        "probe_per_class": pd.DataFrame(probe_class_rows),
    }
    report = {
        **frames,
        "manifest": manifest,
        "train_manifest": train_manifest,
        "embeddings": embeddings,
        "train_embeddings": train_embeddings,
        "config": report_config,
    }
    _write_latent_report(report, output_dir, report_config)
    return report


def combine_latent_reports(
    reports: Iterable[dict[str, object]],
    output_dir: str | os.PathLike,
) -> dict[str, object]:
    """Atomically persist the cumulative view of compatible stage reports."""
    reports = list(reports)
    if not reports:
        raise ValueError("at least one latent report is required")
    manifest = reports[0]["manifest"].reset_index(drop=True)
    train_manifest = reports[0]["train_manifest"].reset_index(drop=True)
    for report in reports[1:]:
        if not manifest.equals(report["manifest"].reset_index(drop=True)):
            raise ValueError("cannot combine latent reports with different evaluation manifests")
        if not train_manifest.equals(report["train_manifest"].reset_index(drop=True)):
            raise ValueError("cannot combine latent reports with different training manifests")
    combined = {
        key: pd.concat([report[key] for report in reports], ignore_index=True)
        for key in _REPORT_FILES
    }
    combined["manifest"] = manifest
    combined["train_manifest"] = train_manifest
    combined["embeddings"] = {}
    combined["train_embeddings"] = {}
    for report in reports:
        for key in ("embeddings", "train_embeddings"):
            for snapshot_id, values in report[key].items():
                if snapshot_id in combined[key] and not np.array_equal(combined[key][snapshot_id], values):
                    raise ValueError(f"conflicting embeddings for snapshot {snapshot_id}")
                combined[key][snapshot_id] = values
    stages = [stage for stage in ("A", "B", "C") if stage in set(combined["snapshots"]["stage"])]
    report_config = {
        "format_version": 2,
        "kind": "cumulative",
        "split": str(manifest["split"].iloc[0]),
        "stages": stages,
        "snapshots": list(combined["embeddings"]),
        "sample_rows_sha256": _rows_digest(
            train_manifest["row_index"].to_numpy(), manifest["row_index"].to_numpy()
        ),
        "source_configs": [report.get("config", {}) for report in reports],
    }
    combined["config"] = report_config
    _write_latent_report(combined, os.fspath(output_dir), report_config)
    return combined


def _embedding_digest(embeddings: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for snapshot_id, values in embeddings.items():
        digest.update(snapshot_id.encode())
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def plot_latent_snapshots(
    report: dict[str, object],
    output_dir: str,
    *,
    method: str = "umap",
    random_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Save paired class/dataset projections for every snapshot."""
    import matplotlib.pyplot as plt

    manifest = report["manifest"]
    embeddings = report["embeddings"]
    projections = {}
    figure_dir = os.path.join(output_dir, "latent_figures")
    os.makedirs(figure_dir, exist_ok=True)
    projection_path = os.path.join(figure_dir, f"{method}_projections.npz")
    projection_config_path = os.path.join(figure_dir, f"{method}_projection_config.json")
    projection_config = {
        "format_version": 1,
        "method": method,
        "random_seed": int(random_seed),
        "rows": int(len(manifest)),
        "snapshots": list(embeddings),
        "embedding_sha256": _embedding_digest(embeddings),
    }
    stored_config = None
    if os.path.isfile(projection_path) and os.path.isfile(projection_config_path):
        try:
            with open(projection_config_path) as handle:
                stored_config = json.load(handle)
            with np.load(projection_path) as stored:
                cached_projections = {key: stored[key] for key in stored.files}
        except (OSError, ValueError, KeyError):
            cached_projections = None
        if stored_config == projection_config and cached_projections is not None:
            expected_figures = [
                os.path.join(figure_dir, f"{snapshot_id.replace('::', '_')}__{method}.png")
                for snapshot_id in embeddings
            ]
            if all(os.path.isfile(path) for path in expected_figures):
                print(f"[latent] reusing {method.upper()} projections at {figure_dir}")
                return cached_projections
    for snapshot_id, values in embeddings.items():
        if method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise RuntimeError("Install umap-learn or use method='pca'") from exc
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=random_seed)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=random_seed)
        else:
            raise ValueError("method must be umap or pca")
        projection = reducer.fit_transform(normalize(values))
        projections[snapshot_id] = projection
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        for axis, column, title in zip(axes, ("class", "dataset"), ("Canonical class", "Source dataset")):
            categories = manifest[column].astype(str)
            for category in sorted(categories.unique()):
                mask = categories == category
                axis.scatter(projection[mask, 0], projection[mask, 1], s=7, alpha=0.55, label=category)
            axis.set_title(f"{snapshot_id} — {title}")
            axis.set_xticks([]); axis.set_yticks([])
            axis.legend(markerscale=2, fontsize=6, frameon=False, loc="best")
        path = os.path.join(figure_dir, f"{snapshot_id.replace('::', '_')}__{method}.png")
        temporary = path + ".tmp"
        fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
        plt.close(fig)
    _atomic_npz(projections, projection_path)
    _atomic_json(projection_config, projection_config_path)
    for stale_snapshot in set((stored_config or {}).get("snapshots", [])) - set(embeddings):
        stale_path = os.path.join(
            figure_dir, f"{stale_snapshot.replace('::', '_')}__{method}.png"
        )
        if os.path.isfile(stale_path):
            os.remove(stale_path)
    return projections


def plot_class_focus(
    report: dict[str, object],
    projections: dict[str, np.ndarray],
    class_name: str,
    output_dir: str,
) -> str:
    """Highlight one requested class across every A/B/C encoder snapshot."""
    import matplotlib.pyplot as plt

    manifest = report["manifest"]
    if class_name not in set(manifest["class"]):
        raise ValueError(f"class {class_name!r} has no rows in the sampled split")
    selected = manifest["class"].to_numpy() == class_name
    figure_dir = os.path.join(output_dir, "latent_figures")
    os.makedirs(figure_dir, exist_ok=True)
    safe_name = class_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(figure_dir, f"class_focus__{safe_name}.png")
    config_path = os.path.join(figure_dir, f"class_focus__{safe_name}.json")
    focus_config = {
        "format_version": 1,
        "class": class_name,
        "snapshots": list(projections),
        "projection_sha256": _embedding_digest(projections),
        "selected_rows_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }
    if os.path.isfile(path) and os.path.isfile(config_path):
        try:
            with open(config_path) as handle:
                stored_config = json.load(handle)
        except (OSError, ValueError):
            stored_config = None
        if stored_config == focus_config:
            return path
    columns = 3
    rows = int(np.ceil(len(projections) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (snapshot_id, projection) in zip(axes, projections.items()):
        axis.scatter(projection[~selected, 0], projection[~selected, 1], s=5, alpha=0.12, color="lightgray")
        axis.scatter(projection[selected, 0], projection[selected, 1], s=14, alpha=0.8, color="crimson")
        axis.set_title(snapshot_id); axis.set_xticks([]); axis.set_yticks([])
    for axis in axes[len(projections):]:
        axis.axis("off")
    fig.suptitle(f"Latent-space focus: {class_name}", fontsize=16)
    temporary = path + ".tmp"
    fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(fig)
    _atomic_json(focus_config, config_path)
    return path
