"""Checkpoint-aware latent-space evaluation for the private-encoder MoE.

The report compares the pooled Stage-A encoder, the Stage-B gate/experts, and
the Stage-C gate/experts on identical stratified rows.  All quantitative
geometry is computed in the original latent space; PCA/UMAP are visual aids.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

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
from training.checkpoint import load_stage_b, load_stage_c, resolve_stage_a_path
from training.model_utils import build_expert_bank
from training.out_of_core_train import build_ooc_model


@dataclass(frozen=True)
class LatentSnapshot:
    stage: str
    encoder: str
    module: torch.nn.Module
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


def load_private_encoder_snapshots(config: dict, context, device: torch.device) -> list[LatentSnapshot]:
    """Load every meaningful A/B/C encoder without conflating stage names."""
    if config["architecture"] != "moe_dataset_private_encoders":
        raise ValueError("private latent reporting requires architecture=moe_dataset_private_encoders")
    input_dim = context.data.train.features.shape[1]
    stage_a_checkpoint = torch.load(resolve_stage_a_path(config), map_location="cpu")
    stage_a = _new_encoder(config, input_dim, device)
    stage_a.load_state_dict(stage_a_checkpoint["encoder_state"])
    snapshots = [LatentSnapshot("A", "shared_initialization", stage_a, "shared")]

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
    # Stage B never updates the dedicated gate encoder; retain an explicit
    # alias so reports do not imply that a second representation was learned.
    stage_b_gate = _new_encoder(config, input_dim, device)
    stage_b_gate.load_state_dict(stage_a_checkpoint["encoder_state"])
    snapshots.append(LatentSnapshot(
        "B", "gate", stage_b_gate, "gate", equivalent_to="A__shared_initialization"
    ))
    for name, expert in zip(context.data.active_datasets, bank.experts):
        snapshots.append(LatentSnapshot("B", f"expert::{name}", expert.encoder, "private_expert"))

    model = build_ooc_model(config, context, device)
    model.load_state_dict(load_stage_c(config["training"]["checkpoint_dir"])["model_state"])
    snapshots.append(LatentSnapshot("C", "gate", model.encoder, "gate"))
    if not isinstance(model.expert_bank, PrivateEncoderExpertBank):
        raise TypeError("Stage-C checkpoint does not contain private encoders")
    for name, expert in zip(context.data.active_datasets, model.expert_bank.experts):
        snapshots.append(LatentSnapshot("C", f"expert::{name}", expert.encoder, "private_expert"))
    for snapshot in snapshots:
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


def evaluate_private_latent_checkpoints(
    config: dict,
    context,
    *,
    split_name: str = "val",
    output_dir: str | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Create CSV reports and return sampled embeddings for visualization."""
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be val or test")
    latent_cfg = config.get("evaluation", {}).get("latent", {})
    seed = int(latent_cfg.get("random_seed", config.get("seed", 0)))
    device = torch.device(device or config["training"].get("device", "cpu"))
    train, evaluation = context.data.train, getattr(context.data, split_name)
    train_rows = stratified_row_sample(
        train.class_idx, train.dataset_idx,
        int(latent_cfg.get("max_train_rows", 20000)),
        int(latent_cfg.get("max_per_class_dataset", 250)), seed,
    )
    eval_rows = stratified_row_sample(
        evaluation.class_idx, evaluation.dataset_idx,
        int(latent_cfg.get("max_eval_rows", 5000)),
        int(latent_cfg.get("max_per_class_dataset", 250)), seed + 1,
    )
    train_y = np.asarray(train.class_idx[train_rows], dtype=np.int64)
    train_d = np.asarray(train.dataset_idx[train_rows], dtype=np.int64)
    eval_y = np.asarray(evaluation.class_idx[eval_rows], dtype=np.int64)
    eval_d = np.asarray(evaluation.dataset_idx[eval_rows], dtype=np.int64)
    batch_size = int(latent_cfg.get("batch_size", 4096))
    snapshots = load_private_encoder_snapshots(config, context, device)
    global_rows, class_rows, probe_rows, probe_class_rows = [], [], [], []
    embeddings = {}
    for snapshot in snapshots:
        train_z = _extract(snapshot.module, train.features, train_rows, device, batch_size)
        eval_z = _extract(snapshot.module, evaluation.features, eval_rows, device, batch_size)
        embeddings[snapshot.snapshot_id] = eval_z
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

    output_dir = output_dir or config["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    frames = {
        "snapshots": pd.DataFrame(global_rows),
        "per_class": pd.DataFrame(class_rows),
        "probes": pd.DataFrame(probe_rows),
        "probe_per_class": pd.DataFrame(probe_class_rows),
    }
    filenames = {
        "snapshots": "Latent_Snapshot_Metrics.csv",
        "per_class": "Latent_Per_Class.csv",
        "probes": "Latent_Probe_Metrics.csv",
        "probe_per_class": "Latent_Probe_Per_Class.csv",
    }
    for key, frame in frames.items():
        frame.to_csv(os.path.join(output_dir, filenames[key]), index=False)
    manifest = pd.DataFrame({
        "row_index": eval_rows,
        "class_idx": eval_y,
        "class": [context.data.class_names[index] for index in eval_y],
        "dataset_idx": eval_d,
        "dataset": [context.data.active_datasets[index] for index in eval_d],
        "split": split_name,
    })
    manifest.to_csv(os.path.join(output_dir, "Latent_Sample_Manifest.csv"), index=False)
    with open(os.path.join(output_dir, "Latent_Report_Config.json"), "w") as handle:
        json.dump({"split": split_name, "seed": seed, "snapshots": list(embeddings)}, handle, indent=2)
    return {**frames, "manifest": manifest, "embeddings": embeddings}


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
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
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
    columns = 3
    rows = int(np.ceil(len(projections) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    selected = manifest["class"].to_numpy() == class_name
    for axis, (snapshot_id, projection) in zip(axes, projections.items()):
        axis.scatter(projection[~selected, 0], projection[~selected, 1], s=5, alpha=0.12, color="lightgray")
        axis.scatter(projection[selected, 0], projection[selected, 1], s=14, alpha=0.8, color="crimson")
        axis.set_title(snapshot_id); axis.set_xticks([]); axis.set_yticks([])
    for axis in axes[len(projections):]:
        axis.axis("off")
    figure_dir = os.path.join(output_dir, "latent_figures")
    os.makedirs(figure_dir, exist_ok=True)
    safe_name = class_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(figure_dir, f"class_focus__{safe_name}.png")
    fig.suptitle(f"Latent-space focus: {class_name}", fontsize=16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
