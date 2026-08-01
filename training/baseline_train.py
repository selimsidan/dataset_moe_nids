"""Trains the baselines. All reuse the same harmonization layer,
PreparedData, and evaluation code as the primary architecture -- only the
model layer and training loop differ, and all are reachable through the
same `architecture` config flag as MoEDatasetNIDS (training/run.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.baselines import HardTwoStageModel, NoFusionModel, PlainPooledSoftmax
from models.encoder import SharedEncoder

from .dataset import HarmonizedTensorDataset, PreparedData
from .sampler import ClassBalancedBatchSampler


def train_plain_pooled(config: dict, data: PreparedData) -> PlainPooledSoftmax:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    model_cfg = config["model"]

    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1],
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    model = PlainPooledSoftmax(encoder, len(data.class_names)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0))

    dataset = HarmonizedTensorDataset(data.train)
    sampler = ClassBalancedBatchSampler(
        data.train.class_idx,
        batch_size=config["training"]["batch_size"],
        min_per_class=config["training"]["min_per_class_per_batch"],
        seed=config.get("seed", 0),
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    epochs = config["training"].get("epochs_c", config["training"].get("epochs_a", 20))
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for features, class_idx, _dataset_idx in loader:
            features, class_idx = features.to(device), class_idx.to(device)
            logits = model(features)["logits"]
            loss = F.cross_entropy(logits, class_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[plain_pooled] epoch {epoch}: CE={total_loss / n_batches:.4f}")
    return model


def _train_per_dataset_heads(
    encoders: dict, heads: dict, active_datasets: list[str], data: PreparedData, config: dict, device: torch.device, tag: str
) -> None:
    epochs = config["training"].get("epochs_c", config["training"].get("epochs_a", 20))
    for name in active_datasets:
        mask = data.train.dataset_name == name
        features = torch.from_numpy(data.train.features[mask])
        labels = torch.from_numpy(data.train.class_idx[mask])
        if len(labels) == 0:
            print(f"[{tag}] dataset '{name}': no rows, skipping.")
            continue

        loader = DataLoader(TensorDataset(features, labels), batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
        sub_params = list(encoders[name].parameters()) + list(heads[name].parameters())
        optimizer = torch.optim.Adam(sub_params, lr=config["training"]["lr"])

        for epoch in range(epochs):
            total_loss, n_batches = 0.0, 0
            for batch_features, batch_labels in loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                logits = heads[name](encoders[name](batch_features))
                loss = F.cross_entropy(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            print(f"[{tag}:{name}] epoch {epoch}: CE={total_loss / max(1, n_batches):.4f}")


def train_no_fusion(config: dict, data: PreparedData) -> NoFusionModel:
    """No cross-dataset code path: builds and trains one encoder+head per
    dataset, each seeing only its own rows."""
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    model_cfg = config["model"]
    active_datasets = data.active_datasets

    model = NoFusionModel(
        active_datasets,
        input_dim=data.train.features.shape[1],
        latent_dim=model_cfg["latent_dim"],
        num_classes=len(data.class_names),
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
    ).to(device)

    _train_per_dataset_heads(model.encoders, model.heads, active_datasets, data, config, device, tag="no_fusion")
    return model


def train_hard_two_stage(config: dict, data: PreparedData) -> HardTwoStageModel:
    """Stage (a): standalone dataset-ID classifier, plain CE against
    ground-truth dataset id, using the same SharedEncoder architecture as
    the primary model for a fair comparison (its own weights, not shared).
    Stage (b): independent per-dataset classifiers (NoFusionModel-style).
    """
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    model_cfg = config["model"]
    active_datasets = data.active_datasets

    model = HardTwoStageModel(
        active_datasets,
        input_dim=data.train.features.shape[1],
        latent_dim=model_cfg["latent_dim"],
        num_classes=len(data.class_names),
        id_hidden_dims=model_cfg["encoder"]["hidden_dims"],
        stage_b_hidden_dims=model_cfg["encoder"]["hidden_dims"],
    ).to(device)

    # Stage (a): dataset-ID classifier, trained on ALL pooled data with
    # plain CE against ground-truth dataset_idx -- the one place in this
    # whole project a model is DELIBERATELY, primarily supervised on
    # dataset identity, since that's exactly what this baseline is for.
    id_params = list(model.id_encoder.parameters()) + list(model.id_head.parameters())
    optimizer = torch.optim.Adam(id_params, lr=config["training"]["lr"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.train.features), torch.from_numpy(data.train.dataset_idx)),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    epochs = config["training"].get("epochs_a", 20)
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for features, dataset_idx in loader:
            features, dataset_idx = features.to(device), dataset_idx.to(device)
            logits = model.id_head(model.id_encoder(features))
            loss = F.cross_entropy(logits, dataset_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[hard_two_stage:id_classifier] epoch {epoch}: CE={total_loss / n_batches:.4f}")

    # Stage (b): independent per-dataset classifiers.
    _train_per_dataset_heads(model.stage_b.encoders, model.stage_b.heads, active_datasets, data, config, device, tag="hard_two_stage:stage_b")
    return model
