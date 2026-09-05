"""Fairness-controlled in-memory trainers for dense and structural baselines."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.baselines import HardTwoStageModel, NoFusionModel, PlainPooledSoftmax
from models.encoder import SharedEncoder

from .checkpoint import load_validated_stage_a, stage_a_metadata
from .dataset import HarmonizedTensorDataset, PreparedData
from .model_utils import build_matched_dense_model
from .sampler import ClassBalancedBatchSampler
from .stage_c_jointfinetune import _set_encoder_trainable


def _baseline_config(config: dict) -> dict:
    values = config.get("training", {}).get("baseline", {})
    encoder_init = values.get("encoder_init", "stage_a")
    warmstart = values.get("stage_b_warmstart", "none")
    if encoder_init not in {"random", "stage_a"}:
        raise ValueError("training.baseline.encoder_init must be random or stage_a")
    if warmstart not in {"none", "matched_exposure"}:
        raise ValueError("training.baseline.stage_b_warmstart must be none or matched_exposure")
    if warmstart == "matched_exposure" and encoder_init != "stage_a":
        raise ValueError("matched_exposure requires training.baseline.encoder_init=stage_a")
    return {"encoder_init": encoder_init, "stage_b_warmstart": warmstart}


def _new_encoder(config: dict, data: PreparedData, device: torch.device) -> SharedEncoder:
    model_cfg = config["model"]
    return SharedEncoder(
        input_dim=data.train.features.shape[1],
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)


def _initialize_encoder(
    encoder: SharedEncoder,
    config: dict,
    data: PreparedData,
    encoder_init: str,
) -> None:
    if encoder_init == "stage_a":
        checkpoint = load_validated_stage_a(config, stage_a_metadata(config, data))
        encoder.load_state_dict(checkpoint["encoder_state"])


def _validation_macro_f1(model, data: PreparedData, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(data.val.features).to(device))["logits"]
        prediction = logits.argmax(dim=1).cpu().numpy()
    truth = data.val.class_idx
    classes = len(data.class_names)
    confusion = np.bincount(truth * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


def _matched_exposure_warmstart(config: dict, data: PreparedData, model, device: torch.device) -> dict:
    """Train one dense head on exactly the Stage-B per-dataset batch schedule."""
    started = time.monotonic()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    dataset = HarmonizedTensorDataset(data.train)
    steps = examples = 0
    for dataset_index, name in enumerate(data.active_datasets):
        rows = np.flatnonzero(data.train.dataset_idx == dataset_index)
        sampler = ClassBalancedBatchSampler(
            data.train.class_idx[rows],
            batch_size=config["training"]["batch_size"],
            min_per_class=config["training"]["min_per_class_per_batch"],
            seed=config.get("seed", 0) + dataset_index,
        )
        for _epoch in range(config["training"]["epochs_b"]):
            for local_ids in sampler:
                global_ids = rows[np.asarray(local_ids)]
                features = dataset.features[global_ids].to(device)
                labels = dataset.class_idx[global_ids].to(device)
                with torch.no_grad():
                    latent = model.encoder(features)
                loss = F.cross_entropy(model.head(latent), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                examples += len(global_ids)
        print(f"[dense Stage B:{name}] completed {config['training']['epochs_b']} matched-exposure epochs")
    return {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "epochs_completed": config["training"]["epochs_b"],
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": config["training"]["epochs_b"],
    }


def _train_dense_stage_c(config: dict, data: PreparedData, model, device: torch.device) -> dict:
    started = time.monotonic()
    _set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    sampler = ClassBalancedBatchSampler(
        data.train.class_idx,
        batch_size=config["training"]["batch_size"],
        min_per_class=config["training"]["min_per_class_per_batch"],
        seed=config.get("seed", 0),
    )
    loader = DataLoader(HarmonizedTensorDataset(data.train), batch_sampler=sampler)
    selection = config["training"].get("selection_mode", "fixed_epochs")
    if selection not in {"fixed_epochs", "best_val"}:
        raise ValueError("training.selection_mode must be fixed_epochs or best_val")
    best_score = -float("inf")
    best_state = None
    best_epoch = None
    patience = int(config["training"].get("early_stopping_patience", 5))
    steps = examples = 0
    completed = 0
    for epoch in range(int(config["training"]["epochs_c"])):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, labels, _dataset_ids in loader:
            features, labels = features.to(device), labels.to(device)
            loss = F.cross_entropy(model(features)["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            steps += 1
            examples += len(labels)
        completed = epoch + 1
        print(f"[{config['architecture']} Stage C] epoch {completed}: CE={total_loss / max(1, batches):.4f}")
        if selection == "best_val":
            score = _validation_macro_f1(model, data, device)
            if score > best_score + 1e-12:
                best_score, best_epoch = score, completed
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = int(config["training"].get("early_stopping_patience", 5))
            else:
                patience -= 1
            if patience <= 0:
                break
    if selection == "best_val" and best_state is not None:
        model.load_state_dict(best_state)
    return {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "epochs_completed": completed,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": best_epoch if selection == "best_val" else completed,
    }


def _train_pooled_dense(config: dict, data: PreparedData, *, matched: bool):
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    curriculum = _baseline_config(config)
    encoder = _new_encoder(config, data, device)
    _initialize_encoder(encoder, config, data, curriculum["encoder_init"])
    model = (
        build_matched_dense_model(encoder, data.active_datasets, data.class_names, config["model"])
        if matched else PlainPooledSoftmax(encoder, len(data.class_names))
    ).to(device)
    summary = {}
    if curriculum["stage_b_warmstart"] == "matched_exposure":
        summary["B"] = _matched_exposure_warmstart(config, data, model, device)
    summary["C"] = _train_dense_stage_c(config, data, model, device)
    model.training_summary = summary
    return model


def train_plain_pooled(config: dict, data: PreparedData) -> PlainPooledSoftmax:
    return _train_pooled_dense(config, data, matched=False)


def train_matched_dense(config: dict, data: PreparedData):
    return _train_pooled_dense(config, data, matched=True)


def _train_per_dataset_heads(
    encoders,
    heads,
    active_datasets: list[str],
    data: PreparedData,
    config: dict,
    device: torch.device,
    tag: str,
    epochs: int,
) -> dict:
    started = time.monotonic()
    steps = examples = 0
    for dataset_index, name in enumerate(active_datasets):
        rows = np.flatnonzero(data.train.dataset_idx == dataset_index)
        sampler = ClassBalancedBatchSampler(
            data.train.class_idx[rows],
            batch_size=config["training"]["batch_size"],
            min_per_class=config["training"]["min_per_class_per_batch"],
            seed=config.get("seed", 0) + dataset_index,
        )
        optimizer = torch.optim.Adam(
            [*encoders[name].parameters(), *heads[name].parameters()],
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 0.0),
        )
        features = torch.from_numpy(data.train.features)
        labels = torch.from_numpy(data.train.class_idx)
        for epoch in range(epochs):
            for local_ids in sampler:
                global_ids = rows[np.asarray(local_ids)]
                batch_features = features[global_ids].to(device)
                batch_labels = labels[global_ids].to(device)
                loss = F.cross_entropy(heads[name](encoders[name](batch_features)), batch_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                examples += len(global_ids)
            print(f"[{tag}:{name}] epoch {epoch + 1} complete")
    return {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "epochs_completed": epochs,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": epochs,
    }


def train_no_fusion(config: dict, data: PreparedData) -> NoFusionModel:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    curriculum = _baseline_config(config)
    model_cfg = config["model"]
    model = NoFusionModel(
        data.active_datasets,
        data.train.features.shape[1],
        model_cfg["latent_dim"],
        len(data.class_names),
        model_cfg["encoder"]["hidden_dims"],
        model_cfg["encoder"]["activation"],
        model_cfg["encoder"]["dropout"],
    ).to(device)
    for encoder in model.encoders.values():
        _initialize_encoder(encoder, config, data, curriculum["encoder_init"])
    model.training_summary = {
        "hard_classifiers": _train_per_dataset_heads(
            model.encoders, model.heads, data.active_datasets, data, config, device,
            "no_fusion", int(config["training"]["epochs_b"]),
        )
    }
    return model


def train_hard_two_stage(config: dict, data: PreparedData) -> HardTwoStageModel:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    curriculum = _baseline_config(config)
    model_cfg = config["model"]
    model = HardTwoStageModel(
        data.active_datasets,
        data.train.features.shape[1],
        model_cfg["latent_dim"],
        len(data.class_names),
        model_cfg["encoder"]["hidden_dims"],
        model_cfg["encoder"]["hidden_dims"],
        model_cfg["encoder"]["activation"],
        model_cfg["encoder"]["dropout"],
    ).to(device)
    _initialize_encoder(model.id_encoder, config, data, curriculum["encoder_init"])
    for encoder in model.stage_b.encoders.values():
        _initialize_encoder(encoder, config, data, curriculum["encoder_init"])

    started = time.monotonic()
    optimizer = torch.optim.Adam(
        [*model.id_encoder.parameters(), *model.id_head.parameters()],
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.train.features), torch.from_numpy(data.train.dataset_idx)),
        batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(config.get("seed", 0)),
    )
    steps = examples = 0
    epochs = int(config["training"]["epochs_c"])
    for epoch in range(epochs):
        for features, dataset_ids in loader:
            features, dataset_ids = features.to(device), dataset_ids.to(device)
            loss = F.cross_entropy(model.dataset_logits(features), dataset_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            examples += len(dataset_ids)
        print(f"[hard_two_stage:router] epoch {epoch + 1} complete")
    router_stats = {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "epochs_completed": epochs,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": epochs,
    }
    classifier_stats = _train_per_dataset_heads(
        model.stage_b.encoders, model.stage_b.heads, data.active_datasets, data,
        config, device, "hard_two_stage:classifiers", int(config["training"]["epochs_b"]),
    )
    model.training_summary = {"hard_router": router_stats, "hard_classifiers": classifier_stats}
    return model
