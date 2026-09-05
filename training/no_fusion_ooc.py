"""Bounded-memory oracle no-fusion baseline with explicit encoder initialization."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.baselines import NoFusionModel

from .baseline_train import _baseline_config
from .checkpoint import (
    BASELINE_MODEL_FILE, _atomic_torch_save, clear_progress, load_progress,
    load_validated_stage_a, save_progress, stage_a_metadata,
)
from .out_of_core_data import OutOfCoreContext
from .out_of_core_train import _batch, _hash, _loss_weights, _settings, shuffled_row_batches


def build_no_fusion_ooc(config: dict, context: OutOfCoreContext) -> NoFusionModel:
    data = context.data; model_cfg = config["model"]
    model = NoFusionModel(
        data.active_datasets, data.train.features.shape[1], model_cfg["latent_dim"],
        len(data.class_names), model_cfg["encoder"]["hidden_dims"],
        model_cfg["encoder"]["activation"], model_cfg["encoder"]["dropout"],
    ).to(torch.device(config["training"]["device"]))
    curriculum = _baseline_config(config)
    if curriculum["encoder_init"] == "stage_a":
        metadata = stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns,
        )
        state = load_validated_stage_a(config, metadata)["encoder_state"]
        for encoder in model.encoders.values():
            encoder.load_state_dict(state)
    return model


def run_no_fusion_ooc(config: dict, context: OutOfCoreContext) -> NoFusionModel:
    torch.manual_seed(config.get("seed", 0))
    model = build_no_fusion_ooc(config, context)
    device = next(model.parameters()).device
    progress = load_progress(config["training"]["checkpoint_dir"], "NF")
    resume_dataset = resume_epoch = 0; optimizer_state = None
    steps = examples = 0; started = time.monotonic()
    if progress:
        model.load_state_dict(progress["model_state"])
        resume_dataset = int(progress["dataset_i"]); resume_epoch = int(progress["epoch"])
        optimizer_state = progress.get("optimizer_state")
        steps = int(progress.get("optimizer_steps", 0)); examples = int(progress.get("examples_seen", 0))
    batch_size, block_rows, buffer_blocks = _settings(config)
    for dataset_index, name in enumerate(context.data.active_datasets):
        if dataset_index < resume_dataset:
            continue
        optimizer = torch.optim.Adam(
            [*model.encoders[name].parameters(), *model.heads[name].parameters()],
            lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0),
        )
        if dataset_index == resume_dataset and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        bounds = context.data.train.dataset_slices[name]
        weights = _loss_weights(context.data.train.class_idx[bounds], len(context.data.class_names), device)
        first_epoch = resume_epoch if dataset_index == resume_dataset else 0
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            rng = np.random.default_rng(config.get("seed", 0) + dataset_index * 10_000 + epoch)
            for row_ids in shuffled_row_batches(bounds.start, bounds.stop, rng, batch_size, block_rows, buffer_blocks):
                features, labels, _ = _batch(context.data.train, row_ids, device)
                loss = F.cross_entropy(model(features, name)["logits"], labels, weight=weights)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                steps += 1; examples += len(row_ids)
            save_progress(config["training"]["checkpoint_dir"], "NF", {
                "dataset_i": dataset_index, "epoch": epoch + 1, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "optimizer_steps": steps,
                "examples_seen": examples,
            })
    summary = {
        "optimizer_steps": steps, "examples_seen": examples,
        "epochs_completed": int(config["training"]["epochs_b"]),
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": int(config["training"]["epochs_b"]),
    }
    model.training_summary = {"hard_classifiers": summary}
    _atomic_torch_save({
        "model_state": model.state_dict(), "training_summary": model.training_summary,
        "class_names": context.data.class_names, "dataset_names": context.data.active_datasets,
    }, os.path.join(config["training"]["checkpoint_dir"], BASELINE_MODEL_FILE))
    clear_progress(config["training"]["checkpoint_dir"], "NF")
    return model


def load_no_fusion_ooc(config: dict, context: OutOfCoreContext) -> NoFusionModel:
    model = build_no_fusion_ooc(config, context)
    checkpoint = torch.load(
        os.path.join(config["training"]["checkpoint_dir"], BASELINE_MODEL_FILE), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state"])
    model.training_summary = checkpoint.get("training_summary") or {}
    return model
