"""Bounded-memory training for the standalone hard two-stage baseline.

This is deliberately separate from the MoE Stage A/B/C implementation:
phase A learns dataset identity with plain cross-entropy, while phase B
learns fully independent encoder+head classifiers for each dataset.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.baselines import HardTwoStageModel

from .checkpoint import (
    HARD_ROUTER_FILE,
    HARD_CLASSIFIERS_FILE,
    _atomic_torch_save,
    clear_progress,
    load_progress,
    save_progress,
    load_validated_stage_a,
    stage_a_metadata,
)
from .out_of_core_data import OutOfCoreContext
from .out_of_core_train import _batch, _hash, _progress_reporter, _settings, shuffled_row_batches


def build_hard_two_stage_ooc(config: dict, context: OutOfCoreContext) -> HardTwoStageModel:
    data = context.data
    model_cfg = config["model"]
    model = HardTwoStageModel(
        data.active_datasets,
        input_dim=data.train.features.shape[1],
        latent_dim=model_cfg["latent_dim"],
        num_classes=len(data.class_names),
        id_hidden_dims=model_cfg["encoder"]["hidden_dims"],
        stage_b_hidden_dims=model_cfg["encoder"]["hidden_dims"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(torch.device(config["training"]["device"]))
    baseline_cfg = config.get("training", {}).get("baseline", {})
    encoder_init = baseline_cfg.get("encoder_init", "stage_a")
    if encoder_init not in {"random", "stage_a"}:
        raise ValueError("training.baseline.encoder_init must be random or stage_a")
    if encoder_init == "stage_a":
        metadata = stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns,
        )
        state = load_validated_stage_a(config, metadata)["encoder_state"]
        model.id_encoder.load_state_dict(state)
        for encoder in model.stage_b.encoders.values():
            encoder.load_state_dict(state)
    return model


def _load_phase_a(checkpoint_dir: str, model: HardTwoStageModel) -> None:
    state = torch.load(os.path.join(checkpoint_dir, HARD_ROUTER_FILE), map_location="cpu")
    model.id_encoder.load_state_dict(state["id_encoder_state"])
    model.id_head.load_state_dict(state["id_head_state"])


def run_hard_phase_a_ooc(config: dict, context: OutOfCoreContext) -> HardTwoStageModel:
    """Train only the dataset-ID classifier with unweighted/plain CE."""
    torch.manual_seed(config.get("seed", 0))
    model = build_hard_two_stage_ooc(config, context)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        [*model.id_encoder.parameters(), *model.id_head.parameters()],
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    checkpoint_dir = config["training"]["checkpoint_dir"]
    progress = load_progress(checkpoint_dir, "HR")
    start_epoch = 0
    steps = examples = 0
    started = time.monotonic()
    if progress:
        model.id_encoder.load_state_dict(progress["id_encoder_state"])
        model.id_head.load_state_dict(progress["id_head_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        steps = int(progress.get("optimizer_steps", 0))
        examples = int(progress.get("examples_seen", 0))
        print(
            f"[hard-two-stage phase A] resuming after completed epoch {start_epoch}; "
            "the interrupted partial epoch will be replayed",
            flush=True,
        )
    batch_size, block_rows, buffer_blocks = _settings(config)
    total_rows = len(context.data.train.dataset_idx)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    phase_a_epochs = int(config["training"]["epochs_a"])
    print(
        f"[hard-two-stage phase A] schedule={phase_a_epochs} epoch(s), "
        f"starting_epoch={start_epoch + 1}",
        flush=True,
    )
    for epoch in range(start_epoch, phase_a_epochs):
        rng = np.random.default_rng(config.get("seed", 0) + epoch)
        model.id_encoder.train(); model.id_head.train()
        loss_sum = 0.0; rows_seen = 0
        report = _progress_reporter(f"hard-two-stage phase A epoch {epoch + 1}", total_rows, progress_every)
        for row_ids in shuffled_row_batches(0, total_rows, rng, batch_size, block_rows, buffer_blocks):
            features, _, dataset_ids = _batch(context.data.train, row_ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.dataset_logits(features), dataset_ids)
            loss.backward(); optimizer.step()
            steps += 1; examples += len(row_ids)
            loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
            report(rows_seen)
        report(rows_seen, force=True)
        print(f"[hard-two-stage phase A] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f}")
        save_progress(checkpoint_dir, "HR", {
            "epoch": epoch + 1,
            "id_encoder_state": model.id_encoder.state_dict(),
            "id_head_state": model.id_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": steps,
            "examples_seen": examples,
        })
        print(
            f"[hard-two-stage phase A] resumable checkpoint saved after epoch {epoch + 1}",
            flush=True,
        )
    _atomic_torch_save({
        "id_encoder_state": model.id_encoder.state_dict(),
        "id_head_state": model.id_head.state_dict(),
        "dataset_names": context.data.active_datasets,
        "training_summary": {
            "optimizer_steps": steps,
            "examples_seen": examples,
            "epochs_completed": phase_a_epochs,
            "wall_seconds": time.monotonic() - started,
            "selected_epoch": phase_a_epochs,
        },
    }, os.path.join(checkpoint_dir, HARD_ROUTER_FILE))
    clear_progress(checkpoint_dir, "HR")
    return model


def run_hard_phase_b_ooc(config: dict, context: OutOfCoreContext) -> HardTwoStageModel:
    """Train each dataset's encoder+head alone; no parameters are shared."""
    torch.manual_seed(config.get("seed", 0))
    model = build_hard_two_stage_ooc(config, context)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    _load_phase_a(checkpoint_dir, model)
    device = next(model.parameters()).device
    progress = load_progress(checkpoint_dir, "HC")
    resume_dataset = 0; resume_epoch = 0; optimizer_state = None
    steps = examples = 0
    started = time.monotonic()
    if progress:
        model.stage_b.load_state_dict(progress["stage_b_state"])
        resume_dataset = int(progress["dataset_i"])
        resume_epoch = int(progress["epoch"])
        optimizer_state = progress.get("optimizer_state")
        steps = int(progress.get("optimizer_steps", 0))
        examples = int(progress.get("examples_seen", 0))
        print(
            f"[hard-two-stage phase B] resuming dataset_index={resume_dataset} "
            f"after completed epoch={resume_epoch}",
            flush=True,
        )
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for dataset_i, name in enumerate(context.data.active_datasets):
        if dataset_i < resume_dataset:
            continue
        params = [*model.stage_b.encoders[name].parameters(), *model.stage_b.heads[name].parameters()]
        optimizer = torch.optim.Adam(
            params,
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 0.0),
        )
        if dataset_i == resume_dataset and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        bounds = context.data.train.dataset_slices[name]
        first_epoch = resume_epoch if dataset_i == resume_dataset else 0
        total_rows = bounds.stop - bounds.start
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            rng = np.random.default_rng(config.get("seed", 0) + dataset_i * 10_000 + epoch)
            model.stage_b.encoders[name].train(); model.stage_b.heads[name].train()
            loss_sum = 0.0; rows_seen = 0
            report = _progress_reporter(
                f"hard-two-stage phase B:{name} epoch {epoch + 1}", total_rows, progress_every
            )
            for row_ids in shuffled_row_batches(bounds.start, bounds.stop, rng, batch_size, block_rows, buffer_blocks):
                features, labels, _ = _batch(context.data.train, row_ids, device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.stage_b(features, name)["logits"], labels)
                loss.backward(); optimizer.step()
                steps += 1; examples += len(row_ids)
                loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
                report(rows_seen)
            report(rows_seen, force=True)
            print(f"[hard-two-stage phase B:{name}] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f}")
            save_progress(checkpoint_dir, "HC", {
                "dataset_i": dataset_i,
                "epoch": epoch + 1,
                "stage_b_state": model.stage_b.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": steps,
                "examples_seen": examples,
            })
            print(
                f"[hard-two-stage phase B:{name}] resumable checkpoint saved "
                f"after epoch {epoch + 1}",
                flush=True,
            )
        save_progress(checkpoint_dir, "HC", {
            "dataset_i": dataset_i + 1,
            "epoch": 0,
            "stage_b_state": model.stage_b.state_dict(),
            "optimizer_state": None,
            "optimizer_steps": steps,
            "examples_seen": examples,
        })
    _atomic_torch_save({
        "model_state": model.state_dict(),
        "class_names": context.data.class_names,
        "dataset_names": context.data.active_datasets,
        "training_summary": {
            "hard_router": torch.load(
                os.path.join(checkpoint_dir, HARD_ROUTER_FILE), map_location="cpu"
            ).get("training_summary", {}),
            "hard_classifiers": {
                "optimizer_steps": steps,
                "examples_seen": examples,
                "epochs_completed": int(config["training"]["epochs_b"]),
                "wall_seconds": time.monotonic() - started,
                "selected_epoch": int(config["training"]["epochs_b"]),
            },
        },
    }, os.path.join(checkpoint_dir, HARD_CLASSIFIERS_FILE))
    clear_progress(checkpoint_dir, "HC")
    return model


def load_hard_two_stage_ooc(config: dict, context: OutOfCoreContext) -> HardTwoStageModel:
    model = build_hard_two_stage_ooc(config, context)
    checkpoint = torch.load(
        os.path.join(config["training"]["checkpoint_dir"], HARD_CLASSIFIERS_FILE), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state"])
    model.training_summary = checkpoint.get("training_summary") or {}
    return model
