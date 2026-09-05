"""Bounded-memory fair training for plain and budget-matched dense baselines."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.baselines import PlainPooledSoftmax

from .baseline_train import _baseline_config, _new_encoder
from .checkpoint import (
    BASELINE_MODEL_FILE,
    BASELINE_STAGE_B_FILE,
    _atomic_torch_save,
    clear_progress,
    load_progress,
    load_validated_stage_a,
    stage_a_metadata,
)
from .model_utils import build_matched_dense_model
from .out_of_core_data import OutOfCoreContext
from .out_of_core_train import (
    _batch,
    _hash,
    _loss_weights,
    _progress_reporter,
    _settings,
    _validation_macro_f1,
    shuffled_row_batches,
)
from .stage_c_jointfinetune import _set_encoder_trainable


def _expected_stage_a_metadata(config: dict, context: OutOfCoreContext) -> dict:
    return stage_a_metadata(
        config,
        context.data,
        split_signature=_hash(context.split_signatures),
        feature_columns=context.feature_columns,
    )


def build_dense_ooc(config: dict, context: OutOfCoreContext):
    device = torch.device(config["training"]["device"])
    curriculum = _baseline_config(config)
    encoder = _new_encoder(config, context.data, device)
    if curriculum["encoder_init"] == "stage_a":
        encoder.load_state_dict(
            load_validated_stage_a(config, _expected_stage_a_metadata(config, context))["encoder_state"]
        )
    if config["architecture"] == "matched_dense":
        return build_matched_dense_model(
            encoder, context.data.active_datasets, context.data.class_names, config["model"]
        ).to(device)
    if config["architecture"] == "plain_pooled":
        return PlainPooledSoftmax(encoder, len(context.data.class_names)).to(device)
    raise ValueError("dense out-of-core builder supports plain_pooled and matched_dense")


def run_dense_stage_b_ooc(config: dict, context: OutOfCoreContext, model=None):
    stage_b_path = os.path.join(config["training"]["checkpoint_dir"], BASELINE_STAGE_B_FILE)
    if model is None and os.path.isfile(stage_b_path):
        model = build_dense_ooc(config, context)
        checkpoint = torch.load(stage_b_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        model.training_summary = {"B": checkpoint["training_summary"]}
        return model
    model = model or build_dense_ooc(config, context)
    curriculum = _baseline_config(config)
    if curriculum["stage_b_warmstart"] == "none":
        model.training_summary = {"B": {
            "optimizer_steps": 0, "examples_seen": 0, "epochs_completed": 0,
            "wall_seconds": 0.0, "selected_epoch": 0,
        }}
        return model
    device = next(model.parameters()).device
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    progress = load_progress(config["training"]["checkpoint_dir"], "DB")
    resume_dataset = resume_epoch = 0
    steps = examples = 0
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        resume_dataset = int(progress["dataset_i"])
        resume_epoch = int(progress["epoch"])
        steps = int(progress.get("optimizer_steps", 0))
        examples = int(progress.get("examples_seen", 0))
    batch_size, block_rows, buffer_blocks = _settings(config)
    started = time.monotonic()
    for dataset_index, name in enumerate(context.data.active_datasets):
        if dataset_index < resume_dataset:
            continue
        bounds = context.data.train.dataset_slices[name]
        weights = _loss_weights(context.data.train.class_idx[bounds], len(context.data.class_names), device)
        first_epoch = resume_epoch if dataset_index == resume_dataset else 0
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            rng = np.random.default_rng(config.get("seed", 0) + dataset_index * 10_000 + epoch)
            for row_ids in shuffled_row_batches(bounds.start, bounds.stop, rng, batch_size, block_rows, buffer_blocks):
                features, labels, _ = _batch(context.data.train, row_ids, device)
                with torch.no_grad():
                    latent = model.encoder(features)
                loss = F.cross_entropy(model.head(latent), labels, weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); optimizer.step()
                steps += 1; examples += len(row_ids)
            from .checkpoint import save_progress
            save_progress(config["training"]["checkpoint_dir"], "DB", {
                "dataset_i": dataset_index, "epoch": epoch + 1,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": steps, "examples_seen": examples,
            })
    clear_progress(config["training"]["checkpoint_dir"], "DB")
    summary = {
        "optimizer_steps": steps, "examples_seen": examples,
        "epochs_completed": int(config["training"]["epochs_b"]),
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": int(config["training"]["epochs_b"]),
    }
    model.training_summary = {"B": summary}
    _atomic_torch_save(
        {"model_state": model.state_dict(), "training_summary": summary}, stage_b_path
    )
    return model


def run_dense_stage_c_ooc(config: dict, context: OutOfCoreContext, model=None):
    model = model or run_dense_stage_b_ooc(config, context)
    device = next(model.parameters()).device
    _set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0),
    )
    weights = _loss_weights(context.data.train.class_idx, len(context.data.class_names), device)
    selection = config["training"].get("selection_mode", "fixed_epochs")
    if selection not in {"fixed_epochs", "best_val"}:
        raise ValueError("training.selection_mode must be fixed_epochs or best_val")
    progress = load_progress(config["training"]["checkpoint_dir"], "DC")
    start_epoch = steps = examples = 0
    best_score = -float("inf"); best_state = None; best_epoch = None
    patience = int(config["training"].get("early_stopping_patience", 5))
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        steps = int(progress.get("optimizer_steps", 0)); examples = int(progress.get("examples_seen", 0))
        best_score = float(progress.get("best_score", best_score)); best_state = progress.get("best_state")
        best_epoch = progress.get("best_epoch"); patience = int(progress.get("patience", patience))
    batch_size, block_rows, buffer_blocks = _settings(config)
    started = time.monotonic(); completed = start_epoch
    for epoch in range(start_epoch, int(config["training"]["epochs_c"])):
        completed = epoch + 1
        rng = np.random.default_rng(config.get("seed", 0) + 100_000 + epoch)
        model.train(); loss_sum = 0.0; rows_seen = 0
        report = _progress_reporter(
            f"{config['architecture']} Stage C/ooc epoch {completed}",
            len(context.data.train.class_idx),
            int(config["training"].get("progress_every_rows", 1_000_000)),
        )
        for row_ids in shuffled_row_batches(0, len(context.data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, _ = _batch(context.data.train, row_ids, device)
            loss = F.cross_entropy(model(features)["logits"], labels, weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            steps += 1; examples += len(row_ids); rows_seen += len(row_ids)
            loss_sum += float(loss.item()) * len(row_ids); report(rows_seen)
        report(rows_seen, force=True)
        print(f"[{config['architecture']} Stage C/ooc] epoch {completed}: CE={loss_sum / rows_seen:.6f}")
        if selection == "best_val":
            score = _validation_macro_f1(
                model, context.data.val, len(context.data.class_names), device,
                int(config["training"].get("validation_chunk_rows", 262_144)),
            )
            if score > best_score + 1e-12:
                best_score, best_epoch = score, completed
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = int(config["training"].get("early_stopping_patience", 5))
            else:
                patience -= 1
        from .checkpoint import save_progress
        save_progress(config["training"]["checkpoint_dir"], "DC", {
            "epoch": completed, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": steps, "examples_seen": examples, "best_score": best_score,
            "best_state": best_state, "best_epoch": best_epoch, "patience": patience,
        })
        if selection == "best_val" and patience <= 0:
            break
    if selection == "best_val" and best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "optimizer_steps": steps, "examples_seen": examples, "epochs_completed": completed,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": best_epoch if selection == "best_val" else completed,
    }
    prior = dict(getattr(model, "training_summary", {})); prior["C"] = summary
    model.training_summary = prior
    _atomic_torch_save({
        "model_state": model.state_dict(), "class_names": context.data.class_names,
        "dataset_names": context.data.active_datasets,
        "match_info": getattr(model, "match_info", None), "training_summary": model.training_summary,
    }, os.path.join(config["training"]["checkpoint_dir"], BASELINE_MODEL_FILE))
    clear_progress(config["training"]["checkpoint_dir"], "DC")
    return model


def load_dense_ooc(config: dict, context: OutOfCoreContext):
    model = build_dense_ooc(config, context)
    checkpoint = torch.load(
        os.path.join(config["training"]["checkpoint_dir"], BASELINE_MODEL_FILE), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state"])
    model.training_summary = checkpoint.get("training_summary") or {}
    return model
