"""Bounded-memory Stage A/B/C training for disk-backed full NF-v3 data."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.encoder import ProbeHead, SharedEncoder
from models.losses import dataset_aux_loss, load_balance_penalty
from models.moe import MoEDatasetNIDS

from .checkpoint import (
    STAGE_A_FILE,
    STAGE_B_FILE,
    STAGE_C_FILE,
    clear_progress,
    load_progress,
    load_stage_a,
    save_progress,
    save_stage_a,
    save_stage_b,
    save_stage_c,
)
from .model_utils import (
    bank_kind_for_architecture,
    build_expert_bank,
    build_model,
    expert_forward_one,
    expert_train_params,
)
from .out_of_core_data import OutOfCoreContext, class_counts

CONTRACT_FILE = "run_contract.json"
STAGE_C_SUMMARY_FILE = "stage_c_training_summary.json"


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def ensure_run_contract(config: dict, context: OutOfCoreContext) -> dict:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    contract = {
        "format_version": 1,
        "execution_mode": "out_of_core_full",
        "architecture": config["architecture"],
        "active_datasets": context.data.active_datasets,
        "class_names": context.data.class_names,
        "feature_columns": context.feature_columns,
        "split_signatures": context.split_signatures,
        "preprocessing_signature": context.preprocessing_signature,
        "model": config["model"],
        "load_balance": config["load_balance"],
        "training": {
            key: value for key, value in config["training"].items()
            if key not in {"checkpoint_dir", "device", "force_restart", "progress_every_rows"}
        },
    }
    contract["signature"] = _hash(contract)
    path = os.path.join(checkpoint_dir, CONTRACT_FILE)
    if os.path.isfile(path):
        with open(path) as handle:
            previous = json.load(handle)
        if previous != contract:
            completed = [name for name in (STAGE_A_FILE, STAGE_B_FILE, STAGE_C_FILE) if os.path.isfile(os.path.join(checkpoint_dir, name))]
            raise ValueError(
                f"Run contract changed for checkpoint directory {checkpoint_dir}; existing stages={completed}. "
                "Use a new RUN_NAME or deliberately set FORCE_RESTART=True."
            )
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    return contract


def _bounded_slices(length: int, batch_size: int):
    start = 0
    while start < length:
        remaining = length - start
        end = start + batch_size - 1 if remaining == batch_size + 1 and batch_size > 2 else min(start + batch_size, length)
        yield slice(start, end)
        start = end


def shuffled_row_batches(
    start: int,
    stop: int,
    rng: np.random.Generator,
    batch_size: int,
    block_rows: int,
    buffer_blocks: int,
):
    """Use every row once while shuffling through a bounded block buffer."""
    if stop <= start:
        return
    blocks = [(left, min(left + block_rows, stop)) for left in range(start, stop, block_rows)]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] == 1:
        blocks[-2] = (blocks[-2][0], blocks[-1][1])
        blocks.pop()
    order = rng.permutation(len(blocks))
    for window_start in range(0, len(order), buffer_blocks):
        window = order[window_start : window_start + buffer_blocks]
        row_ids = np.concatenate([
            np.arange(blocks[int(i)][0], blocks[int(i)][1], dtype=np.int64) for i in window
        ])
        rng.shuffle(row_ids)
        for section in _bounded_slices(len(row_ids), batch_size):
            yield row_ids[section]


def _batch(split, row_ids, device):
    features = np.asarray(split.features[row_ids], dtype=np.float32)
    labels = np.asarray(split.class_idx[row_ids], dtype=np.int64)
    datasets = np.asarray(split.dataset_idx[row_ids], dtype=np.int64)
    return (
        torch.from_numpy(features).to(device, non_blocking=True),
        torch.from_numpy(labels).to(device, non_blocking=True),
        torch.from_numpy(datasets).to(device, non_blocking=True),
    )


def _loss_weights(labels, num_classes: int, device) -> torch.Tensor:
    counts = class_counts(labels, num_classes)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = np.sqrt(counts[present].sum() / (present.sum() * counts[present]))
    weights[present] /= weights[present].mean()
    print("[ooc] sqrt-inverse-frequency class weights:", {
        int(i): round(float(weight), 4) for i, weight in enumerate(weights) if weight > 0
    })
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _settings(config: dict):
    training = config["training"]
    return (
        int(training["batch_size"]),
        int(training.get("shuffle_block_rows", 65_536)),
        int(training.get("shuffle_buffer_blocks", 8)),
    )


def _progress_reporter(tag: str, total_rows: int, every_rows: int):
    """Return a cheap row-progress callback for long Colab epochs."""
    started = time.monotonic()
    next_report = max(1, every_rows)

    def report(rows_seen: int, *, force: bool = False) -> None:
        nonlocal next_report
        if not force and rows_seen < next_report:
            return
        elapsed = time.monotonic() - started
        rate = rows_seen / elapsed if elapsed > 0 else 0.0
        percent = 100.0 * rows_seen / total_rows if total_rows else 100.0
        print(
            f"[{tag}] progress={rows_seen:,}/{total_rows:,} ({percent:.1f}%) "
            f"elapsed={elapsed / 60:.1f}m rate={rate:,.0f} rows/s",
            flush=True,
        )
        while next_report <= rows_seen:
            next_report += max(1, every_rows)

    return report


def run_stage_a_ooc(config: dict, context: OutOfCoreContext) -> SharedEncoder:
    data = context.data
    device = torch.device(config["training"]["device"])
    torch.manual_seed(config.get("seed", 0))
    model_cfg = config["model"]
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1], hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"], activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    probe = ProbeHead(model_cfg["latent_dim"], len(data.class_names)).to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *probe.parameters()], lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    weights = _loss_weights(data.train.class_idx, len(data.class_names), device)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    start_epoch = 0
    progress = load_progress(checkpoint_dir, "A")
    if progress:
        encoder.load_state_dict(progress["encoder_state"])
        probe.load_state_dict(progress["probe_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        print(f"[Stage A/ooc] resuming at epoch {start_epoch}")
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for epoch in range(start_epoch, int(config["training"]["epochs_a"])):
        rng = np.random.default_rng(config.get("seed", 0) + epoch)
        encoder.train(); probe.train()
        loss_sum = 0.0; rows_seen = 0
        report_progress = _progress_reporter(
            f"Stage A/ooc epoch {epoch + 1}", len(data.train.class_idx), progress_every
        )
        for row_ids in shuffled_row_batches(0, len(data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, _ = _batch(data.train, row_ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(encoder(features)), labels, weight=weights)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(f"[Stage A/ooc] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f} rows={rows_seen:,}")
        save_progress(checkpoint_dir, "A", {
            "epoch": epoch + 1, "encoder_state": encoder.state_dict(),
            "probe_state": probe.state_dict(), "optimizer_state": optimizer.state_dict(),
        })
    save_stage_a(checkpoint_dir, encoder.state_dict(), data.class_names)
    clear_progress(checkpoint_dir, "A")
    return encoder


def _frozen_encoder(config: dict, context: OutOfCoreContext, device) -> SharedEncoder:
    data = context.data; model_cfg = config["model"]
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1], hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"], activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    encoder.load_state_dict(load_stage_a(config["training"]["checkpoint_dir"])["encoder_state"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def run_stage_b_ooc(config: dict, context: OutOfCoreContext):
    data = context.data; device = torch.device(config["training"]["device"])
    torch.manual_seed(config.get("seed", 0))
    encoder = _frozen_encoder(config, context, device)
    bank_kind = bank_kind_for_architecture(config["architecture"])
    bank = build_expert_bank(
        bank_kind, data.active_datasets, config["model"]["latent_dim"],
        len(data.class_names), config["model"],
    ).to(device)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    progress = load_progress(checkpoint_dir, "B")
    resume_dataset = 0; resume_epoch = 0; optimizer_state = None
    if progress:
        bank.load_state_dict(progress["expert_bank_state"])
        resume_dataset = int(progress["dataset_i"]); resume_epoch = int(progress["epoch"])
        optimizer_state = progress.get("optimizer_state")
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for dataset_i, name in enumerate(data.active_datasets):
        if dataset_i < resume_dataset:
            continue
        bounds = data.train.dataset_slices[name]
        local_labels = data.train.class_idx[bounds]
        weights = _loss_weights(local_labels, len(data.class_names), device)
        optimizer = torch.optim.Adam(expert_train_params(bank, dataset_i), lr=config["training"]["lr"])
        first_epoch = resume_epoch if dataset_i == resume_dataset else 0
        if dataset_i == resume_dataset and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            rng = np.random.default_rng(config.get("seed", 0) + dataset_i * 10_000 + epoch)
            loss_sum = 0.0; rows_seen = 0
            total_rows = bounds.stop - bounds.start
            report_progress = _progress_reporter(
                f"Stage B/ooc:{name} epoch {epoch + 1}", total_rows, progress_every
            )
            for row_ids in shuffled_row_batches(bounds.start, bounds.stop, rng, batch_size, block_rows, buffer_blocks):
                features, labels, _ = _batch(data.train, row_ids, device)
                with torch.no_grad():
                    latent = encoder(features)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(expert_forward_one(bank, dataset_i, latent), labels, weight=weights)
                loss.backward(); optimizer.step()
                loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
                report_progress(rows_seen)
            report_progress(rows_seen, force=True)
            print(f"[Stage B/ooc:{name}] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f} rows={rows_seen:,}")
            save_progress(checkpoint_dir, "B", {
                "dataset_i": dataset_i, "epoch": epoch + 1,
                "expert_bank_state": bank.state_dict(), "optimizer_state": optimizer.state_dict(),
            })
        # Mark this expert complete before moving to the next one.
        save_progress(checkpoint_dir, "B", {
            "dataset_i": dataset_i + 1, "epoch": 0,
            "expert_bank_state": bank.state_dict(), "optimizer_state": None,
        })
    save_stage_b(checkpoint_dir, bank.state_dict(), data.active_datasets, bank_kind)
    clear_progress(checkpoint_dir, "B")
    return bank


def _set_encoder_trainable(encoder, mode: str):
    for parameter in encoder.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "last_layer":
        for parameter in encoder.net[-1].parameters():
            parameter.requires_grad_(True)
    elif mode not in {"all", "none"}:
        raise ValueError(f"Unknown stage_c_unfreeze mode: {mode}")


def build_ooc_model(config: dict, context: OutOfCoreContext, device) -> torch.nn.Module:
    data = context.data; model_cfg = config["model"]
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1], hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"], activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    encoder.load_state_dict(load_stage_a(config["training"]["checkpoint_dir"])["encoder_state"])
    model = build_model(config["architecture"], encoder, data.active_datasets, data.class_names, model_cfg).to(device)
    from .checkpoint import load_stage_b
    model.expert_bank.load_state_dict(load_stage_b(config["training"]["checkpoint_dir"])["expert_bank_state"])
    return model


def _validation_macro_f1(model, split, num_classes: int, device, chunk_rows: int) -> float:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(split.class_idx), chunk_rows):
            stop = min(start + chunk_rows, len(split.class_idx))
            features = np.asarray(split.features[start:stop], dtype=np.float32)
            truth = np.asarray(split.class_idx[start:stop], dtype=np.int64)
            prediction = model.predict(torch.from_numpy(features).to(device, non_blocking=True)).cpu().numpy()
            confusion += np.bincount(
                truth * num_classes + prediction, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=(precision + recall) != 0)
    return float(f1[support > 0].mean())


def run_stage_c_ooc(config: dict, context: OutOfCoreContext):
    data = context.data; device = torch.device(config["training"]["device"])
    torch.manual_seed(config.get("seed", 0))
    model = build_ooc_model(config, context, device)
    _set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0),
    )
    weights = _loss_weights(data.train.class_idx, len(data.class_names), device)
    stage_cfg = config["training"]["stage_c"]
    supervision = stage_cfg.get("gate_supervision", "light_aux")
    aux_lambda = {
        "none": 0.0,
        "light_aux": float(stage_cfg.get("lambda_dataset_aux", 0.1)),
        "hard": float(stage_cfg.get("lambda_dataset_aux_hard", 5.0)),
    }[supervision]
    balance_lambda = float(config["load_balance"]["lambda_balance"])
    checkpoint_dir = config["training"]["checkpoint_dir"]
    start_epoch = 0
    best_val_macro_f1 = -float("inf")
    best_model_state = None
    best_epoch = None
    patience_left = int(config["training"].get("early_stopping_patience", 5))
    progress = load_progress(checkpoint_dir, "C")
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        best_val_macro_f1 = float(progress.get("best_val_macro_f1", best_val_macro_f1))
        best_model_state = progress.get("best_model_state")
        best_epoch = progress.get("best_epoch")
        patience_left = int(progress.get("patience_left", patience_left))
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for epoch in range(start_epoch, int(config["training"]["epochs_c"])):
        rng = np.random.default_rng(config.get("seed", 0) + 100_000 + epoch)
        totals = {"ce": 0.0, "balance": 0.0, "aux": 0.0}; rows_seen = 0
        model.train()
        report_progress = _progress_reporter(
            f"Stage C/ooc epoch {epoch + 1}", len(data.train.class_idx), progress_every
        )
        for row_ids in shuffled_row_batches(0, len(data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, dataset_ids = _batch(data.train, row_ids, device)
            output = model(features)
            ce = F.nll_loss(MoEDatasetNIDS.combined_probs_to_log_probs(output["combined_probs"]), labels, weight=weights)
            balance = load_balance_penalty(output["gate_weights"])
            aux = dataset_aux_loss(output["gate_weights"], dataset_ids) if aux_lambda > 0 else ce.new_zeros(())
            loss = ce + balance_lambda * balance + aux_lambda * aux
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            amount = len(row_ids); rows_seen += amount
            totals["ce"] += float(ce.item()) * amount
            totals["balance"] += float(balance.item()) * amount
            totals["aux"] += float(aux.item()) * amount
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(
            f"[Stage C/ooc] epoch {epoch + 1}: CE={totals['ce']/rows_seen:.6f} "
            f"balance={totals['balance']/rows_seen:.6f} aux={totals['aux']/rows_seen:.6f} rows={rows_seen:,}"
        )
        val_macro_f1 = _validation_macro_f1(
            model, data.val, len(data.class_names), device,
            int(config["training"].get("validation_chunk_rows", 262_144)),
        )
        improved = val_macro_f1 > best_val_macro_f1 + 1e-12
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_left = int(config["training"].get("early_stopping_patience", 5))
        else:
            patience_left -= 1
        print(
            f"[Stage C/ooc] epoch {epoch + 1}: val_macro_f1={val_macro_f1:.6f} "
            f"best={best_val_macro_f1:.6f} patience_left={patience_left}"
        )
        save_progress(checkpoint_dir, "C", {
            "epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1, "best_model_state": best_model_state,
            "best_epoch": best_epoch, "patience_left": patience_left,
        })
        if patience_left <= 0:
            print(f"[Stage C/ooc] early stopping after epoch {epoch + 1}")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    bank_kind = bank_kind_for_architecture(config["architecture"])
    save_stage_c(checkpoint_dir, model.state_dict(), data.class_names, data.active_datasets, bank_kind)
    summary_path = os.path.join(checkpoint_dir, STAGE_C_SUMMARY_FILE)
    temporary = summary_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(
            {"best_epoch": best_epoch, "best_validation_macro_f1": best_val_macro_f1},
            handle, indent=2, sort_keys=True,
        )
    os.replace(temporary, summary_path)
    clear_progress(checkpoint_dir, "C")
    return model
