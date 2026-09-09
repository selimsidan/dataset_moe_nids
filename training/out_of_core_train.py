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
    load_validated_stage_a,
    save_progress,
    save_stage_a,
    save_stage_b,
    save_stage_c,
    stage_a_metadata,
)
from .model_utils import (
    bank_kind_for_architecture,
    build_expert_bank,
    build_model,
    expert_forward_one,
    expert_names_for_architecture,
    expert_train_params,
)
from .out_of_core_data import OutOfCoreContext, class_counts
from .stage_c_jointfinetune import _gate_weights_for_task, _lambda_dataset_aux

CONTRACT_FILE = "run_contract.json"
STAGE_C_SUMMARY_FILE = "stage_c_training_summary.json"


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def ensure_run_contract(config: dict, context: OutOfCoreContext) -> dict:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    training_contract = {
        key: copy.deepcopy(value) for key, value in config["training"].items()
        if key not in {"checkpoint_dir", "device", "force_restart", "progress_every_rows"}
    }
    # Preserve compatibility with contracts written before these opt-in
    # ownership settings existed. Their neutral defaults do not change the
    # historical Stage-C computation; non-default notebook-09 values remain
    # part of the signed scientific contract.
    stage_c_contract = training_contract.get("stage_c", {})
    if stage_c_contract.get("expert_update_policy", "all") == "all":
        stage_c_contract.pop("expert_update_policy", None)
    if float(stage_c_contract.get("lambda_expert_anchor", 0.0)) == 0.0:
        stage_c_contract.pop("lambda_expert_anchor", None)
    # The dataset warm-start was the historical implicit behavior. Keep old
    # dataset-MoE contracts reusable while signing the non-default basic-MoE
    # random initialization mode as a scientifically meaningful difference.
    stage_b_contract = training_contract.get("stage_b", {})
    if stage_b_contract.get("warmstart_mode", "dataset") == "dataset":
        stage_b_contract.pop("warmstart_mode", None)
    if not stage_b_contract:
        training_contract.pop("stage_b", None)
    model_contract = copy.deepcopy(config["model"])
    # Preserve existing dense-run signatures written before routing became
    # configurable. Only the non-default sparse choice changes the contract.
    gate_contract = model_contract.get("gate", {})
    if gate_contract.get("routing", "dense") == "dense":
        gate_contract.pop("routing", None)
    contract = {
        "format_version": 2,
        "execution_mode": "out_of_core_full",
        "architecture": config["architecture"],
        "active_datasets": context.data.active_datasets,
        "class_names": context.data.class_names,
        "feature_columns": context.feature_columns,
        "split_signatures": context.split_signatures,
        "preprocessing_signature": context.preprocessing_signature,
        "model": model_contract,
        "load_balance": config["load_balance"],
        "training": training_contract,
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

    def memory_status() -> str:
        values = {}
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    if key in {"VmRSS", "VmHWM"}:
                        amount = float(value.split()[0]) / 1024
                        values[key] = f"{amount:.0f}MiB"
        except (OSError, ValueError, IndexError):
            pass
        parts = []
        if "VmRSS" in values:
            parts.append(f"rss={values['VmRSS']}")
        if "VmHWM" in values:
            parts.append(f"rss_peak={values['VmHWM']}")
        if torch.cuda.is_available():
            parts.extend([
                f"cuda_alloc={torch.cuda.memory_allocated() / 2**20:.0f}MiB",
                f"cuda_reserved={torch.cuda.memory_reserved() / 2**20:.0f}MiB",
            ])
        return " ".join(parts)

    def report(rows_seen: int, *, force: bool = False) -> None:
        nonlocal next_report
        if not force and rows_seen < next_report:
            return
        elapsed = time.monotonic() - started
        rate = rows_seen / elapsed if elapsed > 0 else 0.0
        percent = 100.0 * rows_seen / total_rows if total_rows else 100.0
        print(
            f"[{tag}] progress={rows_seen:,}/{total_rows:,} ({percent:.1f}%) "
            f"elapsed={elapsed / 60:.1f}m rate={rate:,.0f} rows/s {memory_status()}",
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
    optimizer_steps = 0
    examples_seen = 0
    started = time.monotonic()
    progress = load_progress(checkpoint_dir, "A")
    if progress:
        encoder.load_state_dict(progress["encoder_state"])
        probe.load_state_dict(progress["probe_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
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
            optimizer_steps += 1; examples_seen += len(row_ids)
            loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(f"[Stage A/ooc] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f} rows={rows_seen:,}")
        save_progress(checkpoint_dir, "A", {
            "epoch": epoch + 1, "encoder_state": encoder.state_dict(),
            "probe_state": probe.state_dict(), "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
        })
    combined_split_signature = _hash(context.split_signatures)
    metadata = stage_a_metadata(
        config,
        data,
        split_signature=combined_split_signature,
        feature_columns=context.feature_columns,
    )
    metadata["training_summary"] = {
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "epochs_completed": int(config["training"]["epochs_a"]),
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": int(config["training"]["epochs_a"]),
    }
    save_stage_a(
        checkpoint_dir,
        encoder.state_dict(),
        data.class_names,
        metadata=metadata,
    )
    clear_progress(checkpoint_dir, "A")
    return encoder


def _frozen_encoder(config: dict, context: OutOfCoreContext, device) -> SharedEncoder:
    data = context.data; model_cfg = config["model"]
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1], hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"], activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    encoder.load_state_dict(load_validated_stage_a(
        config,
        stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns,
        ),
    )["encoder_state"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def run_stage_b_ooc(config: dict, context: OutOfCoreContext):
    data = context.data; device = torch.device(config["training"]["device"])
    torch.manual_seed(config.get("seed", 0))
    bank_kind = bank_kind_for_architecture(config["architecture"])
    expert_names = expert_names_for_architecture(config["architecture"], data.active_datasets)
    bank = build_expert_bank(
        bank_kind, expert_names, config["model"]["latent_dim"],
        len(data.class_names), config["model"],
    ).to(device)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    warmstart_mode = config["training"].get("stage_b", {}).get("warmstart_mode", "dataset")
    if warmstart_mode == "random_init":
        print(
            "[Stage B/ooc] basic MoE: saving randomly initialized generic experts; "
            "no dataset is assigned to or used to warm-start an expert"
        )
        save_stage_b(
            checkpoint_dir, bank.state_dict(), expert_names, bank_kind,
            training_summary={
                "optimizer_steps": 0, "examples_seen": 0, "epochs_completed": 0,
                "wall_seconds": 0.0, "selected_epoch": 0,
            },
        )
        clear_progress(checkpoint_dir, "B")
        return bank
    if warmstart_mode != "dataset":
        raise ValueError("training.stage_b.warmstart_mode must be 'dataset' or 'random_init'")

    encoder = _frozen_encoder(config, context, device)
    progress = load_progress(checkpoint_dir, "B")
    resume_dataset = 0; resume_epoch = 0; optimizer_state = None
    optimizer_steps = examples_seen = 0
    started = time.monotonic()
    if progress:
        bank.load_state_dict(progress["expert_bank_state"])
        resume_dataset = int(progress["dataset_i"]); resume_epoch = int(progress["epoch"])
        optimizer_state = progress.get("optimizer_state")
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
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
                optimizer_steps += 1; examples_seen += len(row_ids)
                loss_sum += float(loss.item()) * len(row_ids); rows_seen += len(row_ids)
                report_progress(rows_seen)
            report_progress(rows_seen, force=True)
            print(f"[Stage B/ooc:{name}] epoch {epoch + 1}: CE={loss_sum / rows_seen:.6f} rows={rows_seen:,}")
            save_progress(checkpoint_dir, "B", {
                "dataset_i": dataset_i, "epoch": epoch + 1,
                "expert_bank_state": bank.state_dict(), "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            })
        # Mark this expert complete before moving to the next one.
        save_progress(checkpoint_dir, "B", {
            "dataset_i": dataset_i + 1, "epoch": 0,
            "expert_bank_state": bank.state_dict(), "optimizer_state": None,
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
        })
    save_stage_b(
        checkpoint_dir,
        bank.state_dict(),
        expert_names,
        bank_kind,
        training_summary={
            "optimizer_steps": optimizer_steps,
            "examples_seen": examples_seen,
            "epochs_completed": int(config["training"]["epochs_b"]),
            "wall_seconds": time.monotonic() - started,
            "selected_epoch": int(config["training"]["epochs_b"]),
        },
    )
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
    encoder.load_state_dict(load_validated_stage_a(
        config,
        stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns,
        ),
    )["encoder_state"])
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


def _expert_anchor(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.expert_bank.named_parameters()
    }


def _expert_anchor_penalty(model, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    squared_sum = None
    parameter_count = 0
    for name, parameter in model.expert_bank.named_parameters():
        value = (parameter - anchor[name]).square().sum()
        squared_sum = value if squared_sum is None else squared_sum + value
        parameter_count += parameter.numel()
    if squared_sum is None or parameter_count == 0:
        raise ValueError("expert bank has no parameters to anchor")
    return squared_sum / parameter_count


def run_stage_c_ooc(config: dict, context: OutOfCoreContext):
    data = context.data; device = torch.device(config["training"]["device"])
    torch.manual_seed(config.get("seed", 0))
    model = build_ooc_model(config, context, device)
    stage_b_anchor = _expert_anchor(model)
    _set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0),
    )
    weights = _loss_weights(data.train.class_idx, len(data.class_names), device)
    stage_cfg = config["training"]["stage_c"]
    expert_update_policy = stage_cfg.get("expert_update_policy", "all")
    if expert_update_policy not in {"all", "assigned_only"}:
        raise ValueError("training.stage_c.expert_update_policy must be 'all' or 'assigned_only'")
    if expert_update_policy == "assigned_only" and bank_kind_for_architecture(config["architecture"]) != "full":
        raise ValueError("assigned_only expert updates require independent full experts, not a shared adapter head")
    anchor_lambda = float(stage_cfg.get("lambda_expert_anchor", 0.0))
    if anchor_lambda < 0:
        raise ValueError("training.stage_c.lambda_expert_anchor must be non-negative")
    supervision = stage_cfg.get("gate_supervision", "light_aux")
    aux_lambda = _lambda_dataset_aux(config)
    balance_lambda = float(config["load_balance"]["lambda_balance"])
    checkpoint_dir = config["training"]["checkpoint_dir"]
    start_epoch = 0
    best_val_macro_f1 = -float("inf")
    best_model_state = None
    best_epoch = None
    patience_left = int(config["training"].get("early_stopping_patience", 5))
    selection_mode = config["training"].get("selection_mode", "fixed_epochs")
    if selection_mode not in {"fixed_epochs", "best_val"}:
        raise ValueError("training.selection_mode must be fixed_epochs or best_val")
    optimizer_steps = examples_seen = 0
    started = time.monotonic()
    progress = load_progress(checkpoint_dir, "C")
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        best_val_macro_f1 = float(progress.get("best_val_macro_f1", best_val_macro_f1))
        best_model_state = progress.get("best_model_state")
        best_epoch = progress.get("best_epoch")
        patience_left = int(progress.get("patience_left", patience_left))
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    completed_epoch = start_epoch
    for epoch in range(start_epoch, int(config["training"]["epochs_c"])):
        completed_epoch = epoch + 1
        rng = np.random.default_rng(config.get("seed", 0) + 100_000 + epoch)
        totals = {"ce": 0.0, "balance": 0.0, "aux": 0.0, "anchor": 0.0}; rows_seen = 0
        model.train()
        report_progress = _progress_reporter(
            f"Stage C/ooc epoch {epoch + 1}", len(data.train.class_idx), progress_every
        )
        for row_ids in shuffled_row_batches(0, len(data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, dataset_ids = _batch(data.train, row_ids, device)
            output = model(features)
            task_gate_weights = _gate_weights_for_task(output["gate_weights"], supervision)
            ownership_ids = dataset_ids if expert_update_policy == "assigned_only" else None
            if model.routing_mode == "top1":
                training_probs = MoEDatasetNIDS.combine_top1_for_training(
                    task_gate_weights,
                    output["selected_probs"],
                    output["selected_experts"],
                    ownership_ids,
                    expert_update_policy,
                )
            else:
                training_probs = MoEDatasetNIDS.combine_probs_for_training(
                    task_gate_weights, output["expert_probs"], ownership_ids, expert_update_policy
                )
            ce = F.nll_loss(MoEDatasetNIDS.combined_probs_to_log_probs(training_probs), labels, weight=weights)
            balance = load_balance_penalty(output["gate_weights"])
            aux = dataset_aux_loss(output["gate_weights"], dataset_ids) if aux_lambda > 0 else ce.new_zeros(())
            anchor = _expert_anchor_penalty(model, stage_b_anchor) if anchor_lambda > 0 else ce.new_zeros(())
            loss = ce + balance_lambda * balance + aux_lambda * aux + anchor_lambda * anchor
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            optimizer_steps += 1; examples_seen += len(row_ids)
            amount = len(row_ids); rows_seen += amount
            totals["ce"] += float(ce.item()) * amount
            totals["balance"] += float(balance.item()) * amount
            totals["aux"] += float(aux.item()) * amount
            totals["anchor"] += float(anchor.item()) * amount
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(
            f"[Stage C/ooc] epoch {epoch + 1}: CE={totals['ce']/rows_seen:.6f} "
            f"balance={totals['balance']/rows_seen:.6f} aux={totals['aux']/rows_seen:.6f} "
            f"anchor={totals['anchor']/rows_seen:.6f} policy={expert_update_policy} "
            f"gate_supervision={supervision} rows={rows_seen:,}"
        )
        if selection_mode == "best_val":
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
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
        })
        if selection_mode == "best_val" and patience_left <= 0:
            print(f"[Stage C/ooc] early stopping after epoch {epoch + 1}")
            break
    if selection_mode == "best_val" and best_model_state is not None:
        model.load_state_dict(best_model_state)
    bank_kind = bank_kind_for_architecture(config["architecture"])
    selected_epoch = best_epoch if selection_mode == "best_val" else completed_epoch
    summary = {
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "epochs_completed": completed_epoch,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": selected_epoch,
    }
    save_stage_c(
        checkpoint_dir, model.state_dict(), data.class_names, data.active_datasets,
        bank_kind, training_summary=summary,
    )
    summary_path = os.path.join(checkpoint_dir, STAGE_C_SUMMARY_FILE)
    temporary = summary_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(
            {
                "best_epoch": selected_epoch,
                "best_validation_macro_f1": best_val_macro_f1 if selection_mode == "best_val" else None,
                "selection_mode": selection_mode,
                "training_summary": summary,
            },
            handle, indent=2, sort_keys=True,
        )
    os.replace(temporary, summary_path)
    clear_progress(checkpoint_dir, "C")
    model.training_summary = {"C": summary}
    return model
