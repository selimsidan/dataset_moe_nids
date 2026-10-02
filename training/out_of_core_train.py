"""Bounded-memory Stage A/B/C training for disk-backed full NF-v3 data."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.encoder import SharedEncoder, build_encoder, resolve_encoder_config
from models.losses import dataset_aux_loss, load_balance_penalty
from models.moe import ClassConditionalMoEDatasetNIDS, MoEDatasetNIDS
from models.representation_losses import (
    StageARepresentationObjective,
    representation_config,
    stage_a_encoder_role,
)
from evaluation.validation_report import (
    write_owned_expert_validation_ooc,
    write_validation_report_ooc,
)

from .checkpoint import (
    STAGE_A_FILE,
    STAGE_B_FILE,
    STAGE_C_FILE,
    clear_progress,
    load_progress,
    load_configured_stage_b,
    load_validated_stage_a,
    save_progress,
    save_stage_a,
    save_stage_b,
    save_stage_c,
    stage_a_metadata,
)
from .model_utils import (
    bank_kind_for_architecture,
    apply_stage_c_trainability,
    build_expert_bank,
    build_model,
    expert_forward_one,
    expert_names_for_architecture,
    expert_train_params,
    initialize_global_residual_head,
    initialize_private_expert_encoders,
)
from .out_of_core_data import OutOfCoreContext, class_counts
from .stage_c_jointfinetune import _gate_weights_for_task, _lambda_dataset_aux
from .sampler import class_domain_balanced_row_batches
from .optim import optimizer_hparams
from .pooled_replay import (
    build_stratified_replay_reservoir,
    draw_class_balanced_replay_rows,
    replay_rows_for_owned_batch,
)
from .training_history import append_training_history

CONTRACT_FILE = "run_contract.json"
STAGE_C_SUMMARY_FILE = "stage_c_training_summary.json"
REPRESENTATION_COVERAGE_FILE = "Representation_Batch_Coverage.csv"
ORCHESTRATION_TRAINING_KEYS = {
    "checkpoint_dir",
    "device",
    "force_restart",
    "progress_every_rows",
    "run_final_evaluation",
    "save_epoch_history",
    "stages",
}


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _normalize_contract(contract: dict) -> dict:
    """Remove execution scheduling fields from old and new run contracts."""
    normalized = copy.deepcopy(contract)
    normalized.pop("signature", None)
    training = normalized.get("training", {})
    for key in ORCHESTRATION_TRAINING_KEYS:
        training.pop(key, None)
    normalized["signature"] = _hash(normalized)
    return normalized


def ensure_run_contract(config: dict, context: OutOfCoreContext) -> dict:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    training_contract = {
        key: copy.deepcopy(value) for key, value in config["training"].items()
        if key not in ORCHESTRATION_TRAINING_KEYS
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
    representation_contract = training_contract.get("representation", {})
    neutral_representation = {
        "objective": "ce",
        "sampling": "legacy",
        "class_weighting": "legacy",
        "weight": 0.1,
        "temperature": 0.1,
        "center_weight": 0.01,
        "arc_margin": 0.3,
        "arc_scale": 30.0,
    }
    if all(representation_contract.get(key, value) == value for key, value in neutral_representation.items()):
        training_contract.pop("representation", None)
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
        previous = _normalize_contract(previous)
        if previous != contract:
            completed = [name for name in (STAGE_A_FILE, STAGE_B_FILE, STAGE_C_FILE) if os.path.isfile(os.path.join(checkpoint_dir, name))]
            raise ValueError(
                f"Run contract changed for checkpoint directory {checkpoint_dir}; existing stages={completed}. "
                "Use a new RUN_NAME or deliberately set FORCE_RESTART=True."
            )
        # Migrate historical contracts that signed the stage schedule. The
        # normalized contract is scientifically identical and permits A/B/C
        # to be launched as separate resumable processes.
        with open(path + ".tmp", "w") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
        os.replace(path + ".tmp", path)
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


def _empty_representation_coverage(num_classes: int) -> dict[str, np.ndarray | int]:
    return {
        "batch_appearances": np.zeros(num_classes, dtype=np.int64),
        "singleton_batches": np.zeros(num_classes, dtype=np.int64),
        "anchors_seen": np.zeros(num_classes, dtype=np.int64),
        "anchors_with_positive": np.zeros(num_classes, dtype=np.int64),
        "batches_seen": 0,
    }


def _restore_representation_coverage(value: dict | None, num_classes: int) -> dict[str, np.ndarray | int]:
    coverage = _empty_representation_coverage(num_classes)
    if not value:
        return coverage
    for key in ("batch_appearances", "singleton_batches", "anchors_seen", "anchors_with_positive"):
        restored = np.asarray(value.get(key, coverage[key]), dtype=np.int64)
        if restored.shape != (num_classes,):
            raise ValueError(f"Stage-A representation coverage field {key!r} has shape {restored.shape}")
        coverage[key] = restored
    coverage["batches_seen"] = int(value.get("batches_seen", 0))
    return coverage


def _update_representation_coverage(coverage: dict, labels: torch.Tensor, num_classes: int) -> None:
    counts = torch.bincount(labels.detach(), minlength=num_classes).cpu().numpy().astype(np.int64)
    present = counts > 0
    valid = counts >= 2
    coverage["batch_appearances"] += present
    coverage["singleton_batches"] += counts == 1
    coverage["anchors_seen"] += counts
    coverage["anchors_with_positive"] += np.where(valid, counts, 0)
    coverage["batches_seen"] += 1


def _coverage_checkpoint_value(coverage: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else int(value)
        for key, value in coverage.items()
    }


def _write_representation_coverage(checkpoint_dir: str, class_names: list[str], coverage: dict) -> str:
    rows = []
    for index, class_name in enumerate(class_names):
        appearances = int(coverage["batch_appearances"][index])
        singleton = int(coverage["singleton_batches"][index])
        anchors = int(coverage["anchors_seen"][index])
        valid = int(coverage["anchors_with_positive"][index])
        rows.append({
            "class": class_name,
            "batches_seen": int(coverage["batches_seen"]),
            "batch_appearances": appearances,
            "singleton_batches": singleton,
            "singleton_fraction_when_present": singleton / appearances if appearances else 0.0,
            "anchors_seen": anchors,
            "anchors_with_same_class_positive": valid,
            "valid_anchor_fraction": valid / anchors if anchors else 0.0,
        })
    anchors = int(np.asarray(coverage["anchors_seen"]).sum())
    valid = int(np.asarray(coverage["anchors_with_positive"]).sum())
    rows.append({
        "class": "__ALL__",
        "batches_seen": int(coverage["batches_seen"]),
        "batch_appearances": int(np.asarray(coverage["batch_appearances"]).sum()),
        "singleton_batches": int(np.asarray(coverage["singleton_batches"]).sum()),
        "singleton_fraction_when_present": (
            float(np.asarray(coverage["singleton_batches"]).sum())
            / max(1, int(np.asarray(coverage["batch_appearances"]).sum()))
        ),
        "anchors_seen": anchors,
        "anchors_with_same_class_positive": valid,
        "valid_anchor_fraction": valid / anchors if anchors else 0.0,
    })
    path = os.path.join(checkpoint_dir, REPRESENTATION_COVERAGE_FILE)
    temporary = path + ".tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


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


def _expert_optimizer(params, config: dict) -> torch.optim.Optimizer:
    lr, weight_decay = optimizer_hparams(config, "b")
    return torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=weight_decay,
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
    encoder_role = stage_a_encoder_role(config)
    encoder = build_encoder(
        data.train.features.shape[1], model_cfg["latent_dim"],
        resolve_encoder_config(model_cfg, encoder_role),
    ).to(device)
    objective = StageARepresentationObjective(
        model_cfg["latent_dim"], len(data.class_names), config, encoder_role
    ).to(device)
    lr, weight_decay = optimizer_hparams(config, "a")
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *objective.parameters()], lr=lr,
        weight_decay=weight_decay,
    )
    representation = representation_config(config, encoder_role)
    if representation["class_weighting"] == "legacy":
        weights = _loss_weights(data.train.class_idx, len(data.class_names), device)
    elif representation["class_weighting"] == "none":
        weights = None
    else:
        raise ValueError("training.representation.class_weighting must be legacy or none")
    checkpoint_dir = config["training"]["checkpoint_dir"]
    start_epoch = 0
    optimizer_steps = 0
    examples_seen = 0
    coverage = _empty_representation_coverage(len(data.class_names))
    started = time.monotonic()
    progress = load_progress(checkpoint_dir, "A")
    if progress:
        encoder.load_state_dict(progress["encoder_state"])
        if "objective_state" in progress:
            objective.load_state_dict(progress["objective_state"])
        else:
            objective.classifier.load_state_dict(progress["probe_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
        coverage = _restore_representation_coverage(
            progress.get("representation_coverage"), len(data.class_names)
        )
        print(f"[Stage A/ooc] resuming at epoch {start_epoch}")
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for epoch in range(start_epoch, int(config["training"]["epochs_a"])):
        epoch_started = time.monotonic()
        rng = np.random.default_rng(config.get("seed", 0) + epoch)
        encoder.train(); objective.train()
        totals = {"ce": 0.0, "metric": 0.0, "total": 0.0}; rows_seen = 0
        report_progress = _progress_reporter(
            f"Stage A/ooc epoch {epoch + 1}", len(data.train.class_idx), progress_every
        )
        sampling = representation["sampling"]
        if sampling == "legacy":
            batches = shuffled_row_batches(
                0, len(data.train.class_idx), rng, batch_size, block_rows, buffer_blocks
            )
        elif sampling == "class_domain_balanced":
            batches = class_domain_balanced_row_batches(
                data.train.class_idx,
                data.train.dataset_idx,
                batch_size,
                int(config["training"].get("min_per_class_per_batch", 4)),
                seed=config.get("seed", 0) + epoch,
            )
        else:
            raise ValueError("training.representation.sampling must be legacy or class_domain_balanced")
        for row_ids in batches:
            features, labels, _ = _batch(data.train, row_ids, device)
            _update_representation_coverage(coverage, labels, len(data.class_names))
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _logits = objective(encoder(features), labels, weights)
            loss.backward(); optimizer.step()
            optimizer_steps += 1; examples_seen += len(row_ids)
            for name in totals:
                totals[name] += float(parts[name].item()) * len(row_ids)
            rows_seen += len(row_ids)
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(
            f"[Stage A/ooc] epoch {epoch + 1}: objective={objective.objective} "
            f"CE={totals['ce'] / rows_seen:.6f} metric={totals['metric'] / rows_seen:.6f} "
            f"total={totals['total'] / rows_seen:.6f} rows={rows_seen:,}"
        )
        append_training_history(config, {
            "stage": "A", "encoder_role": encoder_role, "dataset": "ALL",
            "epoch": epoch + 1, "objective": objective.objective,
            "learning_rate": optimizer.param_groups[0]["lr"], "rows": rows_seen,
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            "epoch_seconds": time.monotonic() - epoch_started,
            "train_ce_loss": totals["ce"] / rows_seen,
            "train_representation_loss": totals["metric"] / rows_seen,
            "train_total_loss": totals["total"] / rows_seen,
        })
        save_progress(checkpoint_dir, "A", {
            "epoch": epoch + 1, "encoder_state": encoder.state_dict(),
            "objective_state": objective.state_dict(), "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            "representation_coverage": _coverage_checkpoint_value(coverage),
        })
    combined_split_signature = _hash(context.split_signatures)
    metadata = stage_a_metadata(
        config,
        data,
        split_signature=combined_split_signature,
        feature_columns=context.feature_columns,
        encoder_role=encoder_role,
    )
    metadata["training_summary"] = {
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "epochs_completed": int(config["training"]["epochs_a"]),
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": int(config["training"]["epochs_a"]),
    }
    _write_representation_coverage(checkpoint_dir, data.class_names, coverage)
    anchors_seen = int(np.asarray(coverage["anchors_seen"]).sum())
    valid_anchors = int(np.asarray(coverage["anchors_with_positive"]).sum())
    metadata["training_summary"]["representation_valid_anchor_fraction"] = (
        valid_anchors / anchors_seen if anchors_seen else 0.0
    )
    metadata["representation_coverage_file"] = REPRESENTATION_COVERAGE_FILE
    save_stage_a(
        checkpoint_dir,
        encoder.state_dict(),
        data.class_names,
        metadata=metadata,
        representation_state=objective.state_dict(),
        representation_config=representation,
    )
    clear_progress(checkpoint_dir, "A")
    return encoder


def _frozen_encoder(
    config: dict, context: OutOfCoreContext, device, role: str = "encoder"
) -> SharedEncoder:
    data = context.data; model_cfg = config["model"]
    encoder = build_encoder(
        data.train.features.shape[1], model_cfg["latent_dim"],
        resolve_encoder_config(model_cfg, role),
    ).to(device)
    encoder.load_state_dict(load_validated_stage_a(
        config,
        stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns, encoder_role=role,
        ),
        role,
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
        len(data.class_names), config["model"], input_dim=data.train.features.shape[1],
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

    stage_b_cfg = config["training"].get("stage_b", {})
    replay_fraction = float(stage_b_cfg.get("replay_fraction", 0.0))
    replay_rows_for_owned_batch(1, replay_fraction)  # validates the fraction
    replay_pools: dict[int, np.ndarray] = {}
    pooled_weights = None
    if replay_fraction > 0:
        replay_pools = build_stratified_replay_reservoir(
            data.train.class_idx,
            data.train.dataset_idx,
            max_per_class_dataset=int(
                stage_b_cfg.get("replay_pool_per_class_dataset", 4096)
            ),
            seed=config.get("seed", 0) + 700_000,
            chunk_rows=int(config["training"].get("shuffle_block_rows", 65_536)),
        )
        pooled_weights = _loss_weights(
            data.train.class_idx, len(data.class_names), device
        )
        print(
            f"[Stage B/ooc] pooled replay enabled: fraction={replay_fraction:.3f} "
            f"reservoir_rows={sum(len(rows) for rows in replay_pools.values()):,} "
            f"classes={len(replay_pools)}",
            flush=True,
        )

    primary_role = "gate_encoder" if bank_kind == "private_encoder" else "encoder"
    encoder = _frozen_encoder(config, context, device, primary_role)
    if bank_kind == "global_residual":
        stage_a = load_validated_stage_a(
            config,
            stage_a_metadata(
                config, data, split_signature=_hash(context.split_signatures),
                feature_columns=context.feature_columns, encoder_role=primary_role,
            ),
            primary_role,
        )
        initialize_global_residual_head(bank, stage_a)
    if bank_kind == "private_encoder":
        private_source = _frozen_encoder(config, context, device, "private_encoder")
        initialize_private_expert_encoders(bank, private_source)
    progress = load_progress(checkpoint_dir, "B")
    resume_dataset = 0; resume_epoch = 0; optimizer_state = None
    optimizer_steps = examples_seen = 0
    owned_examples_seen = replay_examples_seen = 0
    started = time.monotonic()
    if progress:
        bank.load_state_dict(progress["expert_bank_state"])
        resume_dataset = int(progress["dataset_i"]); resume_epoch = int(progress["epoch"])
        optimizer_state = progress.get("optimizer_state")
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
        owned_examples_seen = int(progress.get("owned_examples_seen", 0))
        replay_examples_seen = int(progress.get("replay_examples_seen", 0))
    batch_size, block_rows, buffer_blocks = _settings(config)
    progress_every = int(config["training"].get("progress_every_rows", 1_000_000))
    for dataset_i, name in enumerate(data.active_datasets):
        if dataset_i < resume_dataset:
            continue
        bounds = data.train.dataset_slices[name]
        local_labels = data.train.class_idx[bounds]
        weights = _loss_weights(local_labels, len(data.class_names), device)
        optimizer = _expert_optimizer(expert_train_params(bank, dataset_i), config)
        first_epoch = resume_epoch if dataset_i == resume_dataset else 0
        if dataset_i == resume_dataset and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            epoch_started = time.monotonic()
            rng = np.random.default_rng(config.get("seed", 0) + dataset_i * 10_000 + epoch)
            loss_sum = owned_loss_sum = replay_loss_sum = 0.0
            rows_seen = replay_rows_seen = 0
            total_rows = bounds.stop - bounds.start
            report_progress = _progress_reporter(
                f"Stage B/ooc:{name} epoch {epoch + 1}", total_rows, progress_every
            )
            for row_ids in shuffled_row_batches(bounds.start, bounds.stop, rng, batch_size, block_rows, buffer_blocks):
                features, labels, _ = _batch(data.train, row_ids, device)
                if getattr(bank, "expects_raw_input", False):
                    expert_input = features
                else:
                    with torch.no_grad():
                        expert_input = encoder(features)
                optimizer.zero_grad(set_to_none=True)
                owned_loss = F.cross_entropy(
                    expert_forward_one(bank, dataset_i, expert_input), labels, weight=weights
                )
                replay_count = replay_rows_for_owned_batch(len(row_ids), replay_fraction)
                if replay_count:
                    replay_rng = np.random.default_rng(
                        config.get("seed", 0) + dataset_i * 10_000_000
                        + epoch * 100_000 + rows_seen
                    )
                    replay_ids = draw_class_balanced_replay_rows(
                        replay_pools, replay_count, replay_rng
                    )
                    replay_features, replay_labels, _ = _batch(
                        data.train, replay_ids, device
                    )
                    if getattr(bank, "expects_raw_input", False):
                        replay_input = replay_features
                    else:
                        with torch.no_grad():
                            replay_input = encoder(replay_features)
                    replay_loss = F.cross_entropy(
                        expert_forward_one(bank, dataset_i, replay_input),
                        replay_labels,
                        weight=pooled_weights,
                    )
                    loss = (
                        (1.0 - replay_fraction) * owned_loss
                        + replay_fraction * replay_loss
                    )
                else:
                    replay_loss = owned_loss.new_zeros(())
                    loss = owned_loss
                loss.backward(); optimizer.step()
                optimizer_steps += 1
                examples_seen += len(row_ids) + replay_count
                owned_examples_seen += len(row_ids)
                replay_examples_seen += replay_count
                loss_sum += float(loss.item()) * len(row_ids)
                owned_loss_sum += float(owned_loss.item()) * len(row_ids)
                replay_loss_sum += float(replay_loss.item()) * replay_count
                rows_seen += len(row_ids)
                replay_rows_seen += replay_count
                report_progress(rows_seen)
            report_progress(rows_seen, force=True)
            print(
                f"[Stage B/ooc:{name}] epoch {epoch + 1}: "
                f"CE={loss_sum / rows_seen:.6f} "
                f"owned_CE={owned_loss_sum / rows_seen:.6f} "
                f"replay_CE={replay_loss_sum / max(1, replay_rows_seen):.6f} "
                f"owned_rows={rows_seen:,} replay_rows={replay_rows_seen:,}"
            )
            append_training_history(config, {
                "stage": "B",
                "encoder_role": "private_expert" if bank_kind == "private_encoder" else "expert",
                "dataset": name, "epoch": epoch + 1, "objective": "ce",
                "learning_rate": optimizer.param_groups[0]["lr"],
                "rows": rows_seen + replay_rows_seen,
                "replay_rows": replay_rows_seen,
                "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
                "epoch_seconds": time.monotonic() - epoch_started,
                "train_ce_loss": loss_sum / rows_seen,
                "train_owned_ce_loss": owned_loss_sum / rows_seen,
                "train_replay_ce_loss": (
                    replay_loss_sum / replay_rows_seen if replay_rows_seen else np.nan
                ),
                "train_total_loss": loss_sum / rows_seen,
            })
            save_progress(checkpoint_dir, "B", {
                "dataset_i": dataset_i, "epoch": epoch + 1,
                "expert_bank_state": bank.state_dict(), "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
                "owned_examples_seen": owned_examples_seen,
                "replay_examples_seen": replay_examples_seen,
            })
        # Mark this expert complete before moving to the next one.
        save_progress(checkpoint_dir, "B", {
            "dataset_i": dataset_i + 1, "epoch": 0,
            "expert_bank_state": bank.state_dict(), "optimizer_state": None,
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            "owned_examples_seen": owned_examples_seen,
            "replay_examples_seen": replay_examples_seen,
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
            "replay_fraction": replay_fraction,
            "owned_examples_seen": owned_examples_seen,
            "replay_examples_seen": replay_examples_seen,
            "replay_pool_rows": sum(len(rows) for rows in replay_pools.values()),
        },
    )
    write_owned_expert_validation_ooc(
        encoder, bank, context, config,
        os.path.join(checkpoint_dir, "Validation_Expert_Owned_StageB.csv"),
    )
    clear_progress(checkpoint_dir, "B")
    return bank

def build_ooc_model(config: dict, context: OutOfCoreContext, device) -> torch.nn.Module:
    data = context.data; model_cfg = config["model"]
    role = "gate_encoder" if config["architecture"] == "moe_dataset_private_encoders" else "encoder"
    encoder = build_encoder(
        data.train.features.shape[1], model_cfg["latent_dim"],
        resolve_encoder_config(model_cfg, role),
    ).to(device)
    encoder.load_state_dict(load_validated_stage_a(
        config,
        stage_a_metadata(
            config, data, split_signature=_hash(context.split_signatures),
            feature_columns=context.feature_columns, encoder_role=role,
        ),
        role,
    )["encoder_state"])
    model = build_model(config["architecture"], encoder, data.active_datasets, data.class_names, model_cfg).to(device)
    stage_b = load_configured_stage_b(
        config,
        expected_dataset_names=list(model.expert_bank.dataset_names),
        expected_bank_kind=bank_kind_for_architecture(config["architecture"]),
    )
    model.expert_bank.load_state_dict(stage_b["expert_bank_state"])
    apply_stage_c_trainability(model, config)
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
    apply_stage_c_trainability(model, config)
    lr, weight_decay = optimizer_hparams(config, "c")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    weights = _loss_weights(data.train.class_idx, len(data.class_names), device)
    stage_cfg = config["training"]["stage_c"]
    expert_update_policy = stage_cfg.get("expert_update_policy", "all")
    if expert_update_policy not in {"all", "assigned_only"}:
        raise ValueError("training.stage_c.expert_update_policy must be 'all' or 'assigned_only'")
    if expert_update_policy == "assigned_only" and bank_kind_for_architecture(config["architecture"]) not in {"full", "private_encoder", "global_residual"}:
        raise ValueError("assigned_only expert updates require independent full experts, not a shared adapter head")
    anchor_lambda = float(stage_cfg.get("lambda_expert_anchor", 0.0))
    if anchor_lambda < 0:
        raise ValueError("training.stage_c.lambda_expert_anchor must be non-negative")
    reliability_lambda = float(stage_cfg.get("lambda_reliability", 0.0))
    if reliability_lambda < 0:
        raise ValueError("training.stage_c.lambda_reliability must be non-negative")
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
        epoch_started = time.monotonic()
        completed_epoch = epoch + 1
        rng = np.random.default_rng(config.get("seed", 0) + 100_000 + epoch)
        totals = {
            "ce": 0.0, "balance": 0.0, "aux": 0.0,
            "anchor": 0.0, "reliability": 0.0,
        }; rows_seen = 0
        model.train()
        report_progress = _progress_reporter(
            f"Stage C/ooc epoch {epoch + 1}", len(data.train.class_idx), progress_every
        )
        for row_ids in shuffled_row_batches(0, len(data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, dataset_ids = _batch(data.train, row_ids, device)
            output = model(features)
            task_gate_weights = _gate_weights_for_task(output["gate_weights"], supervision)
            ownership_ids = dataset_ids if expert_update_policy == "assigned_only" else None
            if isinstance(model, ClassConditionalMoEDatasetNIDS):
                class_gate_weights = model.class_gate_weights(task_gate_weights)
                training_probs = model.combine_class_conditional_probs_for_training(
                    class_gate_weights,
                    output["expert_probs"],
                    ownership_ids,
                    expert_update_policy,
                )
            elif model.routing_mode == "top1":
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
            reliability = (
                model.class_reliability.square().mean()
                if isinstance(model, ClassConditionalMoEDatasetNIDS)
                else ce.new_zeros(())
            )
            loss = (
                ce + balance_lambda * balance + aux_lambda * aux
                + anchor_lambda * anchor + reliability_lambda * reliability
            )
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            optimizer_steps += 1; examples_seen += len(row_ids)
            amount = len(row_ids); rows_seen += amount
            totals["ce"] += float(ce.item()) * amount
            totals["balance"] += float(balance.item()) * amount
            totals["aux"] += float(aux.item()) * amount
            totals["anchor"] += float(anchor.item()) * amount
            totals["reliability"] += float(reliability.item()) * amount
            report_progress(rows_seen)
        report_progress(rows_seen, force=True)
        print(
            f"[Stage C/ooc] epoch {epoch + 1}: CE={totals['ce']/rows_seen:.6f} "
            f"balance={totals['balance']/rows_seen:.6f} aux={totals['aux']/rows_seen:.6f} "
            f"anchor={totals['anchor']/rows_seen:.6f} policy={expert_update_policy} "
            f"reliability={totals['reliability']/rows_seen:.6f} "
            f"gate_supervision={supervision} rows={rows_seen:,}"
        )
        val_macro_f1 = np.nan
        improved: bool | float = np.nan
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
        average = {name: value / rows_seen for name, value in totals.items()}
        append_training_history(config, {
            "stage": "C", "encoder_role": "full_model", "dataset": "ALL",
            "epoch": epoch + 1, "objective": "joint",
            "learning_rate": optimizer.param_groups[0]["lr"], "rows": rows_seen,
            "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
            "epoch_seconds": time.monotonic() - epoch_started,
            "train_ce_loss": average["ce"],
            "train_balance_penalty": average["balance"],
            "train_dataset_aux_loss": average["aux"],
            "train_anchor_penalty": average["anchor"],
            "train_reliability_penalty": average["reliability"],
            "train_total_loss": (
                average["ce"] + balance_lambda * average["balance"]
                + aux_lambda * average["aux"] + anchor_lambda * average["anchor"]
                + reliability_lambda * average["reliability"]
            ),
            "val_macro_f1": val_macro_f1,
            "best_val_macro_f1": (
                best_val_macro_f1 if selection_mode == "best_val" else np.nan
            ),
            "improved": improved,
            "patience_left": patience_left if selection_mode == "best_val" else np.nan,
        })
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
    write_validation_report_ooc(model, context, config)
    write_owned_expert_validation_ooc(
        model.encoder, model.expert_bank, context, config,
        os.path.join(config["evaluation"]["output_dir"], "Validation_Expert_Owned_StageC.csv"),
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
