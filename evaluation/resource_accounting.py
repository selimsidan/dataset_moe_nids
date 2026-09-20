"""Architecture-aware parameter, inference-compute, and training-budget accounting.

Forward MACs count Linear-layer weight multiplications per sample.  The report
uses the explicit convention ``FLOPs = 2 * MACs`` (one multiply plus one add)
and excludes nonlinearities, dropout, indexing, and softmax from both values.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

import pandas as pd
import torch
from torch import nn

from models.adapters import AdapterExpertBank
from models.baselines import HardTwoStageModel, MatchedDenseClassifier, NoFusionModel
from models.moe import MoEDatasetNIDS
from models.private_encoder_experts import PrivateEncoderExpertBank


RESOURCE_FILE = "Resource_Accounting.csv"
FLOP_CONVENTION = "linear_only; one_multiply_plus_one_add=2_FLOPs"


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def linear_macs(module: nn.Module) -> int:
    return sum(layer.weight.numel() for layer in module.modules() if isinstance(layer, nn.Linear))


def _stats(values: Iterable[int], prefix: str) -> dict[str, float | int]:
    items = list(values)
    if not items:
        items = [0]
    return {
        f"{prefix}_min": min(items),
        f"{prefix}_mean": float(sum(items) / len(items)),
        f"{prefix}_max": max(items),
    }


def stage_a_checkpoint_hash(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_columns(summary: dict | None) -> dict:
    columns: dict[str, int | float | None] = {}
    for stage in ("A", "B", "C", "hard_router", "hard_classifiers"):
        values = (summary or {}).get(stage, {})
        columns.update(
            {
                f"{stage}_optimizer_steps": values.get("optimizer_steps", 0),
                f"{stage}_examples_seen": values.get("examples_seen", 0),
                f"{stage}_epochs_completed": values.get("epochs_completed", 0),
                f"{stage}_wall_seconds": values.get("wall_seconds", 0.0),
                f"{stage}_selected_epoch": values.get("selected_epoch"),
            }
        )
    columns["total_optimizer_steps"] = sum(
        int(columns[f"{stage}_optimizer_steps"]) for stage in ("A", "B", "C", "hard_router", "hard_classifiers")
    )
    columns["total_examples_seen"] = sum(
        int(columns[f"{stage}_examples_seen"]) for stage in ("A", "B", "C", "hard_router", "hard_classifiers")
    )
    columns["total_wall_seconds"] = sum(
        float(columns[f"{stage}_wall_seconds"]) for stage in ("A", "B", "C", "hard_router", "hard_classifiers")
    )
    return columns


def resource_profile(
    model: nn.Module,
    config: dict,
    *,
    trial_id: str | None = None,
    split_signature: str | None = None,
    checkpoint_hash: str | None = None,
    training_summary: dict | None = None,
) -> dict:
    """Return one report-ready row with architecture-specific active costs."""
    encoder_params = gate_params = classifier_params = 0
    encoder_macs = gate_macs = classifier_macs = 0
    active_params: list[int]
    active_macs: list[int]

    if isinstance(model, MoEDatasetNIDS):
        gate_params = parameter_count(model.gate)
        gate_macs = linear_macs(model.gate)
        if isinstance(model.expert_bank, PrivateEncoderExpertBank):
            encoder_params = parameter_count(model.encoder) + sum(
                parameter_count(expert.encoder) for expert in model.expert_bank.experts
            )
            encoder_macs = linear_macs(model.encoder) + sum(
                linear_macs(expert.encoder) for expert in model.expert_bank.experts
            )
            classifier_params = sum(
                parameter_count(expert.head) for expert in model.expert_bank.experts
            )
            classifier_macs = sum(
                linear_macs(expert.head) for expert in model.expert_bank.experts
            )
            branch_p = [parameter_count(expert) for expert in model.expert_bank.experts]
            branch_m = [linear_macs(expert) for expert in model.expert_bank.experts]
        elif isinstance(model.expert_bank, AdapterExpertBank):
            encoder_params, encoder_macs = parameter_count(model.encoder), linear_macs(model.encoder)
            classifier_params = parameter_count(model.expert_bank)
            classifier_macs = linear_macs(model.expert_bank)
            shared_p = parameter_count(model.expert_bank.shared_head)
            shared_m = linear_macs(model.expert_bank.shared_head)
            branch_p = [parameter_count(adapter) + shared_p for adapter in model.expert_bank.adapters]
            branch_m = [linear_macs(adapter) + shared_m for adapter in model.expert_bank.adapters]
        else:
            encoder_params, encoder_macs = parameter_count(model.encoder), linear_macs(model.encoder)
            classifier_params = parameter_count(model.expert_bank)
            classifier_macs = linear_macs(model.expert_bank)
            branch_p = [parameter_count(expert) for expert in model.expert_bank.experts]
            branch_m = [linear_macs(expert) for expert in model.expert_bank.experts]
        if model.routing_mode == "dense":
            active_params = [parameter_count(model)]
            active_macs = [encoder_macs + gate_macs + classifier_macs]
        else:
            gate_encoder_params = parameter_count(model.encoder)
            gate_encoder_macs = linear_macs(model.encoder)
            active_params = [gate_encoder_params + gate_params + value for value in branch_p]
            active_macs = [gate_encoder_macs + gate_macs + value for value in branch_m]
    elif isinstance(model, HardTwoStageModel):
        router_params = parameter_count(model.id_encoder) + parameter_count(model.id_head)
        router_macs = linear_macs(model.id_encoder) + linear_macs(model.id_head)
        encoder_params = parameter_count(model.id_encoder) + sum(
            parameter_count(value) for value in model.stage_b.encoders.values()
        )
        encoder_macs = linear_macs(model.id_encoder) + sum(
            linear_macs(value) for value in model.stage_b.encoders.values()
        )
        gate_params, gate_macs = parameter_count(model.id_head), linear_macs(model.id_head)
        classifier_params = sum(parameter_count(value) for value in model.stage_b.heads.values())
        classifier_macs = sum(linear_macs(value) for value in model.stage_b.heads.values())
        active_params = [
            router_params + parameter_count(model.stage_b.encoders[name]) + parameter_count(model.stage_b.heads[name])
            for name in model.dataset_names
        ]
        active_macs = [
            router_macs + linear_macs(model.stage_b.encoders[name]) + linear_macs(model.stage_b.heads[name])
            for name in model.dataset_names
        ]
    elif isinstance(model, NoFusionModel):
        encoder_params = sum(parameter_count(value) for value in model.encoders.values())
        encoder_macs = sum(linear_macs(value) for value in model.encoders.values())
        classifier_params = sum(parameter_count(value) for value in model.heads.values())
        classifier_macs = sum(linear_macs(value) for value in model.heads.values())
        active_params = [parameter_count(model.encoders[name]) + parameter_count(model.heads[name]) for name in model.dataset_names]
        active_macs = [linear_macs(model.encoders[name]) + linear_macs(model.heads[name]) for name in model.dataset_names]
    else:
        encoder = getattr(model, "encoder", None)
        head = getattr(model, "head", None)
        encoder_params = parameter_count(encoder) if encoder is not None else 0
        encoder_macs = linear_macs(encoder) if encoder is not None else 0
        classifier_params = parameter_count(head) if head is not None else parameter_count(model) - encoder_params
        classifier_macs = linear_macs(head) if head is not None else linear_macs(model) - encoder_macs
        active_params = [parameter_count(model)]
        active_macs = [linear_macs(model)]

    is_baseline = config.get("architecture") in {"plain_pooled", "matched_dense", "no_fusion", "hard_two_stage"}
    baseline_cfg = config.get("training", {}).get("baseline", {}) if is_baseline else {}
    match_info = model.match_info if isinstance(model, MatchedDenseClassifier) else {}
    row = {
        "Trial_ID": trial_id,
        "run_name": config.get("run_name"),
        "architecture": config.get("architecture"),
        "architecture_role": "oracle_dataset_id" if isinstance(model, NoFusionModel) else "dataset_blind",
        "routing_mode": getattr(model, "routing_mode", config.get("model", {}).get("gate", {}).get("routing", "dense")),
        "matching_axis": match_info.get("axis"),
        "matching_target": match_info.get("target"),
        "matching_achieved": match_info.get("achieved"),
        "matching_residual": match_info.get("residual"),
        "matching_relative_error": match_info.get("relative_error"),
        "resolved_dense_hidden_dims": json.dumps(match_info.get("hidden_dims")) if match_info else None,
        "encoder_init": baseline_cfg.get("encoder_init"),
        "stage_b_warmstart": baseline_cfg.get("stage_b_warmstart"),
        "selection_mode": config.get("training", {}).get("selection_mode", "fixed_epochs"),
        "seed": config.get("seed", 0),
        "split_signature": split_signature,
        "stage_a_checkpoint_sha256": checkpoint_hash,
        "total_parameters": parameter_count(model),
        "trainable_parameters": trainable_parameter_count(model),
        "encoder_parameters": encoder_params,
        "gate_router_parameters": gate_params,
        "classifier_expert_parameters": classifier_params,
        "stored_linear_macs": linear_macs(model),
        "flop_convention": FLOP_CONVENTION,
        **_stats(active_params, "active_parameters_per_sample"),
        **_stats(active_macs, "forward_macs_per_sample"),
        **_stats([2 * value for value in active_macs], "forward_flops_per_sample"),
        **_training_columns(training_summary),
    }
    return row


def write_resource_accounting(output_dir: str, rows: list[dict]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, RESOURCE_FILE)
    temporary = path + ".tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path
