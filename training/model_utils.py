"""Small shared helpers for building the right expert bank / full model for
a given `architecture` config value. Used by training stages B/C,
inference/predict.py, and evaluation/gate_analysis.py so there is exactly
one place that maps `architecture` -> concrete `nn.Module` wiring.
"""
from __future__ import annotations

import torch
from torch import nn

from models.adapters import AdapterExpertBank
from models.baselines import MatchedDenseClassifier, MatchedDenseHead
from models.dataset_experts import DatasetExpertBank
from models.encoder import SharedEncoder
from models.gate import Gate
from models.moe import MoEDatasetNIDS

ADAPTER_ARCHITECTURES = {"moe_dataset_adapters"}
BASIC_MOE_ARCHITECTURES = {"moe_basic"}
DATASET_MOE_ARCHITECTURES = {
    "moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_damex", "moe_dataset_adapters"
}


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _linear_macs(module: nn.Module) -> int:
    """Static per-sample MACs for Linear weights; nonlinearities are reported separately."""
    return sum(layer.weight.numel() for layer in module.modules() if isinstance(layer, nn.Linear))


def _resolve_dense_hidden_dims(
    latent_dim: int,
    num_classes: int,
    target: int,
    axis: str,
    reference_hidden_dims: list[int],
    max_width: int,
) -> tuple[tuple[int, int], int]:
    """Find the deterministic closest two-layer integer architecture."""
    if target <= 0 or max_width < 1:
        raise ValueError("dense matching target and max_width must be positive")
    best_key = None
    best = None
    reference_second = reference_hidden_dims[-1] if reference_hidden_dims else latent_dim
    reference_first = reference_hidden_dims[0] if reference_hidden_dims else latent_dim
    for first in range(1, max_width + 1):
        if axis in {"total_params", "active_params"}:
            constant = (latent_dim + 1) * first + num_classes
            slope = first + 1 + num_classes
        else:
            constant = latent_dim * first
            slope = first + num_classes
        ideal = (target - constant) / slope
        candidates = {
            1,
            max_width,
            first,
            max(1, min(max_width, int(ideal))),
            max(1, min(max_width, int(ideal) + 1)),
        }
        for second in sorted(candidates):
            # Preserve the expert family's pyramidal shape; an exact count
            # obtained via a tiny first layer and enormous second layer would
            # be a materially different architecture rather than a fair widen.
            if second > first:
                continue
            params = (latent_dim + 1) * first + (first + 1) * second + (second + 1) * num_classes
            macs = latent_dim * first + first * second + second * num_classes
            value = params if axis in {"total_params", "active_params"} else macs
            key = (
                abs(value - target),
                abs(second - reference_second),
                abs(first - reference_first),
                first,
                second,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (first, second, value)
    assert best is not None
    return (best[0], best[1]), best[2]


def build_matched_dense_model(
    encoder: SharedEncoder,
    dataset_names: list[str],
    class_names: list[str],
    model_cfg: dict,
) -> MatchedDenseClassifier:
    """Build a dense head matched to the configured full-expert MoE reference."""
    dense_cfg = model_cfg.get("dense_match", {})
    axis = dense_cfg.get("axis", "total_params")
    if axis not in {"total_params", "active_params", "forward_macs"}:
        raise ValueError("model.dense_match.axis must be total_params, active_params, or forward_macs")
    reference_routing = dense_cfg.get(
        "reference_routing", "dense" if axis == "total_params" else "top1"
    )
    if reference_routing not in {"dense", "top1"}:
        raise ValueError("model.dense_match.reference_routing must be dense or top1")
    bank = build_expert_bank(
        "full", dataset_names, model_cfg["latent_dim"], len(class_names), model_cfg
    )
    gate = Gate(model_cfg["latent_dim"], len(dataset_names), model_cfg["gate"]["hidden_dims"])
    if not bank.experts:
        raise ValueError("matched_dense requires at least one reference expert")
    gate_params, gate_macs = _parameter_count(gate), _linear_macs(gate)
    expert_params = [_parameter_count(expert) for expert in bank.experts]
    expert_macs = [_linear_macs(expert) for expert in bank.experts]
    if reference_routing == "dense":
        active_params = gate_params + sum(expert_params)
        forward_macs = gate_macs + sum(expert_macs)
    else:
        active_params = gate_params + max(expert_params)
        forward_macs = gate_macs + max(expert_macs)
    targets = {
        "total_params": gate_params + sum(expert_params),
        "active_params": active_params,
        "forward_macs": forward_macs,
    }
    target = targets[axis]
    hidden_dims, achieved = _resolve_dense_hidden_dims(
        model_cfg["latent_dim"],
        len(class_names),
        target,
        axis,
        list(model_cfg["expert"]["hidden_dims"]),
        int(dense_cfg.get("max_width", 2048)),
    )
    relative_error = abs(achieved - target) / target
    tolerance = float(dense_cfg.get("tolerance", 0.005))
    if relative_error > tolerance:
        raise ValueError(
            f"Could not match {axis}: target={target}, achieved={achieved}, "
            f"relative_error={relative_error:.6f} exceeds tolerance={tolerance:.6f}"
        )
    info = {
        "axis": axis,
        "reference_routing": reference_routing,
        "target": target,
        "achieved": achieved,
        "residual": achieved - target,
        "relative_error": relative_error,
        "hidden_dims": list(hidden_dims),
        "reference_total_downstream_params": targets["total_params"],
        "reference_active_downstream_params": targets["active_params"],
        "reference_forward_downstream_macs": targets["forward_macs"],
    }
    head = MatchedDenseHead(
        model_cfg["latent_dim"],
        len(class_names),
        hidden_dims,
        dropout=float(model_cfg["expert"].get("dropout", 0.1)),
        match_info=info,
    )
    return MatchedDenseClassifier(encoder, head)


def expert_names_for_architecture(architecture: str, dataset_names: list[str]) -> list[str]:
    """Return capacity-matched expert labels without implying false ownership.

    The basic comparator deliberately keeps one expert per active dataset in
    *count* so parameter capacity matches the dataset-MoE, but its experts are
    generic and never assigned an origin dataset.
    """
    if architecture in BASIC_MOE_ARCHITECTURES:
        return [f"expert_{index}" for index in range(len(dataset_names))]
    return list(dataset_names)


def bank_kind_for_architecture(architecture: str) -> str:
    return "adapter" if architecture in ADAPTER_ARCHITECTURES else "full"


def build_expert_bank(bank_kind: str, dataset_names: list[str], latent_dim: int, num_classes: int, model_cfg: dict) -> nn.Module:
    if bank_kind == "adapter":
        adapter_cfg = model_cfg.get("adapter", {})
        return AdapterExpertBank(
            dataset_names,
            latent_dim,
            num_classes,
            rank=adapter_cfg.get("rank", 16),
            dropout=adapter_cfg.get("dropout", 0.1),
        )
    if bank_kind == "full":
        expert_cfg = model_cfg["expert"]
        return DatasetExpertBank(
            dataset_names,
            latent_dim,
            num_classes,
            hidden_dims=expert_cfg["hidden_dims"],
            dropout=expert_cfg["dropout"],
        )
    raise ValueError(f"Unknown bank_kind '{bank_kind}'. Expected 'full' or 'adapter'.")


def expert_train_params(bank: nn.Module, dataset_idx: int) -> list[torch.nn.Parameter]:
    """Parameters to optimize during Stage B's per-dataset independent
    warm-start for dataset `dataset_idx`. For the full-expert bank this is
    just that dataset's own expert; for the adapter bank it's that
    dataset's adapter PLUS the shared head (there is no per-dataset-only
    subset of a shared head) -- sequential warm-start across datasets will
    therefore let later datasets' warm-start nudge the shared head touched
    by earlier ones, which is expected/documented behavior for this
    ablation, not a bug.
    """
    if isinstance(bank, AdapterExpertBank):
        return list(bank.adapters[dataset_idx].parameters()) + list(bank.shared_head.parameters())
    return list(bank.experts[dataset_idx].parameters())


def expert_forward_one(bank: nn.Module, dataset_idx: int, z: torch.Tensor) -> torch.Tensor:
    """Runs only dataset `dataset_idx`'s expert (not the whole bank) -- used
    exclusively by Stage B's per-dataset independent training loop, where
    running every expert on every batch would be wasted compute (Stage B
    trains one expert at a time). Stage C and inference use either the dense
    bank forward or the config-selected top-1 grouped dispatch path.
    """
    if isinstance(bank, AdapterExpertBank):
        return bank.shared_head(bank.adapters[dataset_idx](z))
    return bank.experts[dataset_idx](z)


def build_model(
    architecture: str,
    encoder: SharedEncoder,
    dataset_names: list[str],
    class_names: list[str],
    model_cfg: dict,
) -> MoEDatasetNIDS:
    bank_kind = bank_kind_for_architecture(architecture)
    expert_names = expert_names_for_architecture(architecture, dataset_names)
    bank = build_expert_bank(bank_kind, expert_names, model_cfg["latent_dim"], len(class_names), model_cfg)
    gate = Gate(model_cfg["latent_dim"], len(dataset_names), model_cfg["gate"]["hidden_dims"])
    routing_mode = model_cfg.get("gate", {}).get("routing", "dense")
    return MoEDatasetNIDS(encoder, bank, gate, class_names, routing_mode=routing_mode)
