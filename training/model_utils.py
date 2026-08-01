"""Small shared helpers for building the right expert bank / full model for
a given `architecture` config value. Used by training stages B/C,
inference/predict.py, and evaluation/gate_analysis.py so there is exactly
one place that maps `architecture` -> concrete `nn.Module` wiring.
"""
from __future__ import annotations

import torch
from torch import nn

from models.adapters import AdapterExpertBank
from models.dataset_experts import DatasetExpertBank
from models.encoder import SharedEncoder
from models.gate import Gate
from models.moe import MoEDatasetNIDS

ADAPTER_ARCHITECTURES = {"moe_dataset_adapters"}
DATASET_MOE_ARCHITECTURES = {"moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_adapters"}


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
    trains one expert at a time). Stage C and inference always go through
    `bank.forward`, which runs every expert unconditionally, per the "every
    expert sees every sample" structural constraint.
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
    bank = build_expert_bank(bank_kind, dataset_names, model_cfg["latent_dim"], len(class_names), model_cfg)
    gate = Gate(model_cfg["latent_dim"], len(dataset_names), model_cfg["gate"]["hidden_dims"])
    return MoEDatasetNIDS(encoder, bank, gate, class_names)
