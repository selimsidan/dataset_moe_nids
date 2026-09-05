from __future__ import annotations

from types import SimpleNamespace

import torch

from models.encoder import SharedEncoder
from models.moe import MoEDatasetNIDS
from training.checkpoint import load_stage_b
from training.config import load_config
from training.model_utils import build_model
from training.stage_b_warmstart import run_stage_b


def _model_cfg() -> dict:
    return {
        "latent_dim": 8,
        "encoder": {"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [12, 6], "dropout": 0.0},
        "adapter": {"rank": 4, "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }


def test_basic_moe_preset_removes_dataset_specific_training():
    config = load_config("config/default.yaml", ["architecture=moe_basic"])
    assert config["training"]["stage_b"]["warmstart_mode"] == "random_init"
    assert config["training"]["stage_c"]["gate_supervision"] == "none"
    assert config["training"]["stage_c"]["expert_update_policy"] == "all"
    assert config["training"]["stage_c"]["lambda_expert_anchor"] == 0.0


def test_basic_moe_has_same_parameter_count_but_generic_expert_names():
    cfg = _model_cfg()
    datasets = ["origin_a", "origin_b", "origin_c"]
    classes = ["Benign", "Attack"]
    dataset_model = build_model(
        "moe_dataset_soft", SharedEncoder(5, [16], 8), datasets, classes, cfg
    )
    basic_model = build_model(
        "moe_basic", SharedEncoder(5, [16], 8), datasets, classes, cfg
    )
    assert sum(p.numel() for p in basic_model.parameters()) == sum(
        p.numel() for p in dataset_model.parameters()
    )
    assert basic_model.dataset_names == ["expert_0", "expert_1", "expert_2"]
    assert dataset_model.dataset_names == datasets


def test_basic_stage_b_only_initializes_generic_experts(tmp_path):
    config = {
        "seed": 3,
        "architecture": "moe_basic",
        "model": _model_cfg(),
        "training": {
            "device": "cpu",
            "checkpoint_dir": str(tmp_path),
            "stage_b": {"warmstart_mode": "random_init"},
        },
    }
    data = SimpleNamespace(
        active_datasets=["origin_a", "origin_b"],
        class_names=["Benign", "Attack"],
    )
    bank = run_stage_b(config, data)
    checkpoint = load_stage_b(str(tmp_path))
    assert bank.dataset_names == ["expert_0", "expert_1"]
    assert checkpoint["dataset_names"] == ["expert_0", "expert_1"]


def test_unrestricted_basic_moe_combination_needs_no_dataset_ids():
    gate = torch.tensor([[0.25, 0.75]])
    experts = torch.tensor([[[0.8, 0.2], [0.4, 0.6]]])
    mixed = MoEDatasetNIDS.combine_probs_for_training(gate, experts, None, "all")
    assert torch.allclose(mixed, torch.tensor([[0.5, 0.5]]))
