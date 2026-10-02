from __future__ import annotations

import torch

from evaluation.resource_accounting import resource_profile
from models.encoder import build_encoder
from models.global_residual_experts import GlobalResidualExpertBank
from training.model_utils import (
    apply_stage_c_trainability,
    build_model,
    expert_forward_one,
    expert_train_params,
    initialize_global_residual_head,
)


def _model_config(gate_hidden=None):
    return {
        "latent_dim": 5,
        "encoder": {"hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [], "dropout": 0.0},
        "gate": {"hidden_dims": gate_hidden or [], "routing": "dense"},
    }


def test_global_residual_bank_starts_as_exact_stage_a_global_classifier():
    torch.manual_seed(4)
    bank = GlobalResidualExpertBank(["A", "B", "C"], 5, 3)
    weight = torch.randn(3, 5)
    bias = torch.randn(3)
    initialize_global_residual_head(bank, {
        "representation_state": {
            "classifier.linear.weight": weight,
            "classifier.linear.bias": bias,
        }
    })
    z = torch.randn(11, 5)
    expected = torch.nn.functional.linear(z, weight, bias)
    logits = bank(z)
    assert logits.shape == (11, 3, 3)
    assert torch.allclose(logits, expected[:, None, :].expand_as(logits))
    assert all(not parameter.requires_grad for parameter in bank.global_head.parameters())
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in bank.residuals)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in bank.residuals)


def test_stage_b_updates_only_the_owned_residual_parameters():
    bank = GlobalResidualExpertBank(["A", "B"], 5, 3)
    params = expert_train_params(bank, 1)
    assert {id(value) for value in params} == {
        id(value) for value in bank.residuals[1].parameters()
    }
    assert not ({id(value) for value in params} & {
        id(value) for value in bank.global_head.parameters()
    })
    z = torch.randn(7, 5)
    assert torch.allclose(
        expert_forward_one(bank, 1, z),
        bank.global_head(z) + bank.residuals[1](z),
    )


def test_global_residual_model_parameter_count_and_dense_probabilities():
    cfg = _model_config()
    model = build_model(
        "moe_dataset_global_residual",
        build_encoder(4, 5, cfg["encoder"]),
        ["A", "B"],
        ["ok", "attack", "rare"],
        cfg,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 141
    output = model(torch.randn(13, 4))
    assert output["expert_logits"].shape == (13, 2, 3)
    assert torch.allclose(output["combined_probs"].sum(dim=1), torch.ones(13))


def test_frozen_stage_c_policy_leaves_only_gate_trainable():
    cfg = _model_config([3])
    model = build_model(
        "moe_dataset_soft",
        build_encoder(4, 5, cfg["encoder"]),
        ["A", "B"],
        ["ok", "attack", "rare"],
        cfg,
    )
    config = {
        "training": {
            "stage_c_unfreeze": "none",
            "stage_c": {"freeze_experts": True},
        }
    }
    apply_stage_c_trainability(model, config)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.expert_bank.parameters())
    assert all(parameter.requires_grad for parameter in model.gate.parameters())


def test_global_residual_resource_accounting_counts_shared_head_on_active_path():
    cfg = _model_config()
    cfg["gate"]["routing"] = "top1"
    model = build_model(
        "moe_dataset_global_residual",
        build_encoder(4, 5, cfg["encoder"]),
        ["A", "B"],
        ["ok", "attack", "rare"],
        cfg,
    )
    row = resource_profile(
        model,
        {"architecture": "moe_dataset_global_residual", "model": cfg},
    )
    assert row["total_parameters"] == 141
    assert row["classifier_expert_parameters"] == 54
    assert row["active_parameters_per_sample_mean"] == 123
    assert row["forward_macs_per_sample_mean"] == 103
