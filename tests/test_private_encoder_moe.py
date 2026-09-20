import torch
import torch.nn.functional as F

from evaluation.resource_accounting import parameter_count, resource_profile
from models.encoder import SharedEncoder
from models.moe import MoEDatasetNIDS
from models.private_encoder_experts import PrivateEncoderExpertBank
from training.model_utils import (
    build_model,
    initialize_private_expert_encoders,
)


def _model_cfg(routing="dense"):
    return {
        "latent_dim": 5,
        "encoder": {"hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [6], "dropout": 0.0},
        "adapter": {"rank": 3, "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": routing},
    }


def _build(routing="dense"):
    cfg = _model_cfg(routing)
    encoder = SharedEncoder(4, [7], 5, dropout=0.0)
    model = build_model(
        "moe_dataset_private_encoders",
        encoder,
        ["A", "B"],
        ["Benign", "Attack"],
        cfg,
    )
    return model, cfg


def test_private_encoder_moe_has_independent_full_branches_and_dense_mixture():
    model, _ = _build()
    assert isinstance(model.expert_bank, PrivateEncoderExpertBank)
    left, right = model.expert_bank.experts
    assert left.encoder is not right.encoder
    assert next(left.encoder.parameters()).data_ptr() != next(right.encoder.parameters()).data_ptr()

    output = model(torch.randn(8, 4))
    assert output["expert_logits"].shape == (8, 2, 2)
    assert output["gate_weights"].shape == (8, 2)
    assert output["combined_probs"].shape == (8, 2)
    assert torch.allclose(output["combined_probs"].sum(dim=1), torch.ones(8), atol=1e-6)


def test_stage_a_initialization_clones_values_without_sharing_parameters():
    model, _ = _build()
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.fill_(0.25)
    initialize_private_expert_encoders(model.expert_bank, model.encoder)

    source = list(model.encoder.parameters())
    for branch in model.expert_bank.experts:
        private = list(branch.encoder.parameters())
        assert all(torch.equal(a, b) for a, b in zip(source, private))
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(source, private))


def test_assigned_only_blocks_non_owner_private_encoder_gradients():
    model, _ = _build()
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    output = model(features)
    combined = MoEDatasetNIDS.combine_probs_for_training(
        output["gate_weights"],
        output["expert_probs"],
        dataset_ids=torch.zeros(6, dtype=torch.long),
        expert_update_policy="assigned_only",
    )
    F.nll_loss(torch.log(combined.clamp_min(1e-12)), labels).backward()

    owner_grad = [parameter.grad for parameter in model.expert_bank.experts[0].encoder.parameters()]
    non_owner_grad = [parameter.grad for parameter in model.expert_bank.experts[1].encoder.parameters()]
    assert any(value is not None and torch.count_nonzero(value) for value in owner_grad)
    assert all(value is None or not torch.count_nonzero(value) for value in non_owner_grad)


def test_private_encoder_top1_executes_only_selected_complete_branch():
    model, _ = _build(routing="top1")
    calls = [0, 0]
    hooks = []
    for index, branch in enumerate(model.expert_bank.experts):
        hooks.append(branch.register_forward_hook(
            lambda _module, _args, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        ))
    with torch.no_grad():
        model.gate.net.weight.zero_()
        model.gate.net.bias.copy_(torch.tensor([10.0, -10.0]))
        output = model(torch.randn(5, 4))
    for hook in hooks:
        hook.remove()
    assert output["selected_experts"].tolist() == [0] * 5
    assert calls == [1, 0]


def test_private_encoder_resource_accounting_counts_dense_and_top1_paths():
    dense, cfg = _build("dense")
    config = {
        "architecture": "moe_dataset_private_encoders",
        "model": cfg,
        "training": {"stage_c": {}},
    }
    dense_row = resource_profile(dense, config)
    assert dense_row["active_parameters_per_sample_mean"] == parameter_count(dense)

    sparse, sparse_cfg = _build("top1")
    config["model"] = sparse_cfg
    sparse_row = resource_profile(sparse, config)
    expected = (
        parameter_count(sparse.encoder)
        + parameter_count(sparse.gate)
        + parameter_count(sparse.expert_bank.experts[0])
    )
    assert sparse_row["active_parameters_per_sample_mean"] == expected
    assert sparse_row["active_parameters_per_sample_mean"] < parameter_count(sparse)
