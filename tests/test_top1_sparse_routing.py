from __future__ import annotations

import torch
from torch import nn

from models.dataset_experts import DatasetExpertBank
from models.moe import MoEDatasetNIDS
from training.config import load_config


class FixedGate(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.weights[: z.shape[0]]


def test_forward_selected_executes_only_experts_that_receive_rows():
    bank = DatasetExpertBank(["A", "B", "C"], latent_dim=4, num_classes=3, hidden_dims=[])
    calls = [0, 0, 0]
    row_counts = [0, 0, 0]
    handles = []
    for expert_id, expert in enumerate(bank.experts):
        def record(_module, inputs, _output, expert_id=expert_id):
            calls[expert_id] += 1
            row_counts[expert_id] += inputs[0].shape[0]
        handles.append(expert.register_forward_hook(record))

    z = torch.randn(4, 4, requires_grad=True)
    selected = torch.tensor([0, 2, 0, 2])
    logits = bank.forward_selected(z, selected)
    logits.square().sum().backward()
    for handle in handles:
        handle.remove()

    assert calls == [1, 0, 1]
    assert row_counts == [2, 0, 2]
    assert all(parameter.grad is None for parameter in bank.experts[1].parameters())
    assert all(parameter.grad is not None for parameter in bank.experts[0].parameters())
    assert all(parameter.grad is not None for parameter in bank.experts[2].parameters())


def test_top1_model_returns_only_selected_expert_predictions():
    weights = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    bank = DatasetExpertBank(["A", "B", "C"], latent_dim=4, num_classes=3, hidden_dims=[])
    model = MoEDatasetNIDS(
        nn.Identity(), bank, FixedGate(weights), ["c0", "c1", "c2"], routing_mode="top1"
    )
    output = model(torch.randn(2, 4))

    assert output["selected_experts"].tolist() == [0, 2]
    assert "expert_logits" not in output
    torch.testing.assert_close(output["combined_probs"].sum(dim=1), torch.ones(2))


def test_top1_assigned_only_drops_misrouted_expert_gradient():
    gate_logits = torch.tensor([[0.1, 0.9], [0.2, 0.8]], requires_grad=True)
    gate_weights = torch.softmax(gate_logits, dim=1).detach()
    selected_probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    selected_experts = torch.tensor([1, 1])
    dataset_ids = torch.tensor([0, 1])
    combined = MoEDatasetNIDS.combine_top1_for_training(
        gate_weights, selected_probs, selected_experts, dataset_ids, "assigned_only"
    )
    loss = -torch.log(combined[torch.arange(2), torch.tensor([0, 1])]).sum()
    loss.backward()

    assert torch.count_nonzero(selected_probs.grad[0]) == 0
    assert torch.count_nonzero(selected_probs.grad[1]) > 0
    assert gate_logits.grad is None


def test_sparse_routing_is_opt_in_and_dense_remains_default():
    dense = load_config("config/default.yaml")
    sparse = load_config("config/default.yaml", ["model.gate.routing=top1"])
    assert dense["model"]["gate"]["routing"] == "dense"
    assert sparse["model"]["gate"]["routing"] == "top1"
