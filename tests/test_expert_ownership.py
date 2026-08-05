import pytest
import torch

from models.moe import MoEDatasetNIDS


def test_assigned_only_preserves_soft_forward_and_masks_expert_gradients():
    expert_probs = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.25, 0.60]],
            [[0.20, 0.50, 0.30], [0.65, 0.20, 0.15]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    gate_logits = torch.tensor([[0.4, -0.2], [-0.1, 0.3]], requires_grad=True)
    gate_weights = torch.softmax(gate_logits, dim=1)
    dataset_ids = torch.tensor([0, 1])

    ordinary = torch.einsum("bd,bdc->bc", gate_weights, expert_probs)
    owned = MoEDatasetNIDS.combine_probs_for_training(
        gate_weights, expert_probs, dataset_ids, "assigned_only"
    )
    torch.testing.assert_close(owned, ordinary)

    loss = -torch.log(owned[torch.arange(2), torch.tensor([0, 1])]).sum()
    loss.backward()

    assert torch.count_nonzero(expert_probs.grad[0, 0]) > 0
    assert torch.count_nonzero(expert_probs.grad[1, 1]) > 0
    assert torch.count_nonzero(expert_probs.grad[0, 1]) == 0
    assert torch.count_nonzero(expert_probs.grad[1, 0]) == 0
    assert torch.count_nonzero(gate_logits.grad) > 0


def test_assigned_only_rejects_invalid_dataset_owner():
    expert_probs = torch.full((2, 2, 3), 1 / 3)
    gate_weights = torch.full((2, 2), 1 / 2)
    with pytest.raises(ValueError, match="outside the active expert bank"):
        MoEDatasetNIDS.combine_probs_for_training(
            gate_weights, expert_probs, torch.tensor([0, 2]), "assigned_only"
        )
