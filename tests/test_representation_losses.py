import numpy as np
import torch

from models.representation_losses import (
    BalancedSupervisedContrastiveLoss,
    StageARepresentationObjective,
    SupervisedContrastiveLoss,
)
from training.sampler import ClassDomainBalancedBatchSampler


def test_supcon_is_finite_and_prefers_tight_class_clusters():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]])
    for loss_fn in (SupervisedContrastiveLoss(0.1), BalancedSupervisedContrastiveLoss(0.1)):
        assert torch.isfinite(loss_fn(separated, labels))
        assert loss_fn(separated, labels) < loss_fn(mixed, labels)


def test_supcon_singletons_return_differentiable_zero():
    z = torch.randn(3, 4, requires_grad=True)
    loss = SupervisedContrastiveLoss()(z, torch.tensor([0, 1, 2]))
    loss.backward()
    assert loss.item() == 0.0
    assert z.grad is not None


def test_every_representation_objective_backpropagates():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    for name in ("ce", "supcon", "balanced_supcon", "center", "arcface"):
        config = {"training": {"representation": {"objective": name}}}
        objective = StageARepresentationObjective(5, 3, config)
        z = torch.randn(6, 5, requires_grad=True)
        loss, parts, logits = objective(z, labels)
        loss.backward()
        assert logits.shape == (6, 3)
        assert torch.isfinite(loss)
        assert set(parts) == {"ce", "metric", "total"}
        assert z.grad is not None and torch.isfinite(z.grad).all()


def test_class_domain_sampler_supplies_cross_domain_class_positives():
    labels = np.repeat(np.arange(3), 8)
    domains = np.tile(np.repeat([0, 1], 4), 3)
    sampler = ClassDomainBalancedBatchSampler(
        labels, domains, batch_size=12, min_per_class=2, seed=7, num_batches=3
    )
    first_pass = list(sampler)
    second_pass = list(sampler)
    assert first_pass == second_pass
    for batch in first_pass:
        batch = np.asarray(batch)
        for class_id in range(3):
            selected = batch[labels[batch] == class_id]
            assert len(selected) >= 2
            assert len(np.unique(domains[selected])) >= 2
