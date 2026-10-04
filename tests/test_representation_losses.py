import numpy as np
import torch

from models.representation_losses import (
    BalancedSupervisedContrastiveLoss,
    ConfusionAdaptiveProxyMarginLoss,
    StageARepresentationObjective,
    SupervisedContrastiveLoss,
)
from training.sampler import ClassDomainBalancedBatchSampler
from training.out_of_core_train import (
    _empty_representation_coverage,
    _update_representation_coverage,
    _write_representation_coverage,
)


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


def _adaptive_config(**overrides):
    representation = {
        "objective": "confusion_adaptive_margin",
        "weight": 0.1,
        "temperature": 0.1,
        "confusion_warmup_epochs": 1,
        "confusion_ema": 0.5,
        "confusion_shrinkage": 100.0,
        "confusion_top_k": 2,
        "confusion_max_margin": 0.15,
    }
    representation.update(overrides)
    return {"training": {"representation": representation}}


def test_proxy_margin_is_equivalent_to_exponential_rival_weighting():
    z = torch.tensor([[1.0, 0.2], [-0.4, 1.0]])
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 1])
    margins = torch.tensor([
        [0.0, 0.07, 0.03],
        [0.11, 0.0, 0.04],
        [0.02, 0.06, 0.0],
    ])
    temperature = 0.1
    actual = ConfusionAdaptiveProxyMarginLoss(temperature)(
        z, labels, proxies, margins
    )
    cosine = torch.nn.functional.linear(
        torch.nn.functional.normalize(z, dim=1),
        torch.nn.functional.normalize(proxies, dim=1),
    )
    exponentials = torch.exp(cosine / temperature)
    rival_weights = torch.exp(margins.index_select(0, labels) / temperature)
    expected = -torch.log(
        exponentials[torch.arange(len(labels)), labels]
        / (exponentials * rival_weights).sum(dim=1)
    ).mean()
    assert torch.allclose(actual, expected, atol=1e-6)


def test_adaptive_confusion_preserves_absolute_leakage_and_builds_capped_top_two():
    objective = StageARepresentationObjective(4, 4, _adaptive_config())
    objective.begin_confusion_epoch(1)
    labels = torch.tensor([0] * 10 + [1] * 2 + [2])
    probabilities = torch.tensor(
        [[0.97, 0.01, 0.01, 0.01]] * 10
        + [[0.05, 0.20, 0.55, 0.20]] * 2
        + [[0.05, 0.75, 0.10, 0.10]]
    )
    objective.accumulate_confusion(probabilities.log(), labels)
    snapshot = objective.finalize_confusion_epoch()
    assert snapshot is not None
    raw = snapshot["raw"]
    assert raw[0].sum() < raw[1].sum()
    assert not torch.allclose(raw.sum(dim=1), torch.ones(4, dtype=torch.float64))
    assert snapshot["shrunk"][2].sum() > 0
    margins = snapshot["margins"]
    assert torch.all(torch.count_nonzero(margins, dim=1) <= 2)
    assert margins.max() <= 0.15
    assert margins[0].sum() < margins[1].sum()
    assert torch.count_nonzero(torch.diag(margins)) == 0


def test_adaptive_confusion_warmup_detachment_and_state_round_trip():
    config = _adaptive_config(confusion_warmup_epochs=2, confusion_shrinkage=0.0)
    objective = StageARepresentationObjective(3, 3, config)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.tensor([
        [3.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
        [1.0, 2.0, 0.0], [1.0, 2.0, 0.0],
    ], requires_grad=True)
    objective.begin_confusion_epoch(1)
    objective.accumulate_confusion(logits, labels)
    first = objective.finalize_confusion_epoch()
    assert first is not None and not first["initialized"].item()
    assert not objective.confusion_proxy_active

    objective.begin_confusion_epoch(2)
    objective.accumulate_confusion(logits, labels)
    second = objective.finalize_confusion_epoch()
    assert second is not None and second["initialized"].item()
    assert not objective.confusion_ema.requires_grad
    assert logits.grad is None

    restored = StageARepresentationObjective(3, 3, config)
    restored.load_state_dict(objective.state_dict())
    assert torch.equal(restored.confusion_ema, objective.confusion_ema)
    assert torch.equal(restored.confusion_margins, objective.confusion_margins)
    restored.begin_confusion_epoch(3)
    assert restored.confusion_proxy_active
    z = torch.randn(6, 3, requires_grad=True)
    total, parts, _ = restored(z, labels)
    total.backward()
    assert parts["metric"].item() > 0
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


def test_representation_coverage_reports_singletons_and_valid_anchors(tmp_path):
    coverage = _empty_representation_coverage(3)
    _update_representation_coverage(coverage, torch.tensor([0, 0, 1, 2, 2, 2]), 3)
    _update_representation_coverage(coverage, torch.tensor([0, 1, 1, 1]), 3)
    path = _write_representation_coverage(
        str(tmp_path), ["Benign", "Rare", "Attack"], coverage
    )
    import pandas as pd
    frame = pd.read_csv(path).set_index("class")
    assert frame.loc["Benign", "singleton_batches"] == 1
    assert frame.loc["Rare", "singleton_batches"] == 1
    assert frame.loc["Attack", "batch_appearances"] == 1
    assert frame.loc["Attack", "valid_anchor_fraction"] == 1.0
    assert 0.0 < frame.loc["__ALL__", "valid_anchor_fraction"] < 1.0
