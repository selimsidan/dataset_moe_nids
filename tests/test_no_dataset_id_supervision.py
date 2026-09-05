"""Non-negotiable design constraint: the gate must never be trained with
ground-truth dataset ID as its PRIMARY supervision target. This test goes
beyond checking that the loss term is merely absent/zero-weighted -- it
verifies the training loop's computation graph structurally never calls
`models.losses.dataset_aux_loss` at all when
`training.stage_c.gate_supervision == "none"`, and DOES call it (as a
positive control) when set to "light_aux"/"hard". A monkeypatched
`dataset_aux_loss` that raises makes any accidental call fail loudly rather
than silently passing because its weight happened to be zero.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.encoder import SharedEncoder
from models.losses import dataset_aux_loss, load_balance_penalty
from models.moe import MoEDatasetNIDS
from training import model_utils, stage_c_jointfinetune
from training.checkpoint import save_stage_a, save_stage_b, stage_a_metadata
from training.config import apply_architecture_defaults
from training.dataset import PreparedData, PreparedSplit


def _synthetic_prepared_data(n=64, latent_input_dim=12, num_datasets=2, num_classes=4) -> PreparedData:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, latent_input_dim)).astype(np.float32)
    class_idx = rng.integers(0, num_classes, size=n).astype(np.int64)
    dataset_idx = rng.integers(0, num_datasets, size=n).astype(np.int64)
    dataset_names_arr = np.array([f"Dataset{d}" for d in dataset_idx])

    split = PreparedSplit(features=features, class_idx=class_idx, dataset_idx=dataset_idx, dataset_name=dataset_names_arr)
    return PreparedData(
        harmonizer=None,
        class_names=[f"c{i}" for i in range(num_classes)],
        active_datasets=[f"Dataset{i}" for i in range(num_datasets)],
        train=split,
        val=split,
        test=split,
    )


def _base_config(checkpoint_dir: str, gate_supervision: str) -> dict:
    return {
        "seed": 0,
        "architecture": "moe_dataset_soft",
        "model": {
            "latent_dim": 8,
            "encoder": {"hidden_dims": [16], "activation": "relu", "dropout": 0.0},
            "expert": {"hidden_dims": [8], "dropout": 0.0},
            "adapter": {"rank": 4, "dropout": 0.0},
            "gate": {"hidden_dims": []},
        },
        "load_balance": {"lambda_balance": 0.1},
        "training": {
            "device": "cpu",
            "batch_size": 16,
            "min_per_class_per_batch": 1,
            "epochs_c": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "stage_c_unfreeze": "all",
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_every_n_epochs": 1,
            "stage_c": {
                "gate_supervision": gate_supervision,
                "lambda_dataset_aux": 0.1,
                "lambda_dataset_aux_hard": 5.0,
                "lambda_dataset_aux_damex": 1.0,
            },
        },
    }


def _seed_stage_a_and_b_checkpoints(tmp_path, data: PreparedData, config: dict) -> None:
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1],
        hidden_dims=config["model"]["encoder"]["hidden_dims"],
        latent_dim=config["model"]["latent_dim"],
    )
    save_stage_a(
        str(tmp_path), encoder.state_dict(), data.class_names,
        metadata=stage_a_metadata(config, data),
    )

    bank_kind = model_utils.bank_kind_for_architecture(config["architecture"])
    bank = model_utils.build_expert_bank(bank_kind, data.active_datasets, config["model"]["latent_dim"], len(data.class_names), config["model"])
    save_stage_b(str(tmp_path), bank.state_dict(), data.active_datasets, bank_kind)


def test_gate_supervision_none_never_calls_dataset_aux_loss(tmp_path, monkeypatch):
    data = _synthetic_prepared_data()
    config = _base_config(str(tmp_path), gate_supervision="none")
    _seed_stage_a_and_b_checkpoints(tmp_path, data, config)

    def _boom(*args, **kwargs):
        raise AssertionError("dataset_aux_loss must never be called when gate_supervision == 'none'")

    monkeypatch.setattr(stage_c_jointfinetune, "dataset_aux_loss", _boom)

    assert stage_c_jointfinetune._lambda_dataset_aux(config) == 0.0
    stage_c_jointfinetune.run_stage_c(config, data)  # must not raise


def test_gate_supervision_light_aux_does_call_dataset_aux_loss(tmp_path, monkeypatch):
    data = _synthetic_prepared_data()
    config = _base_config(str(tmp_path), gate_supervision="light_aux")
    _seed_stage_a_and_b_checkpoints(tmp_path, data, config)

    calls = {"n": 0}
    real_fn = stage_c_jointfinetune.dataset_aux_loss

    def _spy(gate_weights, dataset_id):
        calls["n"] += 1
        return real_fn(gate_weights, dataset_id)

    monkeypatch.setattr(stage_c_jointfinetune, "dataset_aux_loss", _spy)

    assert stage_c_jointfinetune._lambda_dataset_aux(config) == pytest.approx(0.1)
    stage_c_jointfinetune.run_stage_c(config, data)
    assert calls["n"] > 0, "expected dataset_aux_loss to be called at least once under gate_supervision=light_aux"


def test_gate_supervision_hard_uses_large_weight(tmp_path):
    config = _base_config(str(tmp_path), gate_supervision="hard")
    assert stage_c_jointfinetune._lambda_dataset_aux(config) == pytest.approx(5.0)
    assert stage_c_jointfinetune._lambda_dataset_aux(config) > config["training"]["stage_c"]["lambda_dataset_aux"]


def test_damex_router_gets_dataset_and_balance_gradients_but_not_task_gradient():
    gate_logits = torch.tensor([[1.0, -0.3], [-0.2, 0.7]], requires_grad=True)
    gate_weights = torch.softmax(gate_logits, dim=1)
    expert_probs = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.1, 0.9]]],
        requires_grad=True,
    )
    dataset_ids = torch.tensor([0, 1])
    labels = torch.tensor([0, 1])

    task_weights = stage_c_jointfinetune._gate_weights_for_task(gate_weights, "damex")
    combined = MoEDatasetNIDS.combine_probs_for_training(
        task_weights, expert_probs, dataset_ids, "assigned_only"
    )
    task_loss = -torch.log(combined[torch.arange(2), labels]).mean()
    task_loss.backward(retain_graph=True)

    assert gate_logits.grad is None
    assert torch.count_nonzero(expert_probs.grad) > 0

    router_loss = dataset_aux_loss(gate_weights, dataset_ids) + 0.1 * load_balance_penalty(gate_weights)
    router_loss.backward()
    assert torch.count_nonzero(gate_logits.grad) > 0


def test_damex_architecture_preset_is_strict_but_explicitly_overridable():
    config = _base_config("unused", gate_supervision="light_aux")
    config["architecture"] = "moe_dataset_damex"
    apply_architecture_defaults(config)
    assert config["training"]["stage_c"]["gate_supervision"] == "damex"
    assert config["training"]["stage_c"]["expert_update_policy"] == "assigned_only"

    config = _base_config("unused", gate_supervision="light_aux")
    config["architecture"] = "moe_dataset_damex"
    config["training"]["stage_c"]["expert_update_policy"] = "all"
    apply_architecture_defaults(config, ["training.stage_c.expert_update_policy=all"])
    assert config["training"]["stage_c"]["gate_supervision"] == "damex"
    assert config["training"]["stage_c"]["expert_update_policy"] == "all"


def test_gate_supervision_damex_uses_direct_dataset_weight(tmp_path):
    config = _base_config(str(tmp_path), gate_supervision="damex")
    assert stage_c_jointfinetune._lambda_dataset_aux(config) == pytest.approx(1.0)


def test_damex_top1_stage_c_runs_with_strict_expert_ownership(tmp_path):
    data = _synthetic_prepared_data()
    config = _base_config(str(tmp_path), gate_supervision="damex")
    config["model"]["gate"]["routing"] = "top1"
    config["training"]["stage_c"]["expert_update_policy"] = "assigned_only"
    _seed_stage_a_and_b_checkpoints(tmp_path, data, config)

    model = stage_c_jointfinetune.run_stage_c(config, data)
    assert model.routing_mode == "top1"
    output = model(torch.from_numpy(data.test.features[:8]))
    assert output["selected_experts"].shape == (8,)
    assert "expert_logits" not in output
