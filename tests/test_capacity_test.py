from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml

from models.encoder import build_encoder, resolve_encoder_config
from training.compact_soft_moe_run import CompactSoftMoEStudy
from training.model_utils import apply_stage_c_trainability, build_model


ROOT_CONFIG = "config/default.yaml"
REFERENCE_CONFIG = (
    "config/compact_soft_moe_balanced_supcon_class_conditional_"
    "frozen_deeper_gate_3seed.yaml"
)
CAPACITY_CONFIG = "config/capacity_test_deep_big_3seed.yaml"
NOTEBOOK = "notebooks/capacity_test.ipynb"


def _study(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _capacity_model(study: dict):
    model_cfg = study["backbone"]["model"]
    encoder = build_encoder(
        47, model_cfg["latent_dim"], resolve_encoder_config(model_cfg)
    )
    return build_model(
        study["backbone"]["architecture"],
        encoder,
        ["UNSW", "ToN", "BoT", "CIC"],
        [f"class_{index}" for index in range(22)],
        model_cfg,
    )


def test_capacity_config_locks_deep_big_topology_and_parameter_count():
    study = _study(CAPACITY_CONFIG)
    backbone = study["backbone"]
    model_cfg = backbone["model"]

    assert study["seeds"] == [0, 1, 2]
    assert study["expected_input_features"] == 47
    assert study["expected_classes"] == 22
    assert study["expected_total_parameters"] == 737_844
    assert backbone["architecture"] == "moe_dataset_class_conditional"
    assert model_cfg["encoder"]["hidden_dims"] == [512, 512]
    assert model_cfg["latent_dim"] == 256
    assert model_cfg["expert"]["hidden_dims"] == [256]
    assert model_cfg["gate"]["hidden_dims"] == [128]
    assert model_cfg["gate"]["routing"] == "dense"

    model = _capacity_model(study)
    assert model.class_reliability.shape == (4, 22)
    assert torch.count_nonzero(model.class_reliability).item() == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 737_844


def test_capacity_config_changes_only_the_registered_capacity_fields():
    reference = _study(REFERENCE_CONFIG)
    capacity = _study(CAPACITY_CONFIG)

    assert "reuse_stage_a_from_prefix" not in capacity
    assert "reuse_stage_b_from_prefix" not in capacity
    assert reference["seeds"] == capacity["seeds"] == [0, 1, 2]
    assert reference["selection"] == capacity["selection"]

    expected = copy.deepcopy(reference["backbone"])
    candidate = copy.deepcopy(capacity["backbone"])
    expected["model"]["latent_dim"] = 256
    expected["model"]["encoder"]["hidden_dims"] = [512, 512]
    expected["model"]["expert"]["hidden_dims"] = [256]
    expected["model"]["gate"]["hidden_dims"] = [128]
    assert candidate == expected


def test_capacity_stage_c_trains_only_the_large_gate_and_reliability():
    study = _study(CAPACITY_CONFIG)
    model = _capacity_model(study)
    apply_stage_c_trainability(model, study["backbone"])

    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.expert_bank.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.gate.parameters())
    assert model.class_reliability.requires_grad
    assert sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    ) == 33_500


def test_capacity_runner_starts_fresh_and_binds_seed_zero(tmp_path):
    runner = CompactSoftMoEStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=CAPACITY_CONFIG,
        prefix="capacity_test",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )
    nested = runner._final_nested(0)

    assert nested["seed"] == 0
    assert nested["data"]["split_seed"] == 0
    assert nested["evaluation"]["latent"]["random_seed"] == 0
    assert nested["training"]["stages"] == ["A", "B", "C"]
    assert "stage_a_checkpoint" not in nested["training"]
    assert "stage_b_checkpoint" not in nested["training"]
    assert nested["training"]["deterministic"] is True
    assert nested["training"]["save_epoch_history"] is True


def test_capacity_notebook_is_safe_and_hard_locks_the_pilot_seed():
    notebook = json.loads(Path(NOTEBOOK).read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "EXECUTE = False" in source
    assert "PILOT_SEED = 0" in source
    assert "--only-seed', str(PILOT_SEED)" in source
    assert "--max-new-seeds', '1'" in source
    assert "capacity['expected_total_parameters'] == 737844" in source
    assert "== 33500" in source
    assert "runner._final_nested(PILOT_SEED)" in source
    assert "live_runner.audit_seed(PILOT_SEED)" in source
    assert "run_streaming_logged" in source
    assert "advance_to_three_seed_confirmation" in source
    assert "GITHUB_TOKEN" in source
    assert "x-access-token:{token}" in source
    assert f"{CAPACITY_CONFIG}" in source
