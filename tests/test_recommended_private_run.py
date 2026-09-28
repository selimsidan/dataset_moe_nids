from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from models.representation_losses import representation_config, stage_a_encoder_role
from models.encoder import build_encoder, resolve_encoder_config
from training.checkpoint import load_validated_stage_a, save_stage_a, stage_a_metadata
from training.model_utils import build_model
from training.recommended_private_run import (
    RecommendedPrivateStudy,
    _mean_sd,
    flatten_overrides,
)


ROOT_CONFIG = "config/default.yaml"
STUDY_CONFIG = "config/recommended_private_3seed.yaml"


def _runner(tmp_path) -> RecommendedPrivateStudy:
    return RecommendedPrivateStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_recommended",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )


def test_recommended_study_contract_is_locked():
    with open(STUDY_CONFIG) as handle:
        study = yaml.safe_load(handle)
    assert study["seeds"] == [0, 1, 2]
    assert study["expected_input_features"] == 47
    assert study["expected_classes"] == 22
    assert study["expected_total_parameters"] == 629_212
    backbone = study["backbone"]
    assert backbone["architecture"] == "moe_dataset_private_encoders"
    assert backbone["model"]["gate_encoder"]["hidden_dims"] == [128]
    assert backbone["model"]["private_encoder"]["kind"] == "residual_mlp"
    assert backbone["model"]["private_encoder"]["blocks"] == 2
    assert backbone["model"]["expert"]["hidden_dims"] == [64]
    assert backbone["model"]["gate"]["hidden_dims"] == [64]
    assert backbone["training"]["epochs_b"] == 15
    assert backbone["training"]["stage_c"]["optimizer"]["lr"] == 0.0003


def test_recommended_topology_has_exact_locked_parameter_count(tmp_path):
    runner = _runner(tmp_path)
    config = runner._final_config(0)
    model_config = config["model"]
    encoder = build_encoder(
        47,
        model_config["latent_dim"],
        resolve_encoder_config(model_config, "gate_encoder"),
    )
    model = build_model(
        config["architecture"],
        encoder,
        ["cicids2017", "cse-cic-ids2018", "ton_iot", "unsw_nb15"],
        [f"class_{index}" for index in range(22)],
        model_config,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 629_212


def test_role_specific_representation_resolution_and_legacy_fallback():
    legacy = {"training": {"representation": {
        "objective": "ce", "sampling": "legacy", "class_weighting": "legacy"
    }}}
    assert stage_a_encoder_role(legacy) == "encoder"
    assert representation_config(legacy)["objective"] == "ce"

    role_aware = {"training": {
        "stage_a_encoder_role": "private_encoder",
        "representation": {
            "objective": "ce", "sampling": "legacy", "class_weighting": "legacy",
            "weight": 0.1, "temperature": 0.1,
        },
        "representation_by_role": {
            "private_encoder": {
                "objective": "balanced_supcon",
                "sampling": "class_domain_balanced",
                "class_weighting": "none",
            }
        },
    }}
    resolved = representation_config(role_aware)
    assert resolved["objective"] == "balanced_supcon"
    assert resolved["sampling"] == "class_domain_balanced"
    assert resolved["class_weighting"] == "none"
    assert representation_config(role_aware, "gate_encoder")["objective"] == "ce"


def test_role_checkpoint_rejects_wrong_topology_and_objective(tmp_path):
    runner = _runner(tmp_path)
    config = runner._config(runner._stage_a_nested(0, "gate_encoder"))
    config["training"]["checkpoint_dir"] = str(tmp_path / "gate")
    data = SimpleNamespace(
        train=SimpleNamespace(features=np.zeros((8, 47), dtype=np.float32)),
        class_names=[f"class_{index}" for index in range(22)],
        active_datasets=["A", "B", "C", "D"],
        harmonizer=None,
    )
    metadata = stage_a_metadata(
        config,
        data,
        split_signature="fixed-split",
        feature_columns=[f"feature_{index}" for index in range(47)],
        encoder_role="gate_encoder",
    )
    encoder = build_encoder(
        47,
        config["model"]["latent_dim"],
        resolve_encoder_config(config["model"], "gate_encoder"),
    )
    save_stage_a(
        config["training"]["checkpoint_dir"],
        encoder.state_dict(),
        data.class_names,
        metadata=metadata,
        representation_config=representation_config(config, "gate_encoder"),
    )
    load_validated_stage_a(config, metadata, "gate_encoder")

    wrong_role = stage_a_metadata(
        config,
        data,
        split_signature="fixed-split",
        feature_columns=[f"feature_{index}" for index in range(47)],
        encoder_role="private_encoder",
    )
    with pytest.raises(ValueError, match="Incompatible Stage-A checkpoint"):
        load_validated_stage_a(config, wrong_role, "gate_encoder")

    wrong_objective = copy.deepcopy(config)
    wrong_objective["training"]["representation_by_role"]["gate_encoder"][
        "objective"
    ] = "supcon"
    with pytest.raises(ValueError, match="representation checkpoint"):
        load_validated_stage_a(wrong_objective, metadata, "gate_encoder")


def test_fixed_runner_builds_separate_role_checkpoints_and_three_seeds(tmp_path):
    runner = _runner(tmp_path)
    gate = runner._stage_a_nested(0, "gate_encoder")
    private = runner._stage_a_nested(0, "private_encoder")
    assert gate["training"]["stage_a_encoder_role"] == "gate_encoder"
    assert private["training"]["stage_a_encoder_role"] == "private_encoder"
    assert runner._stage_a_path(0, "gate_encoder") != runner._stage_a_path(0, "private_encoder")
    stage_b = runner._stage_b_nested(0)
    assert stage_b["training"]["stage_a_checkpoint"] == runner._stage_a_path(0, "gate_encoder")
    assert stage_b["training"]["private_stage_a_checkpoint"] == runner._stage_a_path(0, "private_encoder")
    assert runner._stage_b_path(0).endswith("stage_b_expert_bank.pt")
    assert [runner._final_config(seed)["seed"] for seed in (0, 1, 2)] == [0, 1, 2]
    assert [runner._final_config(seed)["data"]["split_seed"] for seed in (0, 1, 2)] == [0, 1, 2]


def test_runner_dry_graph_has_all_stages_for_exactly_three_seeds(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    commands = []
    latent = []
    monkeypatch.setattr(runner, "_seed_complete", lambda seed: False)
    monkeypatch.setattr(runner, "_test_complete", lambda seed: False)
    monkeypatch.setattr(runner, "_run", lambda label, nested: commands.append((label, nested["seed"])))
    monkeypatch.setattr(runner, "_materialize_latent", lambda seed, stage: latent.append((stage, seed)))
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    runner.run()
    assert commands == [
        (label, seed)
        for seed in (0, 1, 2)
        for label in (
            "stage-a:gate_encoder", "stage-a:private_encoder", "stage-b",
            "stage-c", "locked-test",
        )
    ]
    assert latent == [(stage, seed) for seed in (0, 1, 2) for stage in ("A", "B", "C")]


def test_flatten_overrides_and_mean_sd_are_deterministic():
    assert flatten_overrides({"a": {"b": [1, 2]}, "flag": True}) == [
        "a.b=[1,2]", "flag=true"
    ]
    values = pd.DataFrame({"seed": [0, 1, 2], "macro_f1": [0.6, 0.7, 0.8]})
    summary = _mean_sd(values, [])
    assert np.isclose(summary.iloc[0]["macro_f1__mean"], 0.7)
    assert round(summary.iloc[0]["macro_f1__std"], 8) == 0.1


def test_notebook_30_is_valid_and_targets_only_fixed_runner():
    path = Path("notebooks/30_colab_recommended_private_supcon_3seed.ipynb")
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "training.recommended_private_run" in source
    assert "config/recommended_private_3seed.yaml" in source
    assert "MAX_NEW_SEEDS = 3" in source
    assert "EXECUTE = False" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook30:cell{index}", "exec")
