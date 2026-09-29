from __future__ import annotations

import yaml

from models.encoder import build_encoder, resolve_encoder_config
from training.compact_soft_moe_run import CompactSoftMoEStudy
from training.model_utils import build_model


ROOT_CONFIG = "config/default.yaml"
STUDY_CONFIG = "config/compact_soft_moe_3seed.yaml"


def _runner(tmp_path) -> CompactSoftMoEStudy:
    return CompactSoftMoEStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_compact_soft",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )


def test_compact_soft_study_locks_the_historical_contract():
    with open(STUDY_CONFIG) as handle:
        study = yaml.safe_load(handle)
    assert study["seeds"] == [0, 1, 2]
    assert study["expected_input_features"] == 47
    assert study["expected_classes"] == 22
    assert study["expected_total_parameters"] == 20_380
    backbone = study["backbone"]
    assert backbone["architecture"] == "moe_dataset_soft"
    assert backbone["model"]["encoder"]["hidden_dims"] == [128]
    assert backbone["model"]["latent_dim"] == 64
    assert backbone["model"]["expert"]["hidden_dims"] == []
    assert backbone["model"]["gate"]["hidden_dims"] == []
    assert backbone["model"]["gate"]["routing"] == "dense"
    assert backbone["training"]["epochs_a"] == 30
    assert backbone["training"]["epochs_b"] == 10
    assert backbone["training"]["epochs_c"] == 30
    assert backbone["training"]["selection_mode"] == "best_val"
    assert backbone["training"]["stage_c"]["gate_supervision"] == "none"
    assert backbone["training"]["stage_c"]["expert_update_policy"] == "all"
    assert backbone["training"]["deterministic"] is True


def test_compact_soft_topology_has_exact_historical_parameter_count(tmp_path):
    runner = _runner(tmp_path)
    config = runner._final_config(0)
    model_cfg = config["model"]
    encoder = build_encoder(
        47, model_cfg["latent_dim"], resolve_encoder_config(model_cfg, "encoder")
    )
    model = build_model(
        config["architecture"],
        encoder,
        ["UNSW", "ToN", "BoT", "CIC"],
        [f"class_{index}" for index in range(22)],
        model_cfg,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_380


def test_compact_soft_runner_binds_every_random_source_to_the_seed(tmp_path):
    runner = _runner(tmp_path)
    configs = [runner._final_config(seed) for seed in (0, 1, 2)]
    assert [config["seed"] for config in configs] == [0, 1, 2]
    assert [config["data"]["split_seed"] for config in configs] == [0, 1, 2]
    assert [config["evaluation"]["latent"]["random_seed"] for config in configs] == [0, 1, 2]
    assert all(config["training"]["deterministic"] for config in configs)
    assert all(config["training"]["save_epoch_history"] for config in configs)
    assert len({config["run_name"] for config in configs}) == 3


def test_compact_soft_dry_graph_runs_training_and_all_latent_stages(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    commands = []
    latent = []
    monkeypatch.setattr(runner, "_seed_complete", lambda seed: False)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda label, nested: commands.append((label, nested["seed"])),
    )
    monkeypatch.setattr(
        runner,
        "_materialize_latent",
        lambda seed, stage: latent.append((stage, seed)),
    )
    monkeypatch.setattr(runner, "_materialize_training_report", lambda seed: None)
    runner.run()
    assert commands == [("train-and-test", seed) for seed in (0, 1, 2)]
    assert latent == [
        (stage, seed) for seed in (0, 1, 2) for stage in ("A", "B", "C")
    ]
