from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from models.encoder import build_encoder, resolve_encoder_config
from training.checkpoint import load_progress, stage_complete
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


def test_compact_soft_runner_can_target_exactly_one_seed(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    commands = []
    monkeypatch.setattr(runner, "_seed_complete", lambda seed: False)
    monkeypatch.setattr(
        runner, "_run", lambda label, nested: commands.append((label, nested["seed"]))
    )
    monkeypatch.setattr(runner, "_materialize_latent", lambda seed, stage: None)
    monkeypatch.setattr(runner, "_materialize_training_report", lambda seed: None)

    runner.run(only_seed=1)

    assert commands == [("train-and-test", 1)]
    with pytest.raises(ValueError, match="only_seed"):
        runner.run(only_seed=9)


def test_seed_completion_is_verified_and_committed_atomically(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.execute = True
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir(); result_dir.mkdir()
    checkpoints = {}
    for stage in ("A", "B", "C"):
        path = checkpoint_dir / f"stage_{stage.lower()}.pt"
        torch.save({"stage": stage, "value": torch.tensor([1.0])}, path)
        checkpoints[stage] = str(path)
    reports = [
        result_dir / "report.json",
        result_dir / "report.csv",
        result_dir / "report.npz",
    ]
    reports[0].write_text(json.dumps({"ok": True}))
    pd.DataFrame([{"ok": 1}]).to_csv(reports[1], index=False)
    np.savez(reports[2], values=np.asarray([1, 2, 3]))

    monkeypatch.setattr(runner, "_checkpoint_paths", lambda seed: checkpoints)
    monkeypatch.setattr(runner, "_required_report_paths", lambda seed: [str(path) for path in reports])
    monkeypatch.setattr(runner, "_seed_artifacts_complete", lambda seed: True)
    monkeypatch.setattr(runner, "_result_dir", lambda seed: str(result_dir))
    monkeypatch.setattr(runner, "_selected_stage_c_epoch", lambda seed: 2)

    manifest = runner._commit_seed_completion(0)

    manifest_path = result_dir / "completion_manifest.json"
    assert manifest_path.is_file()
    assert manifest["selected_stage_c_epoch"] == 2
    assert runner._seed_complete(0)
    state = json.loads((tmp_path / "summary" / "study_state.json").read_text())
    assert state["seeds"]["0"]["status"] == "complete"

    torch.save({"corrupted_after_commit": True}, checkpoints["C"])
    assert not runner._seed_complete(0)


def test_recovery_audit_finds_epoch_progress_and_snapshots_it(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoints = {
        "A": str(checkpoint_dir / "stage_a_encoder.pt"),
        "B": str(checkpoint_dir / "stage_b_expert_bank.pt"),
        "C": str(checkpoint_dir / "stage_c_full.pt"),
    }
    torch.save({"encoder_state": {}}, checkpoints["A"])
    torch.save({"expert_bank_state": {}}, checkpoints["B"])
    torch.save(
        {"epoch": 2, "model_state": {}, "optimizer_state": {}},
        checkpoint_dir / "stage_c_progress.pt",
    )
    monkeypatch.setattr(runner, "_checkpoint_paths", lambda seed: checkpoints)
    monkeypatch.setattr(runner, "_seed_artifacts_complete", lambda seed: False)

    status = runner.audit_seed(1)
    snapshot = runner.create_recovery_snapshot(1)

    assert status["status"] == "resumable"
    assert status["progress"]["stage"] == "C"
    assert status["progress"]["epoch"] == 2
    assert snapshot is not None
    snapshot_manifest = json.loads(
        (next((tmp_path / "summary" / "recovery_snapshots").iterdir()) / "snapshot_manifest.json").read_text()
    )
    assert any(row["source"].endswith("stage_c_progress.pt") for row in snapshot_manifest["files"])


def test_unreadable_progress_is_quarantined_and_final_is_not_complete(tmp_path):
    progress = tmp_path / "stage_c_progress.pt"
    final = tmp_path / "stage_c_full.pt"
    progress.write_bytes(b"not a torch checkpoint")
    final.write_bytes(b"not a torch checkpoint")

    assert load_progress(str(tmp_path), "C") is None
    assert not progress.exists()
    assert list(tmp_path.glob("stage_c_progress.pt.corrupt-*"))
    assert not stage_complete(str(tmp_path), "C")
