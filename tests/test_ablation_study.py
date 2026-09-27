from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data import paths
from training.ablation_matrix import (
    GreedyStudyRunner,
    RunSpec,
    expected_evaluated_run_count,
    flatten_overrides,
    load_study,
)


STUDY_PATH = "config/greedy_moe_study.yaml"


def _runner(tmp_path, monkeypatch, *, execute=False):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "outputs"))
    return GreedyStudyRunner(
        base_config_path="config/default.yaml",
        study_config_path=STUDY_PATH,
        prefix="unit_greedy",
        summary_dir=str(tmp_path / "summary"),
        execute=execute,
    )


def _materialize(runner: GreedyStudyRunner, spec: RunSpec, validation: float, test: float) -> None:
    checkpoint_dir, result_dir = runner._paths(spec)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    (Path(checkpoint_dir) / "stage_c_full.pt").write_bytes(b"checkpoint")
    (Path(checkpoint_dir) / "stage_c_training_summary.json").write_text(json.dumps({
        "selection_mode": "best_val",
        "best_validation_macro_f1": validation,
    }))
    pd.DataFrame([{
        "origin": "ALL", "accuracy": test, "balanced_accuracy": test,
        "macro_precision": test, "macro_recall": test, "macro_f1": test,
        "weighted_f1": test,
    }]).to_csv(Path(result_dir) / "Overall_Metrics.csv", index=False)
    pd.DataFrame([{
        "dataset": "A", "accuracy": test, "balanced_accuracy": test,
        "macro_f1": test, "weighted_f1": test,
    }]).to_csv(
        Path(result_dir) / "Per_Dataset_Metrics.csv", index=False
    )
    pd.DataFrame([{
        "origin": "ALL", "class": "Benign", "precision": test, "recall": test,
        "f1": test, "roc_auc_ovr": test, "pr_auc_ovr": test,
    }]).to_csv(
        Path(result_dir) / "Per_Class_Metrics.csv", index=False
    )
    pd.DataFrame([[1]], index=["Benign"], columns=["Benign"]).to_csv(
        Path(result_dir) / "Confusion_Matrix.csv"
    )
    pd.DataFrame([{
        "total_parameters": 1, "active_parameters_per_sample_mean": 1,
        "forward_macs_per_sample_mean": 1, "total_optimizer_steps": 1,
        "total_wall_seconds": 1.0,
    }]).to_csv(
        Path(result_dir) / "Resource_Accounting.csv", index=False
    )
    for filename in (
        "Trials.csv", "Gate_By_Dataset.csv", "Expert_Performance_By_Dataset.csv",
        "Expert_Utilization.csv", "Representation_Batch_Coverage.csv",
    ):
        pd.DataFrame([{"value": 1}]).to_csv(Path(result_dir) / filename, index=False)
    (Path(result_dir) / "manifest.json").write_text("{}")


def test_finalized_study_contract_and_run_count():
    study = load_study(STUDY_PATH)
    backbone = study["backbone"]
    assert expected_evaluated_run_count(study) == 26
    assert study["seeds"] == [0, 1, 2]
    assert study["capacity_candidates"][-1] == [256, 128]
    assert backbone["model"]["expert"]["hidden_dims"] == []
    assert backbone["training"]["epochs_a"] == 30
    assert backbone["training"]["epochs_b"] == 10
    assert backbone["training"]["epochs_c"] == 30
    assert backbone["training"]["selection_mode"] == "best_val"
    assert backbone["training"]["stage_c"]["gate_supervision"] == "none"
    assert backbone["training"]["stage_c"]["expert_update_policy"] == "all"
    assert backbone["training"]["representation"]["sampling"] == "legacy"
    assert backbone["training"]["representation"]["class_weighting"] == "legacy"


def test_flattened_overrides_preserve_lists_and_booleans():
    assert flatten_overrides({"a": {"b": [128, 64], "enabled": True}}) == [
        "a.b=[128,64]", "a.enabled=true"
    ]


def test_capacity_phase_changes_only_expert_head_and_reuses_stage_a(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    baseline = runner.baseline_specs()[0]
    capacities = runner.capacity_coarse_specs()
    assert [list(spec.expert_hidden_dims) for spec in capacities] == [
        [32], [45], [64], [128], [128, 64], [256, 128]
    ]
    base_config = runner._condition_nested(baseline)
    for candidate in capacities:
        candidate_config = runner._condition_nested(candidate)
        candidate_config["model"]["expert"]["hidden_dims"] = []
        assert candidate_config == base_config
        assert runner._stage_a_path(candidate) == runner._stage_a_path(baseline)


def test_private_and_adapter_candidates_use_winning_capacity(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    runner.state["decisions"]["capacity"] = {
        "expert_hidden_dims": [128, 64], "run_ids": [],
    }
    specs = runner.architecture_specs()
    assert len(specs) == 6
    assert {spec.architecture for spec in specs} == {
        "moe_dataset_private_encoders", "moe_dataset_adapters"
    }
    assert all(spec.expert_hidden_dims == (128, 64) for spec in specs)
    assert all(spec.adapter_rank == 16 for spec in specs)


def test_selection_uses_validation_not_test_and_capacity_tie_prefers_smaller(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    small = RunSpec("1", "capacity_32", 0, "moe_dataset_soft", (32,))
    large = RunSpec("1", "capacity_256x128", 0, "moe_dataset_soft", (256, 128))
    # The smaller model has a much worse test score, but ties on validation.
    _materialize(runner, small, validation=0.7, test=0.1)
    _materialize(runner, large, validation=0.7, test=0.99)
    winner = runner._best_by_score(
        [large, small],
        tie_key=lambda spec: sum(spec.expert_hidden_dims),
    )
    assert winner == small
    _materialize(runner, large, validation=0.71, test=0.01)
    assert runner._best_by_score(
        [small, large], tie_key=lambda spec: sum(spec.expert_hidden_dims)
    ) == large


def test_protocol_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    Path(runner.summary_dir).mkdir(parents=True)
    Path(runner.state_path).write_text(json.dumps({
        "format_version": 1,
        "protocol_hash": "wrong",
    }))
    with pytest.raises(ValueError, match="Study protocol changed"):
        _runner(tmp_path, monkeypatch)


def test_notebooks_are_valid_json_and_code_cells_compile():
    expected = [
        "18_colab_ablation_phase0_baseline.ipynb",
        "19_colab_ablation_phase1_capacity.ipynb",
        "20_colab_ablation_phase2_expert_architectures.ipynb",
        "21_colab_ablation_phase3a_supcon_sweep.ipynb",
        "22_colab_ablation_phase3b_supcon_confirm.ipynb",
        "23_colab_ablation_phase3c_balanced_supcon.ipynb",
        "24_colab_ablation_phase4_final_report.ipynb",
    ]
    for name in expected:
        path = Path("notebooks") / name
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:cell{index}", "exec")


def test_complete_synthetic_study_selects_winners_and_phase4_does_not_retrain(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, execute=True)

    baseline = runner.baseline_specs()
    for spec, score in zip(baseline, (0.60, 0.61, 0.62)):
        _materialize(runner, spec, score, score - 0.1)
    runner.run_phase_0()

    coarse = runner.capacity_coarse_specs()
    for spec in coarse:
        score = 0.70 if spec.expert_hidden_dims == (64,) else 0.63
        _materialize(runner, spec, score, 0.99 if spec.expert_hidden_dims == (256, 128) else 0.2)
    confirmations = [RunSpec("1", "capacity_64", seed, "moe_dataset_soft", (64,)) for seed in (1, 2)]
    for spec, score in zip(confirmations, (0.69, 0.68)):
        _materialize(runner, spec, score, 0.2)
    runner.run_phase_1()
    assert runner.state["decisions"]["capacity"]["expert_hidden_dims"] == [64]

    architecture_specs = runner.architecture_specs()
    for spec in architecture_specs:
        score = 0.74 if spec.architecture == "moe_dataset_adapters" else 0.67
        _materialize(runner, spec, score, 0.2)
    runner.run_phase_2()
    assert runner.state["decisions"]["architecture"]["architecture"] == "moe_dataset_adapters"

    sweep = runner.supcon_sweep_specs()
    for spec in sweep:
        score = 0.76 if spec.representation_weight == 0.2 else 0.70
        _materialize(runner, spec, score, 0.2)
    runner.run_phase_3a()
    assert runner.state["decisions"]["supcon_weight"]["weight"] == 0.2

    confirmed = runner._confirmed_supcon_specs()
    for spec, score in zip(confirmed[1:], (0.75, 0.74)):
        _materialize(runner, spec, score, 0.2)
    runner.run_phase_3b()
    assert runner.state["decisions"]["representation_preliminary"]["objective"] == "supcon"

    balanced = runner._balanced_specs()
    for spec, score in zip(balanced, (0.78, 0.77, 0.76)):
        _materialize(runner, spec, score, 0.3)
    runner.run_phase_3c()
    assert runner.state["decisions"]["representation"]["objective"] == "balanced_supcon"

    monkeypatch.setattr(
        "training.ablation_matrix.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phase 4 retrained")),
    )
    runner.run_phase_4()
    assert runner.state["phases"]["4"]["status"] == "complete"
    assert (Path(runner.summary_dir) / "Final_Study_Manifest.json").is_file()
    assert (Path(runner.summary_dir) / "Final_Overall_Summary.csv").is_file()
