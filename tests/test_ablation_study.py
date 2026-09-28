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
STUDY_V2_PATH = "config/greedy_moe_study_v2.yaml"


def _runner(tmp_path, monkeypatch, *, execute=False, study_path=STUDY_PATH):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "outputs"))
    return GreedyStudyRunner(
        base_config_path="config/default.yaml",
        study_config_path=study_path,
        prefix="unit_greedy",
        summary_dir=str(tmp_path / "summary"),
        execute=execute,
    )


def _materialize_latent(
    runner: GreedyStudyRunner,
    spec: RunSpec,
    *,
    stages=("C",),
    plot: bool | None = None,
) -> None:
    latent_dir = Path(runner._latent_dir(spec))
    latent_dir.mkdir(parents=True, exist_ok=True)
    snapshot_for_stage = {
        "A": "A__shared_initialization", "B": "B__gate", "C": "C__gate",
    }
    snapshots = [snapshot_for_stage[stage] for stage in stages]
    (latent_dir / "Latent_Report_Config.json").write_text(json.dumps({
        "stages": list(stages), "snapshots": snapshots,
    }))
    pd.DataFrame([{
        "snapshot": snapshot, "stage": snapshot.split("__", 1)[0], "encoder": "gate",
        "representation": "gate", "equivalent_to": None, "rows": 8,
        "silhouette": 0.5,
    } for snapshot in snapshots]).to_csv(latent_dir / "Latent_Snapshot_Metrics.csv", index=False)
    pd.DataFrame([{
        "snapshot": snapshot, "stage": snapshot.split("__", 1)[0], "encoder": "gate",
        "class": "Benign", "support": 8, "silhouette": 0.5,
    } for snapshot in snapshots]).to_csv(latent_dir / "Latent_Per_Class.csv", index=False)
    pd.DataFrame([{
        "snapshot": snapshot, "stage": snapshot.split("__", 1)[0], "encoder": "gate",
        "probe": "linear", "target": "class", "accuracy": 0.8,
    } for snapshot in snapshots]).to_csv(latent_dir / "Latent_Probe_Metrics.csv", index=False)
    pd.DataFrame([{
        "snapshot": snapshot, "stage": snapshot.split("__", 1)[0], "encoder": "gate",
        "probe": "linear", "class": "Benign", "recall": 0.8, "support": 8,
    } for snapshot in snapshots]).to_csv(latent_dir / "Latent_Probe_Per_Class.csv", index=False)
    should_plot = runner._default_latent_plot(spec) if plot is None else plot
    if should_plot:
        figure_dir = latent_dir / "latent_figures"
        figure_dir.mkdir(exist_ok=True)
        for snapshot in snapshots:
            (figure_dir / f"{snapshot}__umap.png").write_bytes(b"png")


def _materialize(
    runner: GreedyStudyRunner,
    spec: RunSpec,
    validation: float,
    test: float,
    *,
    latent: bool = True,
) -> None:
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
    if latent and runner.latent_enabled:
        _materialize_latent(runner, spec)


def test_finalized_study_contract_and_run_count():
    study = load_study(STUDY_PATH)
    study_v2 = load_study(STUDY_V2_PATH)
    backbone = study["backbone"]
    assert expected_evaluated_run_count(study) == 26
    assert expected_evaluated_run_count(study_v2) == 33
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
    assert study_v2["capacity_candidates"][0] == []
    assert study_v2["capacity_weight_decays"] == [0.0, 0.0001]


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


def test_v2_capacity_decay_challengers_and_checkpoint_identity(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, study_path=STUDY_V2_PATH)
    baseline = runner.baseline_specs()[0]
    challengers = runner.capacity_coarse_specs()

    assert len(challengers) == 13
    assert sum(spec.weight_decay == 0.0 for spec in challengers) == 6
    assert sum(spec.weight_decay == 0.0001 for spec in challengers) == 7
    assert not any(spec.expert_hidden_dims == () and spec.weight_decay == 0.0 for spec in challengers)
    linear_regularized = next(spec for spec in challengers if spec.expert_hidden_dims == ())
    capacity_32_zero = next(
        spec for spec in challengers
        if spec.expert_hidden_dims == (32,) and spec.weight_decay == 0.0
    )
    capacity_64_zero = next(
        spec for spec in challengers
        if spec.expert_hidden_dims == (64,) and spec.weight_decay == 0.0
    )
    capacity_32_regularized = next(
        spec for spec in challengers
        if spec.expert_hidden_dims == (32,) and spec.weight_decay == 0.0001
    )

    assert runner._stage_a_path(capacity_32_zero) == runner._stage_a_path(capacity_64_zero)
    assert runner._stage_a_path(capacity_32_zero) == runner._stage_a_path(baseline)
    assert runner._stage_a_path(capacity_32_regularized) != runner._stage_a_path(baseline)
    assert "wd0p0001" in linear_regularized.run_id
    assert "wd0p0001" in runner._run_name(linear_regularized)
    assert "wd0p0001" in runner._stage_a_name(linear_regularized)


def test_old_runspec_state_defaults_weight_decay_to_zero():
    restored = RunSpec.from_dict({
        "phase": "1",
        "condition": "capacity_32",
        "seed": 0,
        "architecture": "moe_dataset_soft",
        "expert_hidden_dims": [32],
        "representation_objective": "ce",
        "representation_weight": 0.1,
        "adapter_rank": 16,
    })
    assert restored.weight_decay == 0.0
    assert restored.run_id == "capacity_32:seed0"


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


def test_v2_selected_decay_propagates_through_later_phases(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, study_path=STUDY_V2_PATH)
    runner.state["decisions"]["capacity"] = {
        "expert_hidden_dims": [128, 64], "weight_decay": 0.0001, "run_ids": [],
    }
    architectures = runner.architecture_specs()
    assert all(spec.weight_decay == 0.0001 for spec in architectures)

    runner.state["decisions"]["architecture"] = {
        "architecture": "moe_dataset_adapters",
        "expert_hidden_dims": [128, 64],
        "adapter_rank": 16,
        "weight_decay": 0.0001,
        "run_ids": [],
    }
    assert all(spec.weight_decay == 0.0001 for spec in runner.supcon_sweep_specs())
    runner.state["decisions"]["supcon_weight"] = {"weight": 0.2, "seed0_run_id": "unused"}
    assert all(spec.weight_decay == 0.0001 for spec in runner._balanced_specs())


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


def test_v2_capacity_tie_prefers_smaller_then_lower_decay(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, study_path=STUDY_V2_PATH)
    zero = RunSpec("1", "capacity_64_wd0", 0, "moe_dataset_soft", (64,), weight_decay=0.0)
    regularized = RunSpec(
        "1", "capacity_64_wd0p0001", 0, "moe_dataset_soft", (64,),
        weight_decay=0.0001,
    )
    _materialize(runner, zero, validation=0.7, test=0.1)
    _materialize(runner, regularized, validation=0.7, test=0.99)
    winner = runner._best_by_score(
        [regularized, zero],
        tie_key=lambda spec: (
            sum(spec.expert_hidden_dims), spec.weight_decay,
        ),
    )
    assert winner == zero


def test_v2_phase1_confirms_best_challenger_then_can_fall_back_to_baseline(
    tmp_path, monkeypatch
):
    runner = _runner(
        tmp_path, monkeypatch, execute=True, study_path=STUDY_V2_PATH
    )
    for spec in runner.baseline_specs():
        _materialize(runner, spec, validation=0.60, test=0.10)
    runner.run_phase_0()

    challengers = runner.capacity_coarse_specs()
    winner = next(
        spec for spec in challengers
        if spec.expert_hidden_dims == (64,) and spec.weight_decay == 0.0001
    )
    for spec in challengers:
        _materialize(
            runner, spec,
            validation=0.70 if spec == winner else 0.50,
            test=0.99 if spec != winner else 0.01,
        )
    confirmations = [
        RunSpec(
            "1", winner.condition, seed, winner.architecture, winner.expert_hidden_dims,
            weight_decay=winner.weight_decay,
        )
        for seed in (1, 2)
    ]
    for spec in confirmations:
        _materialize(runner, spec, validation=0.40, test=0.99)

    runner.run_phase_1()
    assert runner.state["decisions"]["capacity_coarse"]["weight_decay"] == 0.0001
    assert runner.state["decisions"]["capacity"]["expert_hidden_dims"] == []
    assert runner.state["decisions"]["capacity"]["weight_decay"] == 0.0
    assert runner.state["decisions"]["capacity"]["condition"] == "baseline_linear"


def test_run_overrides_set_latent_eval_policy_by_phase(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, study_path=STUDY_V2_PATH)

    def latent_overrides(spec, **kwargs):
        overrides = runner._run_overrides(spec, **kwargs)
        prefix = "evaluation.latent."
        return {
            override.split("=", 1)[0][len(prefix):]: override.split("=", 1)[1]
            for override in overrides if override.startswith(prefix)
        }

    # Baseline/capacity sweeps: metrics logged, but no plots -- the shared
    # encoder is unchanged by these conditions, so a UMAP plot would be
    # redundant across every capacity/weight-decay variant.
    baseline = latent_overrides(RunSpec("0", "baseline_linear", 0, "moe_dataset_soft", ()))
    assert baseline["enabled"] == "true"
    assert baseline["plot"] == "false"
    assert baseline["stages"] == "[C]"
    capacity = latent_overrides(RunSpec("1", "capacity_64", 0, "moe_dataset_soft", (64,)))
    assert capacity["plot"] == "false"

    # Phase 2+: seed 0 is the representative run per condition/architecture
    # being compared, so it gets a plot; other seeds only log metrics.
    architecture_seed0 = latent_overrides(
        RunSpec("2", "adapters_r16", 0, "moe_dataset_adapters", (64,))
    )
    assert architecture_seed0["plot"] == "true"
    architecture_seed1 = latent_overrides(
        RunSpec("2", "adapters_r16", 1, "moe_dataset_adapters", (64,))
    )
    assert architecture_seed1["enabled"] == "true"
    assert architecture_seed1["plot"] == "false"

    # Phase 4 explicitly upgrades reused specs rather than inventing phase-4
    # identities, so the final policy is passed independently of spec.phase.
    final_seed2 = latent_overrides(
        RunSpec("3c", "balanced_supcon_w0p2", 2, "moe_dataset_adapters", (64,)),
        latent_stages=("A", "B", "C"), latent_plot=True,
    )
    assert final_seed2["plot"] == "true"
    assert final_seed2["stages"] == "[A,B,C]"

    # Fixed, not seed-derived: conditions sharing a data split_seed stay
    # comparable on the same sampled rows.
    assert architecture_seed0["random_seed"] == "0"
    assert architecture_seed1["random_seed"] == "0"


def test_v1_run_overrides_do_not_enable_v2_latent_policy(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, study_path=STUDY_PATH)
    overrides = runner._run_overrides(runner.baseline_specs()[0])
    assert not any(value.startswith("evaluation.latent.") for value in overrides)


def test_missing_v2_latent_report_is_backfilled_without_new_training_run(
    tmp_path, monkeypatch
):
    runner = _runner(
        tmp_path, monkeypatch, execute=True, study_path=STUDY_V2_PATH
    )
    spec = runner.baseline_specs()[0]
    _materialize(runner, spec, validation=0.7, test=0.6, latent=False)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert "--latent-only" in command
        _materialize_latent(runner, spec)

    monkeypatch.setattr("training.ablation_matrix.subprocess.run", fake_run)
    runner.ensure_specs([spec])

    assert len(calls) == 1
    assert runner.new_runs == 0
    assert not runner._missing_artifacts(spec)
    assert (Path(runner.summary_dir) / "Latent_Snapshot_Metrics_By_Run.csv").is_file()


def test_harvest_latent_metrics_appends_and_dedupes_by_run_id(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, execute=True)
    spec = RunSpec("2", "adapters_r16", 0, "moe_dataset_adapters", (64,))
    _materialize(runner, spec, validation=0.7, test=0.6)

    latent_dir = Path(runner._latent_dir(spec))
    latent_dir.mkdir(parents=True)
    _materialize_latent(runner, spec)

    runner._record_complete(spec)
    paths = [
        Path(runner.summary_dir) / value[1]
        for value in runner._LATENT_METRIC_FILES.values()
    ]
    assert all(path.is_file() for path in paths)
    first = {path: pd.read_csv(path) for path in paths}
    assert all(set(frame["run_id"]) == {spec.run_id} for frame in first.values())

    # Re-harvesting the same run must replace, not duplicate, its rows.
    runner._harvest_latent_metrics(spec)
    second = {path: pd.read_csv(path) for path in paths}
    assert all(len(second[path]) == len(first[path]) for path in paths)
    assert (Path(runner.summary_dir) / "Latent_Snapshot_Summary.csv").is_file()


def test_harvest_latent_metrics_is_a_noop_without_a_latent_report(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, execute=True)
    spec = RunSpec("0", "baseline_linear", 0, "moe_dataset_soft", ())
    _materialize(runner, spec, validation=0.7, test=0.6)
    runner._record_complete(spec)
    assert not list(Path(runner.summary_dir).glob("Latent_*_By_Run.csv"))


def test_phase4_upgrades_reused_baseline_and_winner_specs_to_all_stages(
    tmp_path, monkeypatch
):
    runner = _runner(
        tmp_path, monkeypatch, execute=True, study_path=STUDY_V2_PATH
    )
    baseline = runner.baseline_specs()
    winner = [
        RunSpec(
            "3c", "balanced_supcon_w0p2", seed, "moe_dataset_adapters", (64,),
            "balanced_supcon", 0.2,
        )
        for seed in runner.seeds
    ]
    for spec in [*baseline, *winner]:
        _materialize(runner, spec, validation=0.7 + spec.seed / 100, test=0.6)
        runner._record_complete(spec)
    runner.state["decisions"] = {
        "baseline": {"run_ids": [spec.run_id for spec in baseline]},
        "capacity": {
            "expert_hidden_dims": [64], "weight_decay": 0.0,
            "validation_macro_f1_mean": 0.7,
        },
        "architecture": {
            "choice": "adapters", "architecture": "moe_dataset_adapters",
            "adapter_rank": 16, "validation_macro_f1_mean": 0.7,
        },
        "representation": {
            "objective": "balanced_supcon", "weight": 0.2, "weight_decay": 0.0,
            "validation_macro_f1_mean": 0.71,
            "run_ids": [spec.run_id for spec in winner],
        },
    }
    by_name = {runner._run_name(spec): spec for spec in [*baseline, *winner]}
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert "--latent-only" in command
        overrides = [
            command[index + 1] for index, value in enumerate(command[:-1])
            if value == "--set"
        ]
        resolved = dict(value.split("=", 1) for value in overrides)
        spec = by_name[resolved["run_name"]]
        assert resolved["evaluation.latent.stages"] == "[A,B,C]"
        assert resolved["evaluation.latent.plot"] == "true"
        _materialize_latent(runner, spec, stages=("A", "B", "C"), plot=True)

    monkeypatch.setattr("training.ablation_matrix.subprocess.run", fake_run)
    runner.run_phase_4()

    assert len(calls) == 6
    assert runner.state["phases"]["4"]["status"] == "complete"
    for spec in [*baseline, *winner]:
        config = json.loads(
            (Path(runner._latent_dir(spec)) / "Latent_Report_Config.json").read_text()
        )
        assert config["stages"] == ["A", "B", "C"]
        assert set(config["snapshots"]) == {
            "A__shared_initialization", "B__gate", "C__gate",
        }
    assert (Path(runner.summary_dir) / "Final_Latent_Snapshot_Summary.csv").is_file()
    manifest = json.loads(
        (Path(runner.summary_dir) / "Final_Study_Manifest.json").read_text()
    )
    assert "Final_Latent_Snapshot_Summary.csv" in manifest["report_files"]


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
        source = "".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        assert "config/greedy_moe_study_v2.yaml" in source
        assert "nfv3_4way_greedy_v2" in source
        assert "run_id" in source
        assert "latent['phase'].astype(str) == PHASE" not in source
        if name.startswith("18_"):
            assert "EXECUTE = False" in source
        if name.startswith("24_"):
            assert "baseline_run_ids" in source
            assert "winner_run_ids" in source
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
    phase2_runs = pd.read_csv(Path(runner.summary_dir) / "Ablation_Phase2_Runs.csv")
    assert "run_id" in phase2_runs.columns
    assert set(runner.state["decisions"]["capacity"]["run_ids"]).issubset(
        set(phase2_runs["run_id"])
    )

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
    assert (Path(runner.summary_dir) / "Final_Decision_Summary.csv").is_file()
    assert (Path(runner.summary_dir) / "Final_Overall_Summary.csv").is_file()
    decision_summary = pd.read_csv(Path(runner.summary_dir) / "Final_Decision_Summary.csv")
    assert decision_summary.loc[0, "selected_weight_decay"] == 0.0
    assert decision_summary.loc[0, "weight_decay_scope"] == "bounded_comparison_stage_not_final_hpo"
