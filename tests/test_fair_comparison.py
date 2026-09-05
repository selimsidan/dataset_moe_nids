from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.resource_accounting import parameter_count, resource_profile
from evaluation.seed_summary import aggregate_seed_results
from models.baselines import HardTwoStageModel
from models.encoder import SharedEncoder
from training.baseline_train import train_matched_dense
from training.checkpoint import load_validated_stage_a, save_stage_a, stage_a_metadata
from training.dataset import PreparedData, PreparedSplit
from training.dense_ooc import load_dense_ooc, run_dense_stage_b_ooc, run_dense_stage_c_ooc
from training.model_utils import build_matched_dense_model, build_model
from training.out_of_core_data import OutOfCoreContext
from training.out_of_core_train import ensure_run_contract, run_stage_a_ooc
from evaluation.out_of_core_report import evaluate_and_report_ooc


def _model_config(axis="total_params", route="dense"):
    return {
        "latent_dim": 16,
        "encoder": {"hidden_dims": [12], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [32, 16], "dropout": 0.0},
        "adapter": {"rank": 4, "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": route},
        "dense_match": {
            "axis": axis,
            "reference_routing": route,
            "tolerance": 0.005,
            "max_width": 256,
        },
    }


def _data(seed=0):
    rng = np.random.default_rng(seed)

    def split(n):
        return PreparedSplit(
            rng.normal(size=(n, 6)).astype(np.float32),
            np.tile(np.arange(4), n // 4).astype(np.int64),
            np.repeat(np.arange(2), n // 2).astype(np.int64),
            np.repeat(np.asarray(["A", "B"]), n // 2),
        )

    return PreparedData(None, ["Benign", "c1", "c2", "c3"], ["A", "B"], split(64), split(32), split(32))


def _training_config(tmp_path, axis="total_params", warmstart="matched_exposure"):
    return {
        "run_name": "fair-test", "seed": 0, "architecture": "matched_dense",
        "model": _model_config(axis, "dense" if axis == "total_params" else "top1"),
        "load_balance": {"lambda_balance": 0.1},
        "training": {
            "device": "cpu", "batch_size": 16, "min_per_class_per_batch": 1,
            "epochs_a": 1, "epochs_b": 1, "epochs_c": 1, "lr": 0.001,
            "weight_decay": 0.0, "stage_c_unfreeze": "last_layer",
            "selection_mode": "fixed_epochs", "early_stopping_patience": 2,
            "baseline": {"encoder_init": "stage_a", "stage_b_warmstart": warmstart, "stage_a_checkpoint": None},
            "stage_c": {"gate_supervision": "none", "expert_update_policy": "all"},
            "checkpoint_dir": str(tmp_path / "checkpoints"), "stages": ["A", "B", "C"],
            "shuffle_block_rows": 16, "shuffle_buffer_blocks": 2,
        },
        "evaluation": {"output_dir": str(tmp_path / "results"), "prediction_chunk_rows": 8, "roc_auc_bins": 16},
        "data": {"active_datasets": ["A", "B"]},
    }


@pytest.mark.parametrize("axis,route", [
    ("total_params", "dense"), ("active_params", "top1"), ("forward_macs", "top1")
])
@pytest.mark.parametrize("datasets", [2, 3, 4])
@pytest.mark.parametrize("classes", [5, 11])
def test_matched_dense_resolves_within_tolerance(axis, route, datasets, classes):
    cfg = _model_config(axis, route)
    model = build_matched_dense_model(
        SharedEncoder(6, [12], 16, dropout=0.0),
        [f"d{i}" for i in range(datasets)],
        [f"c{i}" for i in range(classes)],
        cfg,
    )
    assert model.match_info["relative_error"] <= 0.005
    assert model.match_info["hidden_dims"][0] >= model.match_info["hidden_dims"][1]
    assert [name for name in inspect.signature(model.forward).parameters if name != "self"] == ["x"]


def test_stage_a_metadata_validation_rejects_a_different_seed(tmp_path):
    data = _data()
    config = _training_config(tmp_path)
    metadata = stage_a_metadata(config, data)
    encoder = SharedEncoder(6, [12], 16, dropout=0.0)
    save_stage_a(config["training"]["checkpoint_dir"], encoder.state_dict(), data.class_names, metadata)
    load_validated_stage_a(config, metadata)
    wrong = dict(metadata); wrong["seed"] = 1
    with pytest.raises(ValueError, match="Incompatible Stage-A checkpoint"):
        load_validated_stage_a(config, wrong)


def test_matched_exposure_uses_reference_stage_b_batch_budget(tmp_path):
    data = _data()
    config = _training_config(tmp_path)
    encoder = SharedEncoder(6, [12], 16, dropout=0.0)
    save_stage_a(
        config["training"]["checkpoint_dir"], encoder.state_dict(), data.class_names,
        stage_a_metadata(config, data),
    )
    model = train_matched_dense(config, data)
    expected_steps = sum(max(1, int((data.train.dataset_idx == i).sum()) // 16) for i in range(2))
    assert model.training_summary["B"]["optimizer_steps"] == expected_steps
    assert model.training_summary["B"]["examples_seen"] == expected_steps * 16


def test_resource_accounting_distinguishes_dense_and_top1_active_budgets():
    cfg = _model_config("total_params", "dense")
    dense = build_model(
        "moe_dataset_soft", SharedEncoder(6, [12], 16, dropout=0.0),
        ["A", "B"], ["a", "b", "c", "d"], cfg,
    )
    config = {"architecture": "moe_dataset_soft", "model": cfg, "training": {"stage_c": {}}}
    dense_row = resource_profile(dense, config)
    dense.routing_mode = "top1"
    sparse_row = resource_profile(dense, config)
    expected_sparse = parameter_count(dense.encoder) + parameter_count(dense.gate) + parameter_count(dense.expert_bank.experts[0])
    assert dense_row["active_parameters_per_sample_mean"] == parameter_count(dense)
    assert sparse_row["active_parameters_per_sample_mean"] == expected_sparse
    assert sparse_row["forward_macs_per_sample_mean"] < dense_row["forward_macs_per_sample_mean"]


def test_hard_two_stage_active_budget_contains_both_sequential_encoders():
    model = HardTwoStageModel(["A", "B"], 6, 8, 4, [12], [12], dropout=0.0)
    config = {"architecture": "hard_two_stage", "model": {"gate": {}}, "training": {"stage_c": {}}}
    row = resource_profile(model, config)
    router = parameter_count(model.id_encoder) + parameter_count(model.id_head)
    selected = parameter_count(model.stage_b.encoders["A"]) + parameter_count(model.stage_b.heads["A"])
    assert row["active_parameters_per_sample_mean"] == router + selected


@pytest.mark.parametrize("axis", ["total_params", "active_params", "forward_macs"])
@pytest.mark.parametrize("warmstart", ["none", "matched_exposure"])
def test_ooc_matched_dense_smoke_and_resource_report(tmp_path, axis, warmstart):
    data = _data(3)
    for split in (data.train, data.val, data.test):
        half = len(split.class_idx) // 2
        split.dataset_slices = {"A": slice(0, half), "B": slice(half, len(split.class_idx))}
    prepared = {name: SimpleNamespace(class_names=tuple(data.class_names)) for name in data.active_datasets}
    context = OutOfCoreContext(data, prepared, {"A": "a", "B": "b"}, "prep", "unused", [f"f{i}" for i in range(6)])
    config = _training_config(tmp_path, axis=axis, warmstart=warmstart)
    contract = ensure_run_contract(config, context)
    run_stage_a_ooc(config, context)
    model = run_dense_stage_b_ooc(config, context)
    model = run_dense_stage_c_ooc(config, context, model)
    reports = evaluate_and_report_ooc(model, context, config, contract)
    assert set(reports["overall"]["origin"]) == {"A", "B", "ALL"}
    assert (tmp_path / "results" / "Resource_Accounting.csv").is_file()
    loaded = load_dense_ooc(config, context)
    assert loaded.match_info["axis"] == axis


def test_ooc_matched_exposure_resumes_without_repeating_completed_batches(tmp_path, monkeypatch):
    data = _data(5)
    for split in (data.train, data.val, data.test):
        half = len(split.class_idx) // 2
        split.dataset_slices = {"A": slice(0, half), "B": slice(half, len(split.class_idx))}
    prepared = {name: SimpleNamespace(class_names=tuple(data.class_names)) for name in data.active_datasets}
    context = OutOfCoreContext(data, prepared, {"A": "a", "B": "b"}, "prep", "unused", [f"f{i}" for i in range(6)])
    config = _training_config(tmp_path)
    run_stage_a_ooc(config, context)

    import training.checkpoint as checkpoint_module
    real_save = checkpoint_module.save_progress
    calls = {"count": 0}

    def save_then_interrupt(*args, **kwargs):
        real_save(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated disconnect")

    monkeypatch.setattr(checkpoint_module, "save_progress", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        run_dense_stage_b_ooc(config, context)
    monkeypatch.setattr(checkpoint_module, "save_progress", real_save)
    model = run_dense_stage_b_ooc(config, context)
    expected_steps = 4  # two 32-row datasets, 16-row batches, one epoch
    assert model.training_summary["B"]["optimizer_steps"] == expected_steps


def test_three_seed_aggregation_rejects_mixed_splits_and_reports_paired_delta():
    trials = pd.DataFrame([
        {"Trial_ID": f"{method}-{seed}", "architecture": method, "seed": seed,
         "split_signature": f"split-{seed}", "routing_mode": "dense"}
        for method in ("reference", "candidate") for seed in (0, 1, 2)
    ])
    overall = pd.DataFrame([
        {"Trial_ID": f"{method}-{seed}", "macro_f1": 0.7 + 0.01 * seed + (0.02 if method == "candidate" else 0)}
        for method in ("reference", "candidate") for seed in (0, 1, 2)
    ])
    per_dataset = pd.DataFrame([
        {"Trial_ID": f"{method}-{seed}", "dataset": "A", "macro_f1": 0.6 + (0.02 if method == "candidate" else 0)}
        for method in ("reference", "candidate") for seed in (0, 1, 2)
    ])
    summary = aggregate_seed_results(
        trials, overall, per_dataset, reference_method="reference|routing_mode=dense"
    )
    candidate = summary[(summary["method_id"] == "candidate|routing_mode=dense") & (summary["dataset"] == "ALL")]
    assert np.isclose(candidate.iloc[0]["paired_delta_mean"], 0.02)
    broken = trials.copy(); broken.loc[broken["Trial_ID"] == "candidate-0", "split_signature"] = "wrong"
    with pytest.raises(ValueError, match="mixes split signatures"):
        aggregate_seed_results(broken, overall, per_dataset, reference_method="reference|routing_mode=dense")
