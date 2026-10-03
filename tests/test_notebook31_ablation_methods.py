from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from types import SimpleNamespace

from models.encoder import build_encoder, resolve_encoder_config
from models.moe import ClassConditionalMoEDatasetNIDS, MoEDatasetNIDS
from training.compact_soft_moe_run import CompactSoftMoEStudy
from training.checkpoint import load_stage_b
from training.config import load_config
from training.dataset import PreparedData, PreparedSplit
from training.model_utils import apply_stage_c_trainability, build_model
from training.out_of_core_data import OutOfCoreContext
from training.out_of_core_train import run_stage_a_ooc, run_stage_b_ooc
from training.pooled_replay import (
    build_stratified_replay_reservoir,
    draw_class_balanced_replay_rows,
    replay_rows_for_owned_batch,
)


ROOT_CONFIG = "config/default.yaml"
CONFIGS = {
    "stage_c": (
        "config/compact_soft_moe_stage_c_lr3e4_3seed.yaml", 20_380
    ),
    "expert": ("config/compact_soft_moe_depth_expert_3seed.yaml", 37_020),
    "encoder": ("config/compact_soft_moe_depth_encoder_3seed.yaml", 36_892),
    "both": ("config/compact_soft_moe_depth_both_3seed.yaml", 53_532),
    "pooled": ("config/compact_soft_moe_pooled_b10_3seed.yaml", 20_380),
    "deeper_gate": (
        "config/compact_soft_moe_deeper_gate_3seed.yaml", 22_332
    ),
    "frozen_deeper_gate": (
        "config/compact_soft_moe_deeper_gate_frozen_stagec_3seed.yaml", 22_332
    ),
    "frozen_linear_gate_lr1e3": (
        "config/compact_soft_moe_frozen_stagec_linear_gate_lr1e3_3seed.yaml", 20_380
    ),
    "frozen_linear_gate_lr3e4": (
        "config/compact_soft_moe_frozen_stagec_linear_gate_lr3e4_3seed.yaml", 20_380
    ),
    "frozen_deeper_gate_lr3e4": (
        "config/compact_soft_moe_frozen_stagec_deeper_gate_lr3e4_3seed.yaml", 22_332
    ),
    "global_residual": (
        "config/compact_soft_moe_global_residual_3seed.yaml", 21_810
    ),
    "class_conditional": (
        "config/compact_soft_moe_class_conditional_3seed.yaml", 20_468
    ),
    "class_conditional_frozen_deeper_gate": (
        "config/compact_soft_moe_class_conditional_deeper_gate_frozen_stagec_3seed.yaml",
        22_420,
    ),
}

NOTEBOOK32_CONFIGS = (
    "config/compact_soft_moe_frozen_stagec_linear_gate_lr1e3_3seed.yaml",
    "config/compact_soft_moe_frozen_stagec_linear_gate_lr3e4_3seed.yaml",
    "config/compact_soft_moe_deeper_gate_frozen_stagec_3seed.yaml",
    "config/compact_soft_moe_frozen_stagec_deeper_gate_lr3e4_3seed.yaml",
)


@pytest.mark.parametrize("_name,item", CONFIGS.items())
def test_ablation_configs_lock_expected_topology(_name, item):
    path, expected = item
    study = yaml.safe_load(Path(path).read_text())
    model_cfg = study["backbone"]["model"]
    encoder = build_encoder(
        47, model_cfg["latent_dim"], resolve_encoder_config(model_cfg)
    )
    model = build_model(
        study["backbone"]["architecture"],
        encoder,
        ["UNSW", "ToN", "BoT", "CIC"],
        [f"class_{index}" for index in range(22)],
        model_cfg,
    )
    assert study["seeds"] == [0, 1, 2]
    assert study["expected_total_parameters"] == expected
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_ablation_checkpoint_reuse_removes_only_unchanged_stages(tmp_path):
    expected = {
        "stage_c": ["C"],
        "expert": ["B", "C"],
        "encoder": ["A", "B", "C"],
        "both": ["B", "C"],
        "pooled": ["B", "C"],
        "deeper_gate": ["C"],
        "frozen_deeper_gate": ["C"],
        "frozen_linear_gate_lr1e3": ["C"],
        "frozen_linear_gate_lr3e4": ["C"],
        "frozen_deeper_gate_lr3e4": ["C"],
        "global_residual": ["B", "C"],
        "class_conditional": ["C"],
        "class_conditional_frozen_deeper_gate": ["C"],
    }
    for name, (path, _parameters) in CONFIGS.items():
        runner = CompactSoftMoEStudy(
            base_config_path=ROOT_CONFIG,
            study_config_path=path,
            prefix=f"test_{name}",
            summary_dir=str(tmp_path / name),
            execute=False,
        )
        assert runner._final_nested(0)["training"]["stages"] == expected[name]


def test_notebook32_is_complete_frozen_body_gate_depth_lr_factorial():
    combinations = set()
    for path in NOTEBOOK32_CONFIGS:
        study = yaml.safe_load(Path(path).read_text())
        backbone = study["backbone"]
        gate_dims = tuple(backbone["model"]["gate"]["hidden_dims"])
        stage_c_lr = backbone["training"]["stage_c"]["optimizer"]["lr"]
        combinations.add((gate_dims, stage_c_lr))
        assert study["reuse_stage_a_from_prefix"] == study["reuse_stage_b_from_prefix"]
        assert backbone["training"]["stage_c_unfreeze"] == "none"
        assert backbone["training"]["stage_c"]["freeze_experts"] is True
        encoder = build_encoder(47, 64, backbone["model"]["encoder"])
        model = build_model(
            backbone["architecture"], encoder, ["A", "B", "C", "D"],
            [f"class_{index}" for index in range(22)], backbone["model"],
        )
        apply_stage_c_trainability(model, backbone)
        expected_trainable = 260 if not gate_dims else 2_212
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_trainable
    assert combinations == {((), 0.001), ((), 0.0003), ((32,), 0.001), ((32,), 0.0003)}


def test_notebook40_changes_only_class_conditional_reliability():
    reference = yaml.safe_load(Path(
        "config/compact_soft_moe_deeper_gate_frozen_stagec_3seed.yaml"
    ).read_text())
    candidate = yaml.safe_load(Path(
        "config/compact_soft_moe_class_conditional_deeper_gate_frozen_stagec_3seed.yaml"
    ).read_text())
    assert reference["seeds"] == candidate["seeds"] == [0, 1, 2]
    assert reference["reuse_stage_a_from_prefix"] == candidate["reuse_stage_a_from_prefix"]
    assert reference["reuse_stage_b_from_prefix"] == candidate["reuse_stage_b_from_prefix"]
    reference_backbone = copy.deepcopy(reference["backbone"])
    candidate_backbone = copy.deepcopy(candidate["backbone"])
    assert reference_backbone.pop("architecture") == "moe_dataset_soft"
    assert candidate_backbone.pop("architecture") == "moe_dataset_class_conditional"
    assert candidate_backbone["training"]["stage_c"].pop("lambda_reliability") == 0.0001
    assert reference_backbone == candidate_backbone

    model_cfg = candidate["backbone"]["model"]
    reference_model = build_model(
        reference["backbone"]["architecture"],
        build_encoder(47, 64, model_cfg["encoder"]),
        ["A", "B", "C", "D"],
        [f"class_{index}" for index in range(22)],
        model_cfg,
    )
    model = build_model(
        candidate["backbone"]["architecture"],
        build_encoder(47, 64, model_cfg["encoder"]),
        ["A", "B", "C", "D"],
        [f"class_{index}" for index in range(22)],
        model_cfg,
    )
    apply_stage_c_trainability(model, candidate["backbone"])
    assert model.class_reliability.shape == (4, 22)
    assert torch.count_nonzero(model.class_reliability) == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 22_420
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 2_300
    model.encoder.load_state_dict(reference_model.encoder.state_dict())
    model.expert_bank.load_state_dict(reference_model.expert_bank.state_dict())
    model.gate.load_state_dict(reference_model.gate.state_dict())
    reference_model.eval(); model.eval()
    inputs = torch.randn(11, 47)
    assert torch.allclose(
        reference_model(inputs)["combined_probs"],
        model(inputs)["combined_probs"],
        atol=1e-7,
    )


def test_zero_reliability_is_exactly_the_original_probability_mixture():
    torch.manual_seed(9)
    model_cfg = {
        "latent_dim": 5,
        "encoder": {"hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [], "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }
    encoder = build_encoder(4, 5, model_cfg["encoder"])
    ordinary = build_model(
        "moe_dataset_soft", encoder, ["A", "B"], ["ok", "attack", "rare"], model_cfg
    )
    conditional = build_model(
        "moe_dataset_class_conditional",
        copy.deepcopy(ordinary.encoder),
        ["A", "B"],
        ["ok", "attack", "rare"],
        model_cfg,
    )
    conditional.expert_bank.load_state_dict(ordinary.expert_bank.state_dict())
    conditional.gate.load_state_dict(ordinary.gate.state_dict())
    inputs = torch.randn(13, 4)
    base = ordinary(inputs)["combined_probs"]
    output = conditional(inputs)
    assert isinstance(conditional, ClassConditionalMoEDatasetNIDS)
    assert output["class_gate_weights"].shape == (13, 2, 3)
    assert torch.allclose(base, output["combined_probs"], atol=1e-7)
    assert torch.allclose(output["combined_probs"].sum(dim=1), torch.ones(13))


def test_class_reliability_receives_task_gradient():
    torch.manual_seed(4)
    model_cfg = {
        "latent_dim": 4,
        "encoder": {"hidden_dims": [6], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [], "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }
    model = build_model(
        "moe_dataset_class_conditional",
        build_encoder(3, 4, model_cfg["encoder"]),
        ["A", "B"],
        ["ok", "attack", "rare"],
        model_cfg,
    )
    output = model(torch.randn(8, 3))["combined_probs"]
    loss = torch.nn.functional.nll_loss(output.log(), torch.arange(8) % 3)
    loss.backward()
    assert model.class_reliability.grad is not None
    assert torch.count_nonzero(model.class_reliability.grad) > 0


def test_pooled_replay_is_deterministic_stratified_and_ten_percent():
    labels = np.repeat(np.arange(3, dtype=np.int64), 40)
    datasets = np.tile(np.repeat(np.arange(2, dtype=np.int64), 20), 3)
    first = build_stratified_replay_reservoir(
        labels, datasets, max_per_class_dataset=7, seed=31, chunk_rows=17
    )
    second = build_stratified_replay_reservoir(
        labels, datasets, max_per_class_dataset=7, seed=31, chunk_rows=17
    )
    assert set(first) == {0, 1, 2}
    assert all(len(first[class_id]) == 14 for class_id in first)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert replay_rows_for_owned_batch(512, 0.10) == 57
    rows = draw_class_balanced_replay_rows(first, 12, np.random.default_rng(5))
    counts = np.bincount(labels[rows], minlength=3)
    assert counts.tolist() == [4, 4, 4]
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        replay_rows_for_owned_batch(512, 1.0)


def test_ooc_stage_b_records_owned_and_pooled_replay_exposure(tmp_path):
    rng = np.random.default_rng(19)

    def split(rows_per_dataset):
        labels = np.tile(np.arange(3, dtype=np.int64), rows_per_dataset * 2 // 3)
        dataset_ids = np.repeat(np.arange(2, dtype=np.int64), rows_per_dataset)
        value = PreparedSplit(
            rng.normal(size=(rows_per_dataset * 2, 4)).astype(np.float32),
            labels,
            dataset_ids,
            np.where(dataset_ids == 0, "A", "B"),
        )
        value.dataset_slices = {
            "A": slice(0, rows_per_dataset),
            "B": slice(rows_per_dataset, rows_per_dataset * 2),
        }
        return value

    data = PreparedData(None, ["benign", "attack", "rare"], ["A", "B"], split(18), split(6), split(6))
    context = OutOfCoreContext(
        data,
        {name: SimpleNamespace(class_names=tuple(data.class_names)) for name in data.active_datasets},
        {"A": "split-a", "B": "split-b"},
        "prepared",
        "unused",
        [f"f{index}" for index in range(4)],
    )
    config = load_config(ROOT_CONFIG)
    config.update({"seed": 0, "architecture": "moe_dataset_soft"})
    config["data"]["active_datasets"] = ["A", "B"]
    config["model"] = {
        "latent_dim": 5,
        "encoder": {"hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "expert": {"hidden_dims": [], "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }
    config["training"].update({
        "device": "cpu", "batch_size": 6, "epochs_a": 1, "epochs_b": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "shuffle_block_rows": 12, "shuffle_buffer_blocks": 2,
        "progress_every_rows": 10_000, "save_epoch_history": True,
    })
    config["training"]["representation"] = {
        "objective": "ce", "sampling": "legacy", "class_weighting": "legacy",
        "weight": 0.1, "temperature": 0.1, "center_weight": 0.01,
        "arc_margin": 0.3, "arc_scale": 30.0,
    }
    config["training"]["stage_a"] = {
        "optimizer": {"lr": 0.001, "weight_decay": 0.0}
    }
    config["training"]["stage_b"] = {
        "warmstart_mode": "dataset", "replay_fraction": 0.10,
        "replay_pool_per_class_dataset": 4,
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
    }
    run_stage_a_ooc(config, context)
    run_stage_b_ooc(config, context)
    summary = load_stage_b(config["training"]["checkpoint_dir"])["training_summary"]
    assert summary["owned_examples_seen"] == 36
    assert summary["replay_examples_seen"] == 6
    assert summary["examples_seen"] == 42
    history = pd.read_csv(
        Path(config["training"]["checkpoint_dir"]) / "Training_History.csv"
    )
    stage_b = history[history["stage"] == "B"]
    assert stage_b["replay_rows"].tolist() == [3, 3]


@pytest.mark.parametrize(
    "filename",
    [
        "31c.ipynb",
        "31depth.ipynb",
        "31_pooledB.ipynb",
        "31_deeper_C.ipynb",
        "31_frozen_encoder_expert.ipynb",
        "31_global-plus-residual.ipynb",
        "31_class_conditional_routing.ipynb",
        "32_frozen_body_gate_depth_lr.ipynb",
        "40_class_conditional_frozen_deep_gate.ipynb",
    ],
)
def test_requested_notebooks_are_valid_and_default_to_dry_run(filename):
    notebook = json.loads((Path("notebooks") / filename).read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "EXECUTE = False" in source
    assert "training.compact_soft_moe_run" in source
    assert "Final_Study_Manifest.json" in source
