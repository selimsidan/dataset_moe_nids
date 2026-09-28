from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from evaluation.validation_report import write_validation_report_ooc
from models.encoder import (
    ResidualMLPBlock,
    SharedEncoder,
    build_encoder,
    resolve_encoder_config,
)
from training.checkpoint import load_configured_stage_b, save_stage_b
from training.depth_search import (
    DepthRunSpec,
    TargetedDepthStudy,
    read_validation_artifact,
)
from training.config import load_config
from training.dataset import PreparedData, PreparedSplit
from training.model_utils import build_model, model_encoder_modules
from training.optim import optimizer_hparams
from training.out_of_core_data import OutOfCoreContext
from training.out_of_core_train import run_stage_a_ooc, run_stage_b_ooc, run_stage_c_ooc
from training.stage_c_jointfinetune import _set_encoder_trainable


ROOT_CONFIG = "config/default.yaml"
STUDY_CONFIG = "config/targeted_depth_search_v3.yaml"


@pytest.mark.parametrize(
    "normalization,norm_type",
    [("batchnorm", nn.BatchNorm1d), ("layernorm", nn.LayerNorm)],
)
def test_residual_encoder_shape_depth_and_normalization(normalization, norm_type):
    encoder = build_encoder(9, 7, {
        "kind": "residual_mlp", "width": 16, "blocks": 3,
        "expansion": 2.0, "normalization": normalization,
        "activation": "relu", "dropout": 0.1,
    })
    encoder.eval()
    assert encoder(torch.randn(5, 9)).shape == (5, 7)
    blocks = [module for module in encoder.modules() if isinstance(module, ResidualMLPBlock)]
    assert len(blocks) == 3
    assert all(isinstance(block.norm, norm_type) for block in blocks)


def test_mlp_factory_preserves_legacy_checkpoint_layout():
    legacy = SharedEncoder(6, [8, 4], 3, "relu", 0.2)
    factory = build_encoder(6, 3, {
        "hidden_dims": [8, 4], "activation": "relu", "dropout": 0.2,
    })
    assert list(legacy.state_dict()) == list(factory.state_dict())
    factory.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize("mode,expected_trainable", [("none", 0), ("last_layer", 2)])
def test_stage_c_freezing_modes_for_residual_encoder(mode, expected_trainable):
    encoder = build_encoder(5, 3, {
        "kind": "residual_mlp", "width": 8, "blocks": 2,
        "normalization": "layernorm", "dropout": 0.0,
    })
    _set_encoder_trainable(encoder, mode)
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    assert len(trainable) == expected_trainable


def test_asymmetric_private_model_uses_role_specific_encoder_topologies():
    model_cfg = {
        "latent_dim": 5,
        "encoder": {"kind": "mlp", "hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "gate_encoder": {"hidden_dims": [11]},
        "private_encoder": {
            "kind": "residual_mlp", "hidden_dims": [], "width": 13, "blocks": 2,
            "normalization": "layernorm", "activation": "relu", "dropout": 0.0,
        },
        "expert": {"hidden_dims": [6], "dropout": 0.0},
        "adapter": {"rank": 3, "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }
    gate_cfg = resolve_encoder_config(model_cfg, "gate_encoder")
    private_cfg = resolve_encoder_config(model_cfg, "private_encoder")
    gate = build_encoder(4, 5, gate_cfg)
    model = build_model(
        "moe_dataset_private_encoders", gate, ["A", "B"], ["ok", "attack"], model_cfg
    )
    assert model.encoder.kind == "mlp"
    assert all(encoder.kind == "residual_mlp" for encoder in model.expert_bank.encoders)
    assert len(model_encoder_modules(model)) == 3
    assert model(torch.randn(4, 4))["combined_probs"].shape == (4, 2)


def test_stage_specific_optimizer_falls_back_and_overrides():
    config = {
        "training": {
            "lr": 0.01, "weight_decay": 0.2,
            "stage_c": {"optimizer": {"lr": 0.0003, "weight_decay": 0.0001}},
        }
    }
    assert optimizer_hparams(config, "a") == (0.01, 0.2)
    assert optimizer_hparams(config, "c") == (0.0003, 0.0001)


def test_external_stage_b_rejects_wrong_bank_or_dataset_order(tmp_path):
    cache = tmp_path / "cache"
    save_stage_b(str(cache), {}, ["A", "B"], "full")
    config = {"training": {"stage_b_checkpoint": str(cache)}}
    load_configured_stage_b(
        config, expected_dataset_names=["A", "B"], expected_bank_kind="full"
    )
    with pytest.raises(ValueError, match="Incompatible Stage-B checkpoint"):
        load_configured_stage_b(
            config, expected_dataset_names=["B", "A"], expected_bank_kind="full"
        )
    with pytest.raises(ValueError, match="Incompatible Stage-B checkpoint"):
        load_configured_stage_b(
            config, expected_dataset_names=["A", "B"], expected_bank_kind="adapter"
        )


def test_selection_reader_structurally_blocks_test_artifacts(tmp_path):
    pd.DataFrame([{"macro_f1": 0.7}]).to_csv(
        tmp_path / "Validation_Overall_Metrics.csv", index=False
    )
    assert read_validation_artifact(
        str(tmp_path), "Validation_Overall_Metrics.csv"
    ).iloc[0]["macro_f1"] == 0.7
    with pytest.raises(ValueError, match="cannot read non-validation"):
        read_validation_artifact(str(tmp_path), "Overall_Metrics.csv")


def test_v3_dry_run_emits_complete_abc_commands_and_base_screen(tmp_path, capsys):
    runner = TargetedDepthStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_v3",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )
    assert len(runner.phase1_base_specs()) == 7
    runner.run_phase_1()
    output = capsys.readouterr().out
    assert "[stage-a]" in output
    assert "[stage-b]" in output
    assert "[evaluate]" in output
    assert "base structural screen is incomplete" in output


def test_hard_configuration_ceiling_is_enforced_before_launch(tmp_path):
    runner = TargetedDepthStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_ceiling",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )
    runner.state["runs"] = {str(index): {} for index in range(runner.ceiling)}
    spec = DepthRunSpec(
        phase="1", condition="overflow", seed=0,
        encoder=runner.encoders["shallow"],
    )
    with pytest.raises(RuntimeError, match="hard configuration ceiling"):
        runner.ensure_specs([spec])


def test_phase1_expansion_rule_is_deterministic(tmp_path, monkeypatch):
    runner = TargetedDepthStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_expansion",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )
    submitted = []

    def complete(specs):
        submitted.append([spec.condition for spec in specs])
        return True

    scores = {
        "shared_shallow_linear_wd0": 0.69,
        "shared_shallow_linear_wd1e4": 0.70,
        "shared_shallow_h64": 0.70,
        "shared_shallow_h256x128": 0.72,
        "shared_plain_deep_h64": 0.704,
        "shared_res2_bn_h64": 0.71,
        "shared_res2_ln_h64": 0.705,
    }

    def profile(spec):
        score = scores.get(spec.condition, 0.725)
        return score, score - 0.1, 0.5, 1000, 2000

    monkeypatch.setattr(runner, "ensure_specs", complete)
    monkeypatch.setattr(runner, "_profile", profile)
    runner.run_phase_1()
    assert submitted[0] == [spec.condition for spec in runner.phase1_base_specs()]
    assert submitted[1] == [
        "shared_cross_best_encoder_head",
        "shared_best_latent128",
        "shared_expand_expert_h256x128x64",
    ]
    assert runner.state["phases"]["1"]["status"] == "complete"


def test_phase3_uses_router_depth_diagnostic_when_gate_gain_clears_threshold(
    tmp_path, monkeypatch
):
    runner = TargetedDepthStudy(
        base_config_path=ROOT_CONFIG,
        study_config_path=STUDY_CONFIG,
        prefix="test_diagnostic",
        summary_dir=str(tmp_path / "summary"),
        execute=False,
    )
    leader = DepthRunSpec(
        phase="2", condition="leader", seed=0,
        encoder=copy.deepcopy(runner.encoders["shallow"]),
        expert_hidden_dims=[64],
    )
    runner.state["decisions"]["leader"] = [
        {**leader.clone(seed=seed).__dict__} for seed in (0, 1, 2)
    ]
    submitted = []

    def complete(specs):
        submitted.extend(spec.condition for spec in specs)
        return True

    def profile(spec):
        score = 0.704 if spec.gate_hidden_dims == [64] else 0.70
        return score, 0.60, 0.50, 1000, 2000

    monkeypatch.setattr(runner, "ensure_specs", complete)
    monkeypatch.setattr(runner, "_profile", profile)
    monkeypatch.setattr(runner, "_owned_expert_degradation", lambda _spec: 0.0)
    runner.run_phase_3()
    assert "leader_gate64x32" in submitted
    assert len(submitted) == 8
    assert runner.state["phases"]["3"]["status"] == "complete"


def test_validation_report_writes_overall_dataset_and_class_tables(tmp_path):
    class AlwaysZero(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def predict(self, features):
            return torch.zeros(len(features), dtype=torch.long, device=features.device)

    split = SimpleNamespace(
        features=np.zeros((6, 3), dtype=np.float32),
        class_idx=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
        dataset_idx=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    context = SimpleNamespace(data=SimpleNamespace(
        val=split, class_names=["benign", "attack"], active_datasets=["A", "B"]
    ))
    config = {
        "training": {"device": "cpu", "validation_chunk_rows": 2},
        "evaluation": {"output_dir": str(tmp_path)},
    }
    frames = write_validation_report_ooc(AlwaysZero(), context, config)
    assert np.isclose(frames["overall"].iloc[0]["accuracy"], 0.5)
    assert set(frames["per_dataset"]["dataset"]) == {"A", "B"}
    assert set(frames["per_class"]["class"]) == {"benign", "attack"}
    assert (tmp_path / "Validation_Overall_Metrics.csv").is_file()


def test_synthetic_asymmetric_private_abc_end_to_end(tmp_path):
    rng = np.random.default_rng(7)

    def make_split(rows):
        half = rows // 2
        split = PreparedSplit(
            rng.normal(size=(rows, 4)).astype(np.float32),
            np.tile(np.asarray([0, 1], dtype=np.int64), rows // 2),
            np.repeat(np.asarray([0, 1], dtype=np.int64), half),
            np.repeat(np.asarray(["A", "B"]), half),
        )
        split.dataset_slices = {"A": slice(0, half), "B": slice(half, rows)}
        return split

    data = PreparedData(
        None, ["benign", "attack"], ["A", "B"],
        make_split(24), make_split(12), make_split(12),
    )
    prepared = {
        name: SimpleNamespace(class_names=tuple(data.class_names))
        for name in data.active_datasets
    }
    context = OutOfCoreContext(
        data, prepared, {"A": "split-a", "B": "split-b"},
        "prepared", "unused", [f"f{index}" for index in range(4)],
    )
    base = load_config(ROOT_CONFIG)
    base["seed"] = 0
    base["architecture"] = "moe_dataset_private_encoders"
    base["data"]["active_datasets"] = ["A", "B"]
    base["model"] = {
        "latent_dim": 5,
        "encoder": {"kind": "mlp", "hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "gate_encoder": {"kind": "mlp", "hidden_dims": [7], "activation": "relu", "dropout": 0.0},
        "private_encoder": {
            "kind": "residual_mlp", "hidden_dims": [], "width": 8, "blocks": 2,
            "expansion": 2.0, "normalization": "layernorm", "activation": "relu",
            "dropout": 0.0, "dropout_second": 0.0,
        },
        "expert": {"hidden_dims": [], "dropout": 0.0},
        "adapter": {"rank": 2, "dropout": 0.0},
        "gate": {"hidden_dims": [], "routing": "dense"},
    }
    base["training"].update({
        "device": "cpu", "batch_size": 6, "epochs_a": 1, "epochs_b": 1,
        "epochs_c": 1, "lr": 0.001, "weight_decay": 0.0,
        "stage_c_unfreeze": "last_layer", "selection_mode": "fixed_epochs",
        "shuffle_block_rows": 12, "shuffle_buffer_blocks": 2,
        "progress_every_rows": 1000, "validation_chunk_rows": 6,
    })
    base["training"]["representation"] = {
        "objective": "ce", "sampling": "legacy", "class_weighting": "legacy",
        "weight": 0.1, "temperature": 0.1, "center_weight": 0.01,
        "arc_margin": 0.3, "arc_scale": 30.0,
    }
    base["training"]["stage_a"] = {"optimizer": {"lr": 0.001, "weight_decay": 0.0}}
    base["training"]["stage_b"] = {
        "warmstart_mode": "dataset",
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
    }
    base["training"]["stage_c"] = {
        "gate_supervision": "none", "expert_update_policy": "all",
        "lambda_expert_anchor": 0.0,
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
    }
    base["load_balance"]["lambda_balance"] = 0.1

    gate_config = copy.deepcopy(base)
    gate_config["model"]["encoder"] = copy.deepcopy(base["model"]["gate_encoder"])
    gate_config["model"].pop("gate_encoder")
    gate_config["model"].pop("private_encoder")
    gate_config["training"]["checkpoint_dir"] = str(tmp_path / "gate_a")
    run_stage_a_ooc(gate_config, context)

    private_config = copy.deepcopy(base)
    private_config["model"]["encoder"] = copy.deepcopy(base["model"]["private_encoder"])
    private_config["model"].pop("gate_encoder")
    private_config["model"].pop("private_encoder")
    private_config["training"]["checkpoint_dir"] = str(tmp_path / "private_a")
    run_stage_a_ooc(private_config, context)

    final = copy.deepcopy(base)
    final["training"]["checkpoint_dir"] = str(tmp_path / "final")
    final["training"]["stage_a_checkpoint"] = str(tmp_path / "gate_a")
    final["training"]["private_stage_a_checkpoint"] = str(tmp_path / "private_a")
    final["evaluation"]["output_dir"] = str(tmp_path / "results")
    run_stage_b_ooc(final, context)
    model = run_stage_c_ooc(final, context)

    assert model.encoder.kind == "mlp"
    assert all(encoder.kind == "residual_mlp" for encoder in model.expert_bank.encoders)
    assert (tmp_path / "results" / "Validation_Overall_Metrics.csv").is_file()
