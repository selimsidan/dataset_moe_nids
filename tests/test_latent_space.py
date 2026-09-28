from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import evaluation.latent_space as latent
from evaluation.latent_space import evaluate_latent_checkpoints, load_latent_snapshots
from models.encoder import SharedEncoder
from training.dataset import PreparedData, PreparedSplit
from training.out_of_core_data import OutOfCoreContext
from training.out_of_core_train import run_stage_a_ooc, run_stage_b_ooc, run_stage_c_ooc


def _fixture(tmp_path):
    features = np.asarray([
        [-2.0, -1.0], [-1.8, -1.2], [2.0, 1.0], [1.8, 1.2],
        [-2.1, -0.9], [-1.9, -1.1], [2.1, 0.9], [1.9, 1.1],
    ], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16)
    datasets = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
    train = PreparedSplit(np.tile(features, (2, 1)), np.tile(labels, 2), np.tile(datasets, 2), None)
    val = PreparedSplit(features, labels, datasets, None)
    test = PreparedSplit(features, labels, datasets, None)
    data = PreparedData(None, ["Benign", "Attack"], ["A", "B"], train, val, test)
    context = SimpleNamespace(data=data)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for filename in ("stage_a_encoder.pt", "stage_b_expert_bank.pt", "stage_c_full.pt"):
        torch.save({}, checkpoint_dir / filename)
    config = {
        "seed": 3,
        "architecture": "moe_dataset_private_encoders",
        "model": {
            "latent_dim": 2,
            "encoder": {"hidden_dims": [], "activation": "relu", "dropout": 0.0},
            "expert": {"hidden_dims": [], "dropout": 0.0},
            "adapter": {"rank": 2, "dropout": 0.0},
            "gate": {"hidden_dims": [], "routing": "dense"},
        },
        "training": {"checkpoint_dir": str(checkpoint_dir), "device": "cpu"},
        "evaluation": {
            "output_dir": str(tmp_path / "results"),
            "latent": {
                "max_train_rows": 16,
                "max_eval_rows": 8,
                "max_silhouette_rows": 8,
                "max_per_class_dataset": 8,
                "batch_size": 4,
                "knn_neighbors": 3,
                "random_seed": 5,
            },
        },
    }
    return config, context


def _identity_encoder():
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2))
    return module


def test_stage_a_snapshot_load_does_not_require_later_checkpoints(tmp_path):
    config, context = _fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "stage_b_expert_bank.pt").unlink()
    (checkpoint_dir / "stage_c_full.pt").unlink()
    encoder = SharedEncoder(2, [], 2, "relu", 0.0)
    torch.save({"encoder_state": encoder.state_dict()}, checkpoint_dir / "stage_a_encoder.pt")

    snapshots = load_latent_snapshots(
        config, context, torch.device("cpu"), stages=["A"]
    )

    assert [snapshot.snapshot_id for snapshot in snapshots] == ["A__shared_initialization"]


def test_private_stage_a_initialization_is_opt_in(tmp_path):
    config, context = _fixture(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    gate = SharedEncoder(2, [], 2, "relu", 0.0)
    private = SharedEncoder(2, [3], 2, "relu", 0.0)
    torch.save({"encoder_state": gate.state_dict()}, checkpoint_dir / "stage_a_encoder.pt")
    private_dir = tmp_path / "private_a"
    private_dir.mkdir()
    torch.save({"encoder_state": private.state_dict()}, private_dir / "stage_a_encoder.pt")
    config["model"]["gate_encoder"] = {
        "hidden_dims": [], "activation": "relu", "dropout": 0.0,
    }
    config["model"]["private_encoder"] = {
        "hidden_dims": [3], "activation": "relu", "dropout": 0.0,
    }
    config["training"]["private_stage_a_checkpoint"] = str(private_dir)
    config["evaluation"]["latent"]["include_private_initialization"] = True

    snapshots = load_latent_snapshots(
        config, context, torch.device("cpu"), stages=["A"]
    )

    assert [snapshot.snapshot_id for snapshot in snapshots] == [
        "A__shared_initialization", "A__private_initialization"
    ]


def test_private_latent_public_names_remain_backwards_compatible():
    assert latent.load_private_encoder_snapshots is latent.load_latent_snapshots
    assert latent.evaluate_private_latent_checkpoints is latent.evaluate_latent_checkpoints


def test_incremental_reports_reuse_manifest_and_stage_a_gate_alias(tmp_path, monkeypatch):
    config, context = _fixture(tmp_path)
    identity = _identity_encoder()
    expert = _identity_encoder()
    with torch.no_grad():
        expert.weight.copy_(torch.tensor([[1.0, 0.2], [0.0, 1.0]]))

    def snapshots(_config, _context, _device, *, stages=None):
        if tuple(stages) == ("A",):
            return [latent.LatentSnapshot("A", "shared_initialization", identity, "shared")]
        if tuple(stages) == ("B",):
            return [
                latent.LatentSnapshot(
                    "B", "gate", None, "gate", equivalent_to="A__shared_initialization"
                ),
                latent.LatentSnapshot("B", "expert::A", expert, "private_expert"),
            ]
        raise AssertionError(stages)

    monkeypatch.setattr(latent, "load_latent_snapshots", snapshots)
    stage_a_dir = tmp_path / "results" / "latent" / "stage_A"
    stage_b_dir = tmp_path / "results" / "latent" / "stage_B"
    report_a = latent.evaluate_latent_checkpoints(
        config, context, stages=["A"], output_dir=str(stage_a_dir), device="cpu"
    )
    report_b = latent.evaluate_latent_checkpoints(
        config,
        context,
        stages=["B"],
        output_dir=str(stage_b_dir),
        device="cpu",
        sample_manifest=stage_a_dir,
        reference_report=report_a,
    )

    pd.testing.assert_frame_equal(report_a["manifest"], report_b["manifest"])
    pd.testing.assert_frame_equal(report_a["train_manifest"], report_b["train_manifest"])
    np.testing.assert_array_equal(
        report_a["embeddings"]["A__shared_initialization"],
        report_b["embeddings"]["B__gate"],
    )
    gate = report_b["snapshots"].query("snapshot == 'B__gate'").iloc[0]
    source = report_a["snapshots"].query("snapshot == 'A__shared_initialization'").iloc[0]
    assert gate["equivalent_to"] == "A__shared_initialization"
    assert gate["silhouette"] == source["silhouette"]
    assert not list(stage_a_dir.glob("*.tmp*"))
    assert not list(stage_b_dir.glob("*.tmp*"))

    cumulative_dir = tmp_path / "results" / "latent" / "cumulative"
    combined = latent.combine_latent_reports([report_a, report_b], cumulative_dir)
    assert set(combined["snapshots"]["stage"]) == {"A", "B"}
    loaded = latent.load_latent_report(cumulative_dir)
    assert set(loaded["embeddings"]) == set(combined["embeddings"])

    monkeypatch.setattr(latent, "_extract", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("compatible completed reports must not recompute embeddings")
    ))
    cached_b = latent.evaluate_latent_checkpoints(
        config,
        context,
        stages=["B"],
        output_dir=str(stage_b_dir),
        device="cpu",
        sample_manifest=stage_a_dir,
        reference_report=report_a,
    )
    assert set(cached_b["embeddings"]) == {"B__gate", "B__expert::A"}


def test_focus_plot_can_be_saved_for_every_supported_class(tmp_path):
    pytest.importorskip("matplotlib")
    manifest = pd.DataFrame({"class": ["Benign", "Benign", "Attack", "Attack"]})
    report = {"manifest": manifest}
    projections = {"A__shared_initialization": np.asarray([
        [-1.0, -1.0], [-0.8, -1.2], [1.0, 1.0], [0.8, 1.2]
    ])}

    paths = [
        latent.plot_class_focus(report, projections, class_name, str(tmp_path))
        for class_name in sorted(manifest["class"].unique())
    ]

    assert all((tmp_path / "latent_figures" / ("class_focus__" + name + ".png")).is_file()
               for name in ("Attack", "Benign"))
    assert all(not path.endswith(".tmp") for path in paths)


def _synthetic_context() -> OutOfCoreContext:
    rng = np.random.default_rng(7)

    def split(n):
        features = rng.normal(size=(n, 6)).astype(np.float32)
        labels = np.tile(np.arange(4), n // 4).astype(np.int16)
        dataset_ids = np.repeat(np.arange(2), n // 2).astype(np.int16)
        return PreparedSplit(features, labels, dataset_ids, None)

    train, val, test = split(64), split(32), split(32)
    for value in (train, val, test):
        value.dataset_slices = {
            "A": slice(0, len(value.class_idx) // 2),
            "B": slice(len(value.class_idx) // 2, len(value.class_idx)),
        }
    data = PreparedData(None, ["Benign", "c1", "c2", "c3"], ["A", "B"], train, val, test)
    prepared = {
        "A": SimpleNamespace(class_names=tuple(data.class_names)),
        "B": SimpleNamespace(class_names=tuple(data.class_names)),
    }
    return OutOfCoreContext(data, prepared, {"A": "a", "B": "b"}, "prep", "unused", [f"f{i}" for i in range(6)])


def _config(tmp_path, architecture: str) -> dict:
    return {
        "run_name": "synthetic", "seed": 0, "architecture": architecture,
        "model": {
            "latent_dim": 8,
            "encoder": {"hidden_dims": [12], "activation": "relu", "dropout": 0.0},
            "expert": {"hidden_dims": [8], "dropout": 0.0},
            "adapter": {"rank": 4, "dropout": 0.0},
            "gate": {"hidden_dims": [], "routing": "dense"},
        },
        "load_balance": {"lambda_balance": 0.1},
        "training": {
            "device": "cpu", "batch_size": 16, "epochs_a": 1, "epochs_b": 1, "epochs_c": 1,
            "lr": 0.001, "weight_decay": 0.0, "stage_c_unfreeze": "all",
            "stage_c": {
                "gate_supervision": "none", "expert_update_policy": "all",
                "lambda_expert_anchor": 0.0,
            },
            "checkpoint_dir": str(tmp_path / "checkpoints"), "stages": ["A", "B", "C"],
            "shuffle_block_rows": 16, "shuffle_buffer_blocks": 2,
        },
        "evaluation": {
            "output_dir": str(tmp_path / "results"), "prediction_chunk_rows": 8,
            "latent": {
                "max_train_rows": 40, "max_eval_rows": 20, "max_silhouette_rows": 20,
                "max_per_class_dataset": 10, "batch_size": 16, "knn_neighbors": 2,
                "random_seed": 0,
            },
        },
    }


@pytest.mark.parametrize(
    "architecture,per_dataset_snapshots",
    [
        ("moe_dataset_soft", 0),
        ("moe_dataset_adapters", 2),
        ("moe_dataset_private_encoders", 2),
    ],
)
def test_load_latent_snapshots_per_architecture(tmp_path, architecture, per_dataset_snapshots):
    # moe_dataset_soft experts classify directly on the shared latent -- there
    # is no per-dataset representation transform, so it contributes only the
    # shared gate snapshot at each stage. Adapters and private encoders each
    # have a genuine per-dataset transform (FiLM adapter / private encoder),
    # so both datasets ("A", "B") contribute one snapshot per stage.
    context = _synthetic_context()
    config = _config(tmp_path, architecture)
    run_stage_a_ooc(config, context)
    run_stage_b_ooc(config, context)
    run_stage_c_ooc(config, context)

    snapshots = load_latent_snapshots(config, context, torch.device("cpu"), stages=("A", "B", "C"))
    snapshot_ids = {snapshot.snapshot_id for snapshot in snapshots}
    assert "A__shared_initialization" in snapshot_ids
    assert "B__gate" in snapshot_ids
    assert "C__gate" in snapshot_ids
    for stage in ("B", "C"):
        per_dataset = [
            snapshot for snapshot in snapshots
            if snapshot.stage == stage and snapshot.encoder.startswith("expert::")
        ]
        assert len(per_dataset) == per_dataset_snapshots
        assert all(snapshot.module is not None for snapshot in per_dataset)


def test_load_latent_snapshots_rejects_unknown_architecture(tmp_path):
    context = _synthetic_context()
    config = _config(tmp_path, "moe_basic")
    with pytest.raises(ValueError, match="does not recognize architecture"):
        load_latent_snapshots(config, context, torch.device("cpu"), stages=("A",))


@pytest.mark.parametrize("architecture", ["moe_dataset_soft", "moe_dataset_adapters", "moe_dataset_private_encoders"])
def test_evaluate_latent_checkpoints_writes_report(tmp_path, architecture):
    context = _synthetic_context()
    config = _config(tmp_path, architecture)
    run_stage_a_ooc(config, context)
    run_stage_b_ooc(config, context)
    run_stage_c_ooc(config, context)

    output_dir = tmp_path / "results" / "latent"
    report = evaluate_latent_checkpoints(
        config, context, split_name="val", output_dir=str(output_dir), stages=("C",),
    )
    assert (output_dir / "Latent_Snapshot_Metrics.csv").is_file()
    assert (output_dir / "Latent_Probe_Metrics.csv").is_file()
    assert (output_dir / "Latent_Report_Config.json").is_file()
    assert "C__gate" in set(report["snapshots"]["snapshot"])
    assert {"knn", "linear"}.issubset(set(report["probes"]["probe"]))
