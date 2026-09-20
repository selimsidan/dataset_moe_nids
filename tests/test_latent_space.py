from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import evaluation.latent_space as latent
from models.encoder import SharedEncoder
from training.dataset import PreparedData, PreparedSplit


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

    snapshots = latent.load_private_encoder_snapshots(
        config, context, torch.device("cpu"), stages=["A"]
    )

    assert [snapshot.snapshot_id for snapshot in snapshots] == ["A__shared_initialization"]


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

    monkeypatch.setattr(latent, "load_private_encoder_snapshots", snapshots)
    stage_a_dir = tmp_path / "results" / "latent" / "stage_A"
    stage_b_dir = tmp_path / "results" / "latent" / "stage_B"
    report_a = latent.evaluate_private_latent_checkpoints(
        config, context, stages=["A"], output_dir=str(stage_a_dir), device="cpu"
    )
    report_b = latent.evaluate_private_latent_checkpoints(
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
    cached_b = latent.evaluate_private_latent_checkpoints(
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
