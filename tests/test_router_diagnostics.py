from types import SimpleNamespace

import numpy as np
import torch

from evaluation.router_diagnostics import (
    VARIANTS, diagnostic_gaps, diagnostic_readout, run_router_diagnostics,
)
from models.baselines import HardTwoStageModel
from models.dataset_experts import DatasetExpertBank
from models.encoder import SharedEncoder
from models.gate import Gate
from models.moe import MoEDatasetNIDS


def test_router_diagnostic_runs_all_counterfactuals_and_writes_outputs(tmp_path):
    torch.manual_seed(0)
    dataset_names = ["alpha", "beta"]
    class_names = ["Benign", "Attack"]
    features = np.asarray(
        [[-2.0, -1.0], [-1.0, -2.0], [-0.5, -1.5], [2.0, 1.0], [1.0, 2.0], [0.5, 1.5]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
    dataset_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    split = SimpleNamespace(
        features=features,
        class_idx=labels,
        dataset_idx=dataset_ids,
        dataset_name=np.asarray(["alpha"] * 3 + ["beta"] * 3),
    )
    data = SimpleNamespace(
        active_datasets=dataset_names,
        class_names=class_names,
        test=split,
    )

    encoder = SharedEncoder(2, [4], 2, "relu", 0.0)
    moe = MoEDatasetNIDS(
        encoder,
        DatasetExpertBank(dataset_names, 2, 2, [], 0.0),
        Gate(2, 2, []),
        class_names,
    )
    hard = HardTwoStageModel(
        dataset_names, 2, 2, 2,
        id_hidden_dims=[4], stage_b_hidden_dims=[4], dropout=0.0,
    )

    result = run_router_diagnostics(
        moe, hard, data, output_dir=str(tmp_path), chunk_rows=2
    )

    assert set(result.overall["variant"]) == set(VARIANTS)
    assert len(result.per_dataset) == len(VARIANTS) * len(dataset_names)
    assert len(result.per_class) == len(VARIANTS) * len(class_names)
    assert len(result.per_dataset_per_class) == len(VARIANTS) * len(dataset_names) * len(class_names)
    assert int(result.moe_route_confusion.to_numpy().sum()) == len(features)
    assert int(result.hard_route_confusion.to_numpy().sum()) == len(features)
    assert len(result.expert_cross_dataset) == len(dataset_names) ** 2
    assert int(result.confidence_bins["rows"].sum()) == len(features)
    assert len(result.route_conditioned) == 6
    assert len(diagnostic_gaps(result.overall)) == 4
    readout = diagnostic_readout(result.overall, result.routing, result.confidence_bins)
    assert "Hard oracle minus MoE oracle" in readout
    assert (tmp_path / "Diagnostic_Overall.csv").is_file()
    assert (tmp_path / "Diagnostic_Per_Class.csv").is_file()
    assert (tmp_path / "Diagnostic_Per_Dataset_Per_Class.csv").is_file()
    assert (tmp_path / "MoE_Confidence_Bins.csv").is_file()
    assert (tmp_path / "MoE_Route_Confusion.csv").is_file()
