import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data import paths
from data.out_of_core import class_split_quotas, prepare_dataset
from data.registry import DatasetSpec
from training.out_of_core_train import shuffled_row_batches
from training.out_of_core_data import OutOfCoreContext, prepare_out_of_core_data
from training.dataset import PreparedData, PreparedSplit
from training.out_of_core_train import ensure_run_contract, run_stage_a_ooc, run_stage_b_ooc, run_stage_c_ooc
from evaluation.out_of_core_report import evaluate_and_report_ooc


def _fixture(tmp_path):
    rows = []
    for label, count in [("Benign", 31), ("raw_dos", 17), ("tiny", 2)]:
        for index in range(count):
            rows.append({"f1": float(index), "f2": np.inf if index == 0 else index * 2.0, "Attack": label})
    pd.DataFrame(rows).to_csv(tmp_path / "fixture.csv", index=False)
    paths.PATH_OVERRIDES["Fixture"] = str(tmp_path / "fixture.csv")
    spec = DatasetSpec(
        name="Fixture", kind="file", default_paths=[], label_col="Attack",
        benign_label="Benign", feature_alias={"one": "f1", "two": "f2"},
    )
    return spec, {"raw_dos": "DoS", "tiny": "Tiny"}


def test_exact_class_quotas_and_train_only_rare_class():
    cfg = {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "min_class_count": 3}
    quotas = class_split_quotas({"A": 11, "rare": 2}, cfg)
    assert sum(quotas["A"].values()) == 11
    assert quotas["rare"] == {"train": 2, "val": 0, "test": 0}


def test_signed_preparation_is_exhaustive_reusable_and_globally_shuffled(tmp_path):
    spec, mapping = _fixture(tmp_path)
    cfg = {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "min_class_count": 3}
    prepared = prepare_dataset(spec, mapping, ["f1", "f2"], cfg, 7, str(tmp_path / "cache"), chunk_rows=9)
    reused = prepare_dataset(spec, mapping, ["f1", "f2"], cfg, 7, str(tmp_path / "cache"), chunk_rows=5)
    assert prepared.signature == reused.signature
    assert sum(prepared.split_rows.values()) == 50
    assert prepared.class_split_counts["Tiny"] == {"train": 2, "val": 0, "test": 0}
    with open(os.path.join(prepared.directory, "metadata.json")) as handle:
        assert json.load(handle)["contract"]["split_seed"] == 7
    for split in ("train", "val", "test"):
        assert len(prepared.features(split)) == len(prepared.labels(split))


def test_bounded_shuffle_uses_every_row_once_and_mixes_blocks():
    batches = list(shuffled_row_batches(0, 800, np.random.default_rng(0), 32, 100, 8))
    rows = np.concatenate(batches)
    assert np.array_equal(np.sort(rows), np.arange(800))
    assert len(np.unique(batches[0] // 100)) > 1


def test_synthetic_ooc_moe_runs_all_stages_and_reports(tmp_path):
    rng = np.random.default_rng(4)

    def split(n):
        features = rng.normal(size=(n, 6)).astype(np.float32)
        labels = np.tile(np.arange(4), n // 4).astype(np.int16)
        dataset_ids = np.repeat(np.arange(2), n // 2).astype(np.int16)
        return PreparedSplit(features, labels, dataset_ids, None)

    train, val, test = split(64), split(32), split(32)
    for value in (train, val, test):
        value.dataset_slices = {"A": slice(0, len(value.class_idx) // 2), "B": slice(len(value.class_idx) // 2, len(value.class_idx))}
    data = PreparedData(None, ["Benign", "c1", "c2", "c3"], ["A", "B"], train, val, test)
    prepared = {
        "A": SimpleNamespace(class_names=tuple(data.class_names)),
        "B": SimpleNamespace(class_names=tuple(data.class_names)),
    }
    context = OutOfCoreContext(data, prepared, {"A": "a", "B": "b"}, "prep", "unused", [f"f{i}" for i in range(6)])
    config = {
        "run_name": "synthetic", "seed": 0, "architecture": "moe_dataset_soft",
        "model": {
            "latent_dim": 8,
            "encoder": {"hidden_dims": [12], "activation": "relu", "dropout": 0.0},
            "expert": {"hidden_dims": [8], "dropout": 0.0},
            "adapter": {"rank": 4, "dropout": 0.0}, "gate": {"hidden_dims": []},
        },
        "load_balance": {"lambda_balance": 0.1},
        "training": {
            "device": "cpu", "batch_size": 16, "epochs_a": 1, "epochs_b": 1, "epochs_c": 1,
            "lr": 0.001, "weight_decay": 0.0, "stage_c_unfreeze": "all",
            "stage_c": {"gate_supervision": "none", "lambda_dataset_aux": 0.1, "lambda_dataset_aux_hard": 5.0},
            "checkpoint_dir": str(tmp_path / "checkpoints"), "stages": ["A", "B", "C"],
            "shuffle_block_rows": 16, "shuffle_buffer_blocks": 2,
        },
        "evaluation": {"output_dir": str(tmp_path / "results"), "prediction_chunk_rows": 8},
    }
    contract = ensure_run_contract(config, context)
    run_stage_a_ooc(config, context)
    run_stage_b_ooc(config, context)
    model = run_stage_c_ooc(config, context)
    reports = evaluate_and_report_ooc(model, context, config, contract)
    assert set(reports["overall"]["origin"]) == {"A", "B", "ALL"}
    assert (tmp_path / "results" / "Per_Class_Metrics.csv").is_file()


def test_two_way_configuration_builds_logical_pooled_views(tmp_path, monkeypatch):
    aliases = {f"canonical_{i}": f"F{i}" for i in range(47)}
    specs = {}
    for dataset_index, name in enumerate(["A", "B"]):
        frame = pd.DataFrame({raw: np.arange(40, dtype=np.float32) + i for i, raw in enumerate(aliases.values())})
        frame["Attack"] = np.where(np.arange(40) % 3, "Benign", "raw_dos")
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.PATH_OVERRIDES[name] = str(path)
        specs[name] = DatasetSpec(name, "file", [], "Attack", "Benign", aliases)

    import training.out_of_core_data as module
    monkeypatch.setattr(module, "get_spec", lambda name: specs[name])
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(paths, "SCRATCH_DIR", str(tmp_path / "scratch"))
    config = {
        "seed": 0,
        "data": {
            "active_datasets": ["A", "B"], "active_classes": ["DoS"],
            "label_mapping": {"A": {"raw_dos": "DoS"}, "B": {"raw_dos": "DoS"}},
            "split": {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "min_class_count": 3},
            "chunksize": 13, "out_of_core_chunk_rows": 13, "stage_out_of_core_to_local": False,
        },
    }
    context = prepare_out_of_core_data(config)
    assert context.data.active_datasets == ["A", "B"]
    assert context.data.train.features.shape[1] == 47
    assert len(context.data.train.class_idx) == sum(
        value.split_rows["train"] for value in context.prepared_by_dataset.values()
    )
    assert context.data.train.dataset_slices["A"].stop == context.data.train.dataset_slices["B"].start
