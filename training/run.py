"""CLI entry point. Architecture and stage selection are config-driven, not
separate scripts/forks:

    python -m training.run --config config/default.yaml \
        --set architecture=moe_dataset_soft --set training.stages=[A,B,C]

    python -m training.run --config config/default.yaml --set architecture=moe_dataset_hard_gate
    python -m training.run --config config/default.yaml --set architecture=moe_dataset_adapters
    python -m training.run --config config/default.yaml --set architecture=plain_pooled
    python -m training.run --config config/default.yaml --set architecture=no_fusion
    python -m training.run --config config/default.yaml --set architecture=hard_two_stage

    # Ablation: re-run only Stage C from an existing Stage A+B checkpoint
    python -m training.run --config config/default.yaml --set training.stages=[C]
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

import numpy as np
import torch

from data.paths import local_mirror_dir, sync_from_drive, sync_to_drive
from evaluation.bootstrap_ci import bootstrap_per_class_ci
from evaluation.metrics import evaluate_per_dataset, evaluate_predictions
from evaluation.report import write_tracker_csvs

from .baseline_train import train_hard_two_stage, train_no_fusion, train_plain_pooled
from .checkpoint import clear_progress, save_harmonizer, stage_complete
from .config import load_config
from .dataset import prepare_datasets
from .logging_utils import tee_stdout_to_file
from .stage_a_pretrain import run_stage_a
from .stage_b_warmstart import run_stage_b
from .stage_c_jointfinetune import build_model_from_checkpoints, run_stage_c

MOE_ARCHITECTURES = ("moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_adapters")
ALL_ARCHITECTURES = (*MOE_ARCHITECTURES, "plain_pooled", "no_fusion", "hard_two_stage")

# moe_dataset_hard_gate is an explicit ablation of the "don't supervise the
# gate on dataset ID" design decision -- forcing gate_supervision=hard here
# means selecting this architecture is enough on its own (no extra --set
# needed) to reproduce the ablation described in the project spec.
ARCHITECTURE_STAGE_C_DEFAULTS = {
    "moe_dataset_hard_gate": "hard",
}

RUN_MODE_OVERRIDES = {
    "full": [],
    "smoke": [
        "run_name=dataset_moe_nids_smoke",
        "data.max_rows_per_dataset=5000",
        "data.chunksize=20000",
        "training.epochs_a=1",
        "training.epochs_b=1",
        "training.epochs_c=1",
        "training.batch_size=64",
        "training.min_per_class_per_batch=3",
        "training.force_restart=true",
    ],
}


def _predict_moe(config: dict, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(config["training"].get("device", "cpu"))
    model = build_model_from_checkpoints(config, data, device)
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data.test.features).to(device)
        preds = model.predict(features).cpu().numpy()
    return data.test.class_idx, preds, data.test.dataset_name


def _predict_plain_pooled(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data.test.features).to(device)
        preds = model.predict(features).cpu().numpy()
    return data.test.class_idx, preds, data.test.dataset_name


def _predict_no_fusion(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    y_true_all, y_pred_all, ds_all = [], [], []
    with torch.no_grad():
        for name in model.dataset_names:
            mask = data.test.dataset_name == name
            if mask.sum() == 0:
                continue
            features = torch.from_numpy(data.test.features[mask]).to(device)
            preds = model.predict(features, name).cpu().numpy()
            y_true_all.append(data.test.class_idx[mask])
            y_pred_all.append(preds)
            ds_all.append(data.test.dataset_name[mask])
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(ds_all)


def _predict_hard_two_stage(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data.test.features).to(device)
        preds = model.predict(features).cpu().numpy()
    return data.test.class_idx, preds, data.test.dataset_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--mode",
        choices=sorted(RUN_MODE_OVERRIDES),
        default="full",
        help="full uses the config as-is; smoke applies small-data/short-epoch overrides for a pipeline check.",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, [*RUN_MODE_OVERRIDES[args.mode], *args.overrides])
    architecture = config["architecture"]
    if architecture not in ALL_ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected one of {ALL_ARCHITECTURES}.")
    if architecture in ARCHITECTURE_STAGE_C_DEFAULTS and "gate_supervision" not in " ".join(args.overrides):
        config["training"]["stage_c"]["gate_supervision"] = ARCHITECTURE_STAGE_C_DEFAULTS[architecture]

    requested_device = config["training"].get("device", "cpu")
    trial_id = f"{config['run_name']}-{architecture}-{uuid.uuid4().hex[:8]}"

    drive_checkpoint_dir = config["training"]["checkpoint_dir"]
    local_checkpoint_dir = local_mirror_dir(drive_checkpoint_dir)
    sync_from_drive(drive_checkpoint_dir, local_checkpoint_dir)
    config["training"]["checkpoint_dir"] = local_checkpoint_dir

    force_restart = bool(config["training"].get("force_restart", False))
    if force_restart:
        for stage in ("A", "B", "C"):
            clear_progress(local_checkpoint_dir, stage)

    tee_stdout_to_file(os.path.join(local_checkpoint_dir, "train.log"))

    print(f"[run] mode={args.mode} architecture={architecture} trial_id={trial_id}")
    print(f"[run] python={sys.executable}")
    print(
        f"[run] requested_device={requested_device} "
        f"torch.cuda.is_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}"
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device=cuda was requested, but PyTorch cannot see a CUDA GPU. "
            "In Colab, switch Runtime -> Change runtime type -> GPU, then restart and rerun the notebook."
        )
    if requested_device == "cpu" and torch.cuda.is_available():
        print("[run] WARNING: CUDA is available, but training.device=cpu was requested.")
    print(f"[run] checkpoint_dir(local scratch)={local_checkpoint_dir} -> synced to {drive_checkpoint_dir}")

    data = prepare_datasets(config)
    save_harmonizer(local_checkpoint_dir, data.harmonizer)
    sync_to_drive(local_checkpoint_dir, drive_checkpoint_dir)
    print(f"[run] class_names ({len(data.class_names)}): {data.class_names}")
    print(f"[run] active_datasets ({len(data.active_datasets)}): {data.active_datasets}")

    if architecture in MOE_ARCHITECTURES:
        stages = config["training"]["stages"]
        for stage, run_fn in (("A", run_stage_a), ("B", run_stage_b), ("C", run_stage_c)):
            if stage not in stages:
                continue
            if not force_restart and stage_complete(drive_checkpoint_dir, stage):
                print(f"[run] Stage {stage} checkpoint already exists in {drive_checkpoint_dir} -- skipping "
                      f"(pass --set training.force_restart=true to redo it)")
                continue
            run_fn(config, data)
            sync_to_drive(local_checkpoint_dir, drive_checkpoint_dir)
        y_true, y_pred, ds_names = _predict_moe(config, data)

    elif architecture == "plain_pooled":
        model = train_plain_pooled(config, data)
        y_true, y_pred, ds_names = _predict_plain_pooled(model, data)

    elif architecture == "no_fusion":
        model = train_no_fusion(config, data)
        y_true, y_pred, ds_names = _predict_no_fusion(model, data)

    elif architecture == "hard_two_stage":
        model = train_hard_two_stage(config, data)
        y_true, y_pred, ds_names = _predict_hard_two_stage(model, data)

    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    result = evaluate_predictions(y_true, y_pred, data.class_names)
    per_dataset = evaluate_per_dataset(y_true, y_pred, ds_names, data.class_names)
    print(f"[run] macro_f1={result.macro_f1:.4f} weighted_f1(reference only)={result.weighted_f1:.4f}")
    for cm in result.per_class:
        print(f"  {cm.class_name:20s} support={cm.support:6d} precision={cm.precision:.3f} recall={cm.recall:.3f} f1={cm.f1:.3f}")
    print("[run] per-dataset macro_f1:")
    for name, ds_result in per_dataset.items():
        print(f"  {name:20s} macro_f1={ds_result.macro_f1:.4f}")

    eval_cfg = config["evaluation"]
    ci = bootstrap_per_class_ci(
        y_true,
        y_pred,
        data.class_names,
        low_sample_threshold=eval_cfg["low_sample_threshold"],
        n_bootstrap=eval_cfg["bootstrap_n"],
        seed=eval_cfg["bootstrap_seed"],
    )
    write_tracker_csvs(
        eval_cfg["output_dir"], trial_id, config,
        {architecture: result}, {architecture: ci}, {architecture: per_dataset},
    )
    print(f"[run] wrote tracker CSVs to {eval_cfg['output_dir']}")


if __name__ == "__main__":
    main()
