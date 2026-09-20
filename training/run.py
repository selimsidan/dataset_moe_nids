"""CLI entry point. Architecture and stage selection are config-driven, not
separate scripts/forks:

    python -m training.run --config config/default.yaml \
        --set architecture=moe_dataset_soft --set training.stages=[A,B,C]

    python -m training.run --config config/default.yaml --set architecture=moe_dataset_hard_gate
    python -m training.run --config config/default.yaml --set architecture=moe_dataset_damex
    python -m training.run --config config/default.yaml --set architecture=moe_dataset_adapters
    python -m training.run --config config/default.yaml --set architecture=moe_dataset_private_encoders
    python -m training.run --config config/default.yaml --set architecture=moe_basic
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
import torch.nn.functional as F

from data.paths import local_mirror_dir, sync_from_drive, sync_to_drive
from evaluation.bootstrap_ci import bootstrap_per_class_ci
from evaluation.metrics import evaluate_per_dataset, evaluate_predictions
from evaluation.report import write_tracker_csvs
from evaluation.resource_accounting import resource_profile, stage_a_checkpoint_hash

from .baseline_train import train_hard_two_stage, train_matched_dense, train_no_fusion, train_plain_pooled
from .checkpoint import (
    BASELINE_MODEL_FILE, STAGE_A_FILE, _atomic_torch_save, clear_progress,
    load_stage_b, load_stage_c,
    resolve_stage_a_path, save_harmonizer, split_signature_for_data, stage_complete,
)
from .config import load_config
from .dataset import prepare_datasets
from .logging_utils import tee_stdout_to_file
from .stage_a_pretrain import run_stage_a
from .stage_b_warmstart import run_stage_b
from .stage_c_jointfinetune import build_model_from_checkpoints, run_stage_c

MOE_ARCHITECTURES = (
    "moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_damex",
    "moe_dataset_adapters", "moe_dataset_private_encoders", "moe_basic",
)
ALL_ARCHITECTURES = (*MOE_ARCHITECTURES, "matched_dense", "plain_pooled", "no_fusion", "hard_two_stage")

# Architecture-specific Stage-C presets are applied in training.config while
# preserving any explicit --set overrides.
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


def _predict_moe(config: dict, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(config["training"].get("device", "cpu"))
    model = build_model_from_checkpoints(config, data, device)
    model.load_state_dict(load_stage_c(config["training"]["checkpoint_dir"])["model_state"])
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data.test.features).to(device)
        scores = model(features)["combined_probs"].cpu().numpy()
        preds = scores.argmax(axis=1)
    return data.test.class_idx, preds, data.test.dataset_name, scores


def _predict_plain_pooled(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data.test.features).to(device)
        scores = F.softmax(model(features)["logits"], dim=1).cpu().numpy()
        preds = scores.argmax(axis=1)
    return data.test.class_idx, preds, data.test.dataset_name, scores


_predict_dense_baseline = _predict_plain_pooled


def _predict_no_fusion(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    y_true_all, y_pred_all, ds_all, score_all = [], [], [], []
    with torch.no_grad():
        for name in model.dataset_names:
            mask = data.test.dataset_name == name
            if mask.sum() == 0:
                continue
            features = torch.from_numpy(data.test.features[mask]).to(device)
            scores = F.softmax(model(features, name)["logits"], dim=1).cpu().numpy()
            preds = scores.argmax(axis=1)
            y_true_all.append(data.test.class_idx[mask])
            y_pred_all.append(preds)
            ds_all.append(data.test.dataset_name[mask])
            score_all.append(scores)
    return (
        np.concatenate(y_true_all), np.concatenate(y_pred_all),
        np.concatenate(ds_all), np.concatenate(score_all),
    )


def _predict_hard_two_stage(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    chunk_rows = 262_144
    class_predictions, dataset_predictions, class_scores = [], [], []
    with torch.no_grad():
        for start in range(0, len(data.test.features), chunk_rows):
            features = torch.from_numpy(data.test.features[start : start + chunk_rows]).to(device)
            out = model(features)
            scores = F.softmax(out["logits"], dim=1).cpu().numpy()
            class_scores.append(scores)
            class_predictions.append(scores.argmax(axis=1))
            dataset_predictions.append(out["dataset_pred"].cpu().numpy())
    preds = np.concatenate(class_predictions)
    route_preds = np.concatenate(dataset_predictions)
    route_accuracy = float((route_preds == data.test.dataset_idx).mean())
    print(f"[hard_two_stage] stage-a test dataset-ID accuracy={route_accuracy:.4f}")
    return data.test.class_idx, preds, data.test.dataset_name, np.concatenate(class_scores)


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

    baseline_cfg = config["training"].get("baseline", {})
    if architecture not in MOE_ARCHITECTURES and baseline_cfg.get("encoder_init", "stage_a") == "stage_a":
        source = config["training"].get("stage_a_checkpoint") or baseline_cfg.get("stage_a_checkpoint")
        local_stage_a = os.path.join(local_checkpoint_dir, STAGE_A_FILE)
        if not source and not os.path.isfile(local_stage_a):
            if "A" not in config["training"].get("stages", ["A", "B", "C"]):
                raise FileNotFoundError("Stage-A initialization requested but Stage A is absent and not scheduled")
            run_stage_a(config, data)
            sync_to_drive(local_checkpoint_dir, drive_checkpoint_dir)

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
        stage_a_ready = os.path.isfile(resolve_stage_a_path(config))
        if not stage_a_ready or not all(stage_complete(local_checkpoint_dir, stage) for stage in ("B", "C")):
            print("[run] Requested stages completed; full evaluation awaits Stage A+B+C checkpoints")
            return
        if not config["training"].get("run_final_evaluation", True):
            print("[run] All checkpoints are ready; final evaluation deferred by configuration")
            return
        model = build_model_from_checkpoints(config, data, torch.device(config["training"].get("device", "cpu")))
        model.load_state_dict(load_stage_c(config["training"]["checkpoint_dir"])["model_state"])
        y_true, y_pred, ds_names, y_score = _predict_moe(config, data)

    elif architecture == "plain_pooled":
        model = train_plain_pooled(config, data)
        y_true, y_pred, ds_names, y_score = _predict_plain_pooled(model, data)

    elif architecture == "matched_dense":
        model = train_matched_dense(config, data)
        y_true, y_pred, ds_names, y_score = _predict_dense_baseline(model, data)

    elif architecture == "no_fusion":
        model = train_no_fusion(config, data)
        y_true, y_pred, ds_names, y_score = _predict_no_fusion(model, data)

    elif architecture == "hard_two_stage":
        model = train_hard_two_stage(config, data)
        y_true, y_pred, ds_names, y_score = _predict_hard_two_stage(model, data)

    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    if architecture not in MOE_ARCHITECTURES:
        _atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "class_names": data.class_names,
                "dataset_names": data.active_datasets,
                "match_info": getattr(model, "match_info", None),
                "training_summary": getattr(model, "training_summary", {}),
            },
            os.path.join(local_checkpoint_dir, BASELINE_MODEL_FILE),
        )
        sync_to_drive(local_checkpoint_dir, drive_checkpoint_dir)

    result = evaluate_predictions(y_true, y_pred, data.class_names, y_score)
    per_dataset = evaluate_per_dataset(y_true, y_pred, ds_names, data.class_names, y_score)
    print(
        f"[run] accuracy={result.accuracy:.4f} balanced_accuracy={result.balanced_accuracy:.4f} "
        f"macro_p/r/f1={result.macro_precision:.4f}/{result.macro_recall:.4f}/{result.macro_f1:.4f} "
        f"micro_p/r/f1={result.micro_precision:.4f}/{result.micro_recall:.4f}/{result.micro_f1:.4f} "
        f"weighted_p/r/f1={result.weighted_precision:.4f}/{result.weighted_recall:.4f}/{result.weighted_f1:.4f} "
        f"roc_auc_ovr_macro={result.roc_auc_ovr_macro:.4f} "
        f"roc_auc_ovr_weighted={result.roc_auc_ovr_weighted:.4f} "
        f"roc_auc_ovr_micro={result.roc_auc_ovr_micro:.4f}"
    )
    for cm in result.per_class:
        print(f"  {cm.class_name:20s} support={cm.support:6d} precision={cm.precision:.3f} recall={cm.recall:.3f} f1={cm.f1:.3f} roc_auc_ovr={cm.roc_auc_ovr:.3f}")
    print("[run] per-dataset metrics:")
    for name, ds_result in per_dataset.items():
        print(
            f"  {name:20s} accuracy={ds_result.accuracy:.4f} "
            f"macro_p/r/f1={ds_result.macro_precision:.4f}/{ds_result.macro_recall:.4f}/{ds_result.macro_f1:.4f} "
            f"micro_p/r/f1={ds_result.micro_precision:.4f}/{ds_result.micro_recall:.4f}/{ds_result.micro_f1:.4f} "
            f"weighted_p/r/f1={ds_result.weighted_precision:.4f}/{ds_result.weighted_recall:.4f}/{ds_result.weighted_f1:.4f} "
            f"roc_auc_ovr_macro={ds_result.roc_auc_ovr_macro:.4f}"
        )

    eval_cfg = config["evaluation"]
    split_signature = split_signature_for_data(data)
    stage_path = resolve_stage_a_path(config)
    training_summary = dict(getattr(model, "training_summary", {}))
    if os.path.isfile(stage_path):
        stage_metadata = torch.load(stage_path, map_location="cpu").get("metadata") or {}
        if stage_metadata.get("training_summary"):
            training_summary["A"] = stage_metadata["training_summary"]
    if architecture in MOE_ARCHITECTURES:
        stage_b_summary = load_stage_b(config["training"]["checkpoint_dir"]).get("training_summary")
        stage_c_summary = load_stage_c(config["training"]["checkpoint_dir"]).get("training_summary")
        if stage_b_summary:
            training_summary["B"] = stage_b_summary
        if stage_c_summary:
            training_summary["C"] = stage_c_summary
    resource_row = resource_profile(
        model,
        config,
        trial_id=trial_id,
        split_signature=split_signature,
        checkpoint_hash=stage_a_checkpoint_hash(stage_path),
        training_summary=training_summary,
    )
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
        resource_rows=[resource_row],
    )
    print(f"[run] wrote tracker CSVs to {eval_cfg['output_dir']}")


if __name__ == "__main__":
    main()
