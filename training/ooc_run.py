"""CLI entry point for configurable 2/3/4-way full-data NF-v3 MoE runs."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

if __name__ == "__main__":
    print("[ooc-run] child process started; importing training dependencies...", flush=True)

import torch

from evaluation.latent_space import evaluate_latent_checkpoints, plot_latent_snapshots
from evaluation.out_of_core_report import evaluate_and_report_ooc

from .checkpoint import (
    HARMONIZER_FILE,
    STAGE_A_FILE,
    STAGE_B_FILE,
    STAGE_C_FILE,
    BASELINE_MODEL_FILE,
    BASELINE_STAGE_B_FILE,
    HARD_ROUTER_FILE,
    HARD_CLASSIFIERS_FILE,
    clear_progress,
    load_stage_c,
    load_configured_stage_b,
    resolve_stage_a_path,
    resolve_stage_b_path,
    stage_complete,
    hard_stage_complete,
)
from .config import load_config
from .logging_utils import tee_stdout_to_file
from .out_of_core_data import prepare_out_of_core_data
from .reproducibility import configure_reproducibility
from .hard_two_stage_ooc import (
    load_hard_two_stage_ooc,
    run_hard_phase_a_ooc,
    run_hard_phase_b_ooc,
)
from .dense_ooc import load_dense_ooc, run_dense_stage_b_ooc, run_dense_stage_c_ooc
from .no_fusion_ooc import load_no_fusion_ooc, run_no_fusion_ooc
from .out_of_core_train import (
    CONTRACT_FILE,
    STAGE_C_SUMMARY_FILE,
    build_ooc_model,
    ensure_run_contract,
    run_stage_a_ooc,
    run_stage_b_ooc,
    run_stage_c_ooc,
)
from .training_history import TRAINING_HISTORY_FILE


def _restart(checkpoint_dir: str) -> None:
    for stage in ("A", "B", "C", "DB", "DC", "NF", "HR", "HC"):
        clear_progress(checkpoint_dir, stage)
    for filename in (
        STAGE_A_FILE, STAGE_B_FILE, STAGE_C_FILE, BASELINE_MODEL_FILE, BASELINE_STAGE_B_FILE,
        HARD_ROUTER_FILE, HARD_CLASSIFIERS_FILE,
        HARMONIZER_FILE, CONTRACT_FILE, STAGE_C_SUMMARY_FILE,
        TRAINING_HISTORY_FILE,
    ):
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--latent-only",
        action="store_true",
        help="load completed checkpoints and regenerate only the configured latent report",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if "deterministic" in config.get("training", {}):
        configure_reproducibility(
            config.get("seed", 0),
            deterministic=bool(config["training"]["deterministic"]),
        )
    checkpoint_dir = config["training"]["checkpoint_dir"]
    tee_stdout_to_file(os.path.join(checkpoint_dir, "train.log"))
    print("\n" + "=" * 80, flush=True)
    print(f"[ooc-run] started={time.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    print(f"[ooc-run] command={' '.join(sys.argv)}", flush=True)
    print(
        f"[ooc-run] cuda_available={torch.cuda.is_available()} "
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        flush=True,
    )
    scratch_dir = os.environ.get("NIDS_SCRATCH_DIR", "/content/dataset_moe_nids_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    for label, path in (("checkpoint", checkpoint_dir), ("scratch", scratch_dir)):
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            print(f"[ooc-run] {label}_storage path={path} unavailable: {exc}", flush=True)
        else:
            print(
                f"[ooc-run] {label}_storage path={path} "
                f"free={usage.free / 2**30:.1f}GiB total={usage.total / 2**30:.1f}GiB",
                flush=True,
            )
    if config["architecture"] not in {
        "moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_damex",
        "moe_dataset_adapters", "moe_dataset_private_encoders", "moe_basic",
        "moe_dataset_class_conditional",
        "hard_two_stage", "plain_pooled", "matched_dense", "no_fusion",
    }:
        raise ValueError("out_of_core_full supports MoE, dense, and hard_two_stage architectures")
    if config["training"].get("device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select a GPU Colab runtime.")
    if config["training"].get("force_restart", False):
        _restart(checkpoint_dir)

    print(f"[ooc-run] python={sys.executable} architecture={config['architecture']}", flush=True)
    print(
        f"[ooc-run] seed={config.get('seed', 0)} "
        f"split_seed={config['data'].get('split_seed', config.get('seed', 0))} "
        f"deterministic={bool(config['training'].get('deterministic', False))}",
        flush=True,
    )
    print(f"[ooc-run] datasets={config['data']['active_datasets']}", flush=True)
    print(f"[ooc-run] checkpoint_dir={checkpoint_dir}", flush=True)
    print(
        f"[ooc-run] schedule stages={config['training'].get('stages')} "
        f"epochs_a={config['training'].get('epochs_a')} "
        f"epochs_b={config['training'].get('epochs_b')} "
        f"epochs_c={config['training'].get('epochs_c')} "
        f"batch_size={config['training'].get('batch_size')}",
        flush=True,
    )
    context = prepare_out_of_core_data(config)
    contract = ensure_run_contract(config, context)
    print(f"[ooc-run] contract={contract['signature']} split_signatures={context.split_signatures}")
    print(
        f"[ooc-run] rows train={len(context.data.train.class_idx):,} "
        f"val={len(context.data.val.class_idx):,} test={len(context.data.test.class_idx):,}"
    )

    if args.latent_only:
        if (
            not os.path.isfile(resolve_stage_a_path(config))
            or not os.path.isfile(resolve_stage_b_path(config))
            or not stage_complete(checkpoint_dir, "C")
        ):
            raise RuntimeError("--latent-only requires completed Stage A, B, and C checkpoints")
        _run_latent_evaluation(config, context)
        return

    stages = config["training"].get("stages", ["A", "B", "C"])
    if config["architecture"] == "no_fusion":
        baseline_cfg = config["training"].get("baseline", {})
        if baseline_cfg.get("encoder_init", "stage_a") == "stage_a":
            source = config["training"].get("stage_a_checkpoint") or baseline_cfg.get("stage_a_checkpoint")
            if not source and not stage_complete(checkpoint_dir, "A"):
                if "A" not in stages:
                    raise RuntimeError("No-fusion Stage-A initialization requested but no checkpoint exists")
                run_stage_a_ooc(config, context)
        model = (
            load_no_fusion_ooc(config, context)
            if os.path.isfile(os.path.join(checkpoint_dir, BASELINE_MODEL_FILE))
            else run_no_fusion_ooc(config, context)
        )
        reports = evaluate_and_report_ooc(model, context, config, contract)
        print("\n=== Overall and per-origin metrics ===")
        print(reports["overall"].to_string(index=False))
        return

    if config["architecture"] in {"plain_pooled", "matched_dense"}:
        baseline_cfg = config["training"].get("baseline", {})
        if baseline_cfg.get("encoder_init", "stage_a") == "stage_a":
            source = config["training"].get("stage_a_checkpoint") or baseline_cfg.get("stage_a_checkpoint")
            if not source and not stage_complete(checkpoint_dir, "A"):
                if "A" not in stages:
                    raise RuntimeError("Dense Stage-A initialization requested but no checkpoint exists")
                run_stage_a_ooc(config, context)
        if not os.path.isfile(os.path.join(checkpoint_dir, BASELINE_MODEL_FILE)):
            if "C" not in stages:
                if "B" in stages:
                    run_dense_stage_b_ooc(config, context)
                print("[ooc-run] Dense Stage B complete; Stage C was not requested")
                return
            if "B" not in stages and not config["training"].get("dense_stage_b_checkpoint"):
                raise RuntimeError(
                    "Dense Stage C requires Stage B in training.stages or "
                    "training.dense_stage_b_checkpoint"
                )
            model = run_dense_stage_b_ooc(config, context)
            model = run_dense_stage_c_ooc(config, context, model)
        else:
            model = load_dense_ooc(config, context)
        if not config["training"].get("run_final_evaluation", True):
            print("[ooc-run] Dense checkpoints and validation report are ready; test evaluation deferred")
            return
        reports = evaluate_and_report_ooc(model, context, config, contract)
        print("\n=== Overall and per-origin metrics ===")
        print(reports["overall"].to_string(index=False))
        print(f"[ooc-run] detailed reports={config['evaluation']['output_dir']}")
        return

    if config["architecture"] == "hard_two_stage":
        baseline_cfg = config["training"].get("baseline", {})
        if baseline_cfg.get("encoder_init", "stage_a") == "stage_a":
            source = config["training"].get("stage_a_checkpoint") or baseline_cfg.get("stage_a_checkpoint")
            if not source and not stage_complete(checkpoint_dir, "A"):
                if "A" not in stages:
                    raise RuntimeError("Hard two-stage Stage-A initialization requested but no checkpoint exists")
                run_stage_a_ooc(config, context)
        if "B" in stages and not hard_stage_complete(checkpoint_dir, "router"):
            run_hard_phase_a_ooc(config, context)
        if "B" in stages and not hard_stage_complete(checkpoint_dir, "classifiers"):
            run_hard_phase_b_ooc(config, context)
        if not all(hard_stage_complete(checkpoint_dir, stage) for stage in ("router", "classifiers")):
            raise RuntimeError("Hard two-stage evaluation requires completed router and classifier phases")
        model = load_hard_two_stage_ooc(config, context)
        reports = evaluate_and_report_ooc(model, context, config, contract)
        print("\n=== Overall and per-origin metrics ===")
        print(reports["overall"].to_string(index=False))
        print(f"[ooc-run] detailed reports={config['evaluation']['output_dir']}")
        return

    for stage, function in (("A", run_stage_a_ooc), ("B", run_stage_b_ooc), ("C", run_stage_c_ooc)):
        if stage not in stages:
            continue
        if stage_complete(checkpoint_dir, stage):
            print(f"[ooc-run] Stage {stage} complete and contract-compatible; skipping")
            continue
        function(config, context)

    if (
        not os.path.isfile(resolve_stage_a_path(config))
        or not os.path.isfile(resolve_stage_b_path(config))
        or not stage_complete(checkpoint_dir, "C")
    ):
        print("[ooc-run] Requested stages completed; full evaluation awaits Stage A+B+C checkpoints")
        return
    if not config["training"].get("run_final_evaluation", True):
        print("[ooc-run] All checkpoints are ready; final evaluation deferred by configuration")
        return
    model = build_ooc_model(config, context, torch.device(config["training"]["device"]))
    stage_c_checkpoint = load_stage_c(checkpoint_dir)
    model.load_state_dict(stage_c_checkpoint["model_state"])
    model.training_summary = {}
    stage_b_summary = load_configured_stage_b(config).get("training_summary")
    stage_c_summary = stage_c_checkpoint.get("training_summary")
    if stage_b_summary:
        model.training_summary["B"] = stage_b_summary
    if stage_c_summary:
        model.training_summary["C"] = stage_c_summary
    reports = evaluate_and_report_ooc(model, context, config, contract)
    print("\n=== Overall and per-origin metrics ===")
    print(reports["overall"].to_string(index=False))
    print(f"[ooc-run] detailed reports={config['evaluation']['output_dir']}")
    _run_latent_evaluation(config, context)


def _run_latent_evaluation(config: dict, context) -> None:
    latent_cfg = config.get("evaluation", {}).get("latent", {})
    if not latent_cfg.get("enabled", False):
        return
    latent_output_dir = os.path.join(config["evaluation"]["output_dir"], "latent")
    report = evaluate_latent_checkpoints(
        config, context,
        split_name=latent_cfg.get("split", "val"),
        output_dir=latent_output_dir,
        device=config["training"]["device"],
        stages=latent_cfg.get("stages", ["C"]),
    )
    print(f"[ooc-run] latent report written to {latent_output_dir}")
    if latent_cfg.get("plot", False):
        plot_latent_snapshots(report, latent_output_dir, method=latent_cfg.get("plot_method", "umap"))
        print(f"[ooc-run] latent plots written to {os.path.join(latent_output_dir, 'latent_figures')}")


if __name__ == "__main__":
    started = time.monotonic()
    try:
        main()
    except Exception as exc:
        print(
            f"[ooc-run] FAILED after {(time.monotonic() - started) / 60:.1f} min: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    else:
        print(
            f"[ooc-run] completed successfully in {(time.monotonic() - started) / 60:.1f} min",
            flush=True,
        )
