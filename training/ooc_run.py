"""CLI entry point for configurable 2/3/4-way full-data NF-v3 MoE runs."""
from __future__ import annotations

import argparse
import os
import sys

import torch

from evaluation.out_of_core_report import evaluate_and_report_ooc

from .checkpoint import (
    HARMONIZER_FILE,
    STAGE_A_FILE,
    STAGE_B_FILE,
    STAGE_C_FILE,
    clear_progress,
    load_stage_c,
    stage_complete,
)
from .config import load_config
from .out_of_core_data import prepare_out_of_core_data
from .out_of_core_train import (
    CONTRACT_FILE,
    STAGE_C_SUMMARY_FILE,
    build_ooc_model,
    ensure_run_contract,
    run_stage_a_ooc,
    run_stage_b_ooc,
    run_stage_c_ooc,
)


def _restart(checkpoint_dir: str) -> None:
    for stage in ("A", "B", "C"):
        clear_progress(checkpoint_dir, stage)
    for filename in (STAGE_A_FILE, STAGE_B_FILE, STAGE_C_FILE, HARMONIZER_FILE, CONTRACT_FILE, STAGE_C_SUMMARY_FILE):
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if config["architecture"] not in {"moe_dataset_soft", "moe_dataset_hard_gate", "moe_dataset_adapters"}:
        raise ValueError("out_of_core_full supports the three MoE architectures, not baseline architectures")
    if config["architecture"] == "moe_dataset_hard_gate":
        config["training"]["stage_c"]["gate_supervision"] = "hard"
    if config["training"].get("device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select a GPU Colab runtime.")
    checkpoint_dir = config["training"]["checkpoint_dir"]
    if config["training"].get("force_restart", False):
        _restart(checkpoint_dir)

    print(f"[ooc-run] python={sys.executable} architecture={config['architecture']}")
    print(f"[ooc-run] datasets={config['data']['active_datasets']}")
    print(f"[ooc-run] checkpoint_dir={checkpoint_dir}")
    context = prepare_out_of_core_data(config)
    contract = ensure_run_contract(config, context)
    print(f"[ooc-run] contract={contract['signature']} split_signatures={context.split_signatures}")
    print(
        f"[ooc-run] rows train={len(context.data.train.class_idx):,} "
        f"val={len(context.data.val.class_idx):,} test={len(context.data.test.class_idx):,}"
    )

    stages = config["training"].get("stages", ["A", "B", "C"])
    for stage, function in (("A", run_stage_a_ooc), ("B", run_stage_b_ooc), ("C", run_stage_c_ooc)):
        if stage not in stages:
            continue
        if stage_complete(checkpoint_dir, stage):
            print(f"[ooc-run] Stage {stage} complete and contract-compatible; skipping")
            continue
        function(config, context)

    if not all(stage_complete(checkpoint_dir, stage) for stage in ("A", "B", "C")):
        raise RuntimeError("Evaluation requires completed Stage A, B, and C checkpoints")
    model = build_ooc_model(config, context, torch.device(config["training"]["device"]))
    model.load_state_dict(load_stage_c(checkpoint_dir)["model_state"])
    reports = evaluate_and_report_ooc(model, context, config, contract)
    print("\n=== Overall and per-origin metrics ===")
    print(reports["overall"].to_string(index=False))
    print(f"[ooc-run] detailed reports={config['evaluation']['output_dir']}")


if __name__ == "__main__":
    main()
