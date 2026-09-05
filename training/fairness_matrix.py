"""Generate or execute the complete three-seed fairness experiment matrix."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

from evaluation.seed_summary import aggregate_seed_results, method_id, write_seed_summary
from .config import load_config


SEEDS = (0, 1, 2)


def _conditions():
    yield "moe_soft_dense", ["architecture=moe_dataset_soft", "model.gate.routing=dense"]
    yield "moe_soft_top1", ["architecture=moe_dataset_soft", "model.gate.routing=top1"]
    for warm in ("none", "matched_exposure"):
        yield f"plain_{warm}", ["architecture=plain_pooled", f"training.baseline.stage_b_warmstart={warm}"]
        for axis, route in (("total_params", "dense"), ("active_params", "top1"), ("forward_macs", "top1")):
            yield f"matched_{axis}_{warm}", [
                "architecture=matched_dense", f"model.dense_match.axis={axis}",
                f"model.dense_match.reference_routing={route}",
                f"training.baseline.stage_b_warmstart={warm}",
            ]
    for init in ("stage_a", "random"):
        yield f"hard_{init}", ["architecture=hard_two_stage", f"training.baseline.encoder_init={init}"]
        yield f"no_fusion_{init}", ["architecture=no_fusion", f"training.baseline.encoder_init={init}"]
    yield "moe_basic", ["architecture=moe_basic", "model.gate.routing=dense"]
    yield "moe_damex", ["architecture=moe_dataset_damex", "model.gate.routing=dense"]
    yield "moe_hard_gate", ["architecture=moe_dataset_hard_gate", "model.gate.routing=dense"]
    yield "moe_adapters", ["architecture=moe_dataset_adapters", "model.gate.routing=dense"]


def _command(module: str, config_path: str, overrides: list[str]) -> list[str]:
    command = [sys.executable, "-m", module, "--config", config_path]
    for value in overrides:
        command.extend(["--set", value])
    return command


def build_matrix(config_path: str, prefix: str, module: str):
    commands = []
    results = []
    for seed in SEEDS:
        pretrain_name = f"{prefix}_seed{seed}_stage_a"
        pretrain_overrides = [
            f"run_name={pretrain_name}", f"seed={seed}", "architecture=moe_dataset_soft",
            "training.stages=[A]", "training.selection_mode=fixed_epochs",
        ]
        pretrain_config = load_config(config_path, pretrain_overrides)
        stage_a_path = os.path.join(pretrain_config["training"]["checkpoint_dir"], "stage_a_encoder.pt")
        commands.append((f"seed{seed}:stage_a", _command(module, config_path, pretrain_overrides)))
        for label, condition in _conditions():
            run_name = f"{prefix}_seed{seed}_{label}"
            init = next((value.split("=", 1)[1] for value in condition if value.startswith("training.baseline.encoder_init=")), "stage_a")
            stages = "[B,C]" if condition[0].split("=", 1)[1].startswith("moe_") else "[B,C]"
            overrides = [
                f"run_name={run_name}", f"seed={seed}", "training.selection_mode=fixed_epochs",
                f"training.stages={stages}", *condition,
            ]
            if init == "stage_a":
                overrides.append(f"training.stage_a_checkpoint={stage_a_path}")
            run_config = load_config(config_path, overrides)
            results.append(run_config["evaluation"]["output_dir"])
            commands.append((f"seed{seed}:{label}", _command(module, config_path, overrides)))
    return commands, results


def _aggregate(result_dirs: list[str], output_dir: str) -> str:
    trials = pd.concat([pd.read_csv(os.path.join(path, "Trials.csv")) for path in result_dirs], ignore_index=True)
    overall_frames = []
    dataset_frames = []
    for path in result_dirs:
        overall = pd.read_csv(os.path.join(path, "Overall_Metrics.csv"))
        if "origin" in overall:
            overall = overall[overall["origin"] == "ALL"]
        overall_frames.append(overall)
        dataset_frames.append(pd.read_csv(os.path.join(path, "Per_Dataset_Metrics.csv")))
    overall = pd.concat(overall_frames, ignore_index=True)
    per_dataset = pd.concat(dataset_frames, ignore_index=True)
    reference_row = trials[
        (trials["architecture"] == "moe_dataset_soft") & (trials["routing_mode"] == "dense")
    ].iloc[0]
    summary = aggregate_seed_results(
        trials, overall, per_dataset, reference_method=method_id(reference_row)
    )
    return write_seed_summary(output_dir, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--prefix", default="fair_comparison_v1")
    parser.add_argument("--execution-mode", choices=("in_memory", "out_of_core"), default="out_of_core")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summary-dir")
    args = parser.parse_args()
    module = "training.run" if args.execution_mode == "in_memory" else "training.ooc_run"
    commands, result_dirs = build_matrix(args.config, args.prefix, module)
    for label, command in commands:
        print(f"[{label}] {' '.join(command)}")
        if args.execute:
            subprocess.run(command, check=True)
    if args.execute:
        output_dir = args.summary_dir or os.path.join(os.path.dirname(result_dirs[0]), f"{args.prefix}_summary")
        print(f"[fairness-matrix] seed summary={_aggregate(result_dirs, output_dir)}")


if __name__ == "__main__":
    main()
