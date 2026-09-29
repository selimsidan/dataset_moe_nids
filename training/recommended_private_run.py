"""Fixed three-seed runner for the recommended asymmetric private MoE.

This workflow is intentionally independent from the greedy-v2 and depth-v3
state machines.  It performs no hyperparameter search: validation macro-F1
selects an epoch within each seed, then the locked checkpoint is evaluated on
test data.  Latent reports use validation rows only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import load_config
from .training_history import (
    HISTORY_COLUMNS,
    load_or_reconstruct_history,
    plot_cross_seed_training_history,
    plot_seed_training_history,
    summarize_training_history,
)


STAGE_A_FILE = "stage_a_encoder.pt"
STAGE_B_FILE = "stage_b_expert_bank.pt"
STAGE_C_FILE = "stage_c_full.pt"

REQUIRED_TEST_REPORTS = (
    "manifest.json",
    "Trials.csv",
    "Overall_Metrics.csv",
    "Per_Dataset_Metrics.csv",
    "Per_Class_Metrics.csv",
    "Resource_Accounting.csv",
)
LATENT_REPORT_FILES = (
    "Latent_Report_Config.json",
    "Latent_Snapshot_Metrics.csv",
    "Latent_Per_Class.csv",
    "Latent_Probe_Metrics.csv",
    "Latent_Probe_Per_Class.csv",
    "Latent_Sample_Manifest.csv",
    "Latent_Train_Sample_Manifest.csv",
    "Latent_Embeddings.npz",
    "Latent_Train_Embeddings.npz",
)
TRAINING_REPORT_FILES = (
    "Training_History.csv",
    "Training_Log_Manifest.csv",
    "Training_Report_Config.json",
)


def _hash(value: Any, length: int = 64) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_format_override(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)


def flatten_overrides(node: dict[str, Any], prefix: str = "") -> list[str]:
    result = []
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.extend(flatten_overrides(value, dotted))
        else:
            result.append(f"{dotted}={_format_override(value)}")
    return result


def _atomic_csv(frame: pd.DataFrame, path: str | os.PathLike) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: str | os.PathLike) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _mean_sd(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {*keys, "seed"}
    ]
    if not numeric:
        return frame[keys].drop_duplicates().reset_index(drop=True)
    if keys:
        grouped = frame.groupby(keys, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    else:
        values = {}
        for column in numeric:
            values[f"{column}__mean"] = [float(frame[column].mean())]
            values[f"{column}__std"] = [float(frame[column].std(ddof=1))]
        return pd.DataFrame(values)
    grouped.columns = [
        column if isinstance(column, str) else "__".join(value for value in column if value)
        for column in grouped.columns
    ]
    return grouped


def _save_summary_plots(summary_dir: str, latent: dict[str, pd.DataFrame]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(summary_dir, "latent_figures")
    os.makedirs(figure_dir, exist_ok=True)
    written = []

    snapshot = latent["snapshot_summary"]
    labels = snapshot["snapshot"].astype(str).tolist()
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 0.75), 10), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "silhouette", "Cosine silhouette"),
        (axes[1], "effective_rank", "Effective rank"),
    ):
        mean = snapshot[f"{metric}__mean"].to_numpy()
        std = snapshot[f"{metric}__std"].fillna(0).to_numpy()
        axis.errorbar(positions, mean, yerr=std, marker="o", capsize=3)
        axis.set_title(f"{title} across three seeds")
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.grid(alpha=0.25)
    path = os.path.join(figure_dir, "latent_geometry_across_seeds.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); written.append(path)

    probes = latent["probe_summary"]
    fig, axis = plt.subplots(figsize=(max(12, len(probes) * 0.45), 7), constrained_layout=True)
    probe_labels = (
        probes["snapshot"].astype(str) + "\n" + probes["target"].astype(str)
        + "/" + probes["probe"].astype(str)
    )
    axis.errorbar(
        np.arange(len(probes)), probes["macro_f1_present__mean"],
        yerr=probes["macro_f1_present__std"].fillna(0), marker="o", capsize=3,
    )
    axis.set_ylim(0, 1); axis.set_ylabel("Macro-F1"); axis.grid(alpha=0.25)
    axis.set_title("Frozen class/domain probes across three seeds")
    axis.set_xticks(np.arange(len(probes)), probe_labels, rotation=70, ha="right", fontsize=7)
    path = os.path.join(figure_dir, "latent_probe_across_seeds.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); written.append(path)

    def heatmap(frame, value, filename, title, row="class", column="snapshot"):
        table = frame.pivot_table(index=row, columns=column, values=value, aggfunc="mean")
        fig, axis = plt.subplots(
            figsize=(max(10, 0.7 * len(table.columns)), max(7, 0.34 * len(table.index))),
            constrained_layout=True,
        )
        image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(np.arange(len(table.columns)), table.columns, rotation=70, ha="right", fontsize=7)
        axis.set_yticks(np.arange(len(table.index)), table.index, fontsize=7)
        axis.set_title(title); fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        path = os.path.join(figure_dir, filename)
        fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); written.append(path)

    probe_class = latent["probe_class_all"]
    linear = probe_class[probe_class["probe"] == "linear"]
    heatmap(
        linear, "recall", "latent_class_probe_recall_heatmap.png",
        "Linear-probe class recall across latent snapshots",
    )
    heatmap(
        latent["per_class_all"], "cross_dataset_centroid_dispersion",
        "latent_cross_dataset_dispersion_heatmap.png",
        "Cross-dataset class-centroid dispersion",
    )
    return written


class RecommendedPrivateStudy:
    def __init__(
        self,
        *,
        base_config_path: str,
        study_config_path: str,
        prefix: str,
        execute: bool = False,
        max_new_seeds: int | None = None,
        summary_dir: str | None = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.study_config_path = study_config_path
        self.prefix = prefix
        self.execute = execute
        self.max_new_seeds = max_new_seeds
        with open(study_config_path) as handle:
            self.study = yaml.safe_load(handle)
        if self.study.get("format_version") != 1:
            raise ValueError("recommended private study requires format_version: 1")
        if self.study.get("seeds") != [0, 1, 2]:
            raise ValueError("recommended private study requires seeds [0, 1, 2]")
        base = load_config(base_config_path)
        self.summary_dir = summary_dir or os.path.join(
            base["OUTPUT_DIR"], "results", f"{prefix}_summary"
        )
        self.state_path = os.path.join(self.summary_dir, "study_state.json")
        with open(base_config_path, "rb") as handle:
            base_sha = hashlib.sha256(handle.read()).hexdigest()
        self.protocol_hash = _hash({"study": self.study, "base_sha256": base_sha})
        self.state = self._load_state()
        self._contexts: dict[int, Any] = {}

    def _load_state(self) -> dict:
        if not os.path.isfile(self.state_path):
            return {
                "format_version": 1,
                "protocol_hash": self.protocol_hash,
                "prefix": self.prefix,
                "seeds": {},
            }
        with open(self.state_path) as handle:
            state = json.load(handle)
        if state.get("protocol_hash") != self.protocol_hash:
            raise ValueError("Study protocol changed after execution began; use a new prefix")
        return state

    def _save_state(self) -> None:
        if self.execute:
            _atomic_json(self.state, self.state_path)

    def _base_nested(self, seed: int, run_name: str) -> dict:
        nested = copy.deepcopy(self.study["backbone"])
        nested["run_name"] = run_name
        nested["seed"] = seed
        nested.setdefault("data", {})["split_seed"] = seed
        nested.setdefault("evaluation", {}).setdefault("latent", {})["random_seed"] = seed
        nested.setdefault("training", {})["save_epoch_history"] = True
        return nested

    def _config(self, nested: dict) -> dict:
        return load_config(self.base_config_path, flatten_overrides(nested))

    def _stage_a_nested(self, seed: int, role: str) -> dict:
        label = "gate" if role == "gate_encoder" else "private"
        nested = self._base_nested(seed, f"{self.prefix}_cache_{label}_a_seed{seed}")
        nested["training"]["stages"] = ["A"]
        nested["training"]["stage_a_encoder_role"] = role
        nested["training"]["run_final_evaluation"] = False
        return nested

    def _stage_a_path(self, seed: int, role: str) -> str:
        config = self._config(self._stage_a_nested(seed, role))
        return os.path.join(config["training"]["checkpoint_dir"], STAGE_A_FILE)

    def _stage_b_nested(self, seed: int) -> dict:
        nested = self._base_nested(seed, f"{self.prefix}_cache_b_seed{seed}")
        nested["training"]["stages"] = ["B"]
        nested["training"]["run_final_evaluation"] = False
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(seed, "gate_encoder")
        nested["training"]["private_stage_a_checkpoint"] = self._stage_a_path(seed, "private_encoder")
        return nested

    def _stage_b_path(self, seed: int) -> str:
        config = self._config(self._stage_b_nested(seed))
        return os.path.join(config["training"]["checkpoint_dir"], STAGE_B_FILE)

    def _final_nested(self, seed: int, *, evaluate: bool) -> dict:
        nested = self._base_nested(seed, f"{self.prefix}_seed{seed}")
        nested["training"]["stages"] = [] if evaluate else ["C"]
        nested["training"]["run_final_evaluation"] = bool(evaluate)
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(seed, "gate_encoder")
        nested["training"]["private_stage_a_checkpoint"] = self._stage_a_path(seed, "private_encoder")
        nested["training"]["stage_b_checkpoint"] = self._stage_b_path(seed)
        return nested

    def _final_config(self, seed: int) -> dict:
        return self._config(self._final_nested(seed, evaluate=False))

    def _command(self, nested: dict) -> list[str]:
        command = [sys.executable, "-u", "-m", "training.ooc_run", "--config", self.base_config_path]
        for override in flatten_overrides(nested):
            command.extend(["--set", override])
        return command

    def _run(self, label: str, nested: dict) -> None:
        command = self._command(nested)
        resolved = self._config(nested)
        print(
            f"[{label}] starting seed={nested['seed']} "
            f"train_log={os.path.join(resolved['training']['checkpoint_dir'], 'train.log')}",
            flush=True,
        )
        print(f"[{label}] {shlex.join(command)}", flush=True)
        if self.execute:
            child_env = os.environ.copy(); child_env["PYTHONUNBUFFERED"] = "1"
            subprocess.run(command, check=True, env=child_env)
            print(f"[{label}] completed seed={nested['seed']}", flush=True)

    def _result_dir(self, seed: int) -> str:
        return self._final_config(seed)["evaluation"]["output_dir"]

    def _test_complete(self, seed: int) -> bool:
        directory = self._result_dir(seed)
        return all(os.path.isfile(os.path.join(directory, name)) for name in REQUIRED_TEST_REPORTS)

    def _training_dir(self, seed: int) -> str:
        return os.path.join(self._result_dir(seed), "training")

    def _training_complete(self, seed: int) -> bool:
        directory = self._training_dir(seed)
        if not all(os.path.isfile(os.path.join(directory, name)) for name in TRAINING_REPORT_FILES):
            return False
        try:
            with open(os.path.join(directory, "Training_Report_Config.json")) as handle:
                figures = json.load(handle).get("figure_files", [])
        except (OSError, ValueError):
            return False
        return all(os.path.isfile(path) for path in figures)

    def _latent_dir(self, seed: int, stage: str) -> str:
        return os.path.join(self._result_dir(seed), "latent", f"stage_{stage}")

    def _latent_complete(self, seed: int, stage: str) -> bool:
        directory = self._latent_dir(seed, stage)
        return all(os.path.isfile(os.path.join(directory, name)) for name in LATENT_REPORT_FILES)

    def _projection_complete(self, directory: str, method: str) -> bool:
        figure_dir = os.path.join(directory, "latent_figures")
        config_path = os.path.join(figure_dir, f"{method}_projection_config.json")
        archive_path = os.path.join(figure_dir, f"{method}_projections.npz")
        if not os.path.isfile(config_path) or not os.path.isfile(archive_path):
            return False
        try:
            with open(config_path) as handle:
                snapshots = json.load(handle)["snapshots"]
        except (OSError, ValueError, KeyError):
            return False
        return all(
            os.path.isfile(os.path.join(
                figure_dir, f"{snapshot.replace('::', '_')}__{method}.png"
            ))
            for snapshot in snapshots
        )

    def _latent_seed_complete(self, seed: int) -> bool:
        cumulative = os.path.join(self._result_dir(seed), "latent", "cumulative")
        if not all(self._latent_complete(seed, stage) for stage in ("A", "B", "C")):
            return False
        if not all(
            self._projection_complete(self._latent_dir(seed, stage), method)
            for stage in ("A", "B", "C") for method in ("pca", "umap")
        ):
            return False
        if not all(os.path.isfile(os.path.join(cumulative, name)) for name in LATENT_REPORT_FILES):
            return False
        if not all(self._projection_complete(cumulative, method) for method in ("pca", "umap")):
            return False
        manifest_path = os.path.join(cumulative, "Latent_Sample_Manifest.csv")
        if not os.path.isfile(manifest_path):
            return False
        classes = sorted(set(pd.read_csv(manifest_path)["class"].astype(str)))
        return all(
            os.path.isfile(os.path.join(
                cumulative, "latent_figures",
                f"class_focus__{name.replace('/', '_').replace(' ', '_')}.png",
            ))
            for name in classes
        )

    def _context(self, seed: int):
        if seed not in self._contexts:
            from .out_of_core_data import prepare_out_of_core_data
            self._contexts[seed] = prepare_out_of_core_data(self._final_config(seed))
        return self._contexts[seed]

    def _preflight(self, seed: int) -> None:
        import torch
        from evaluation.resource_accounting import parameter_count
        from models.encoder import build_encoder, resolve_encoder_config
        from .model_utils import build_model

        config = self._final_config(seed)
        context = self._context(seed)
        data = context.data
        expected_features = int(self.study["expected_input_features"])
        expected_classes = int(self.study["expected_classes"])
        if data.train.features.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, found {data.train.features.shape[1]}")
        if len(data.class_names) != expected_classes:
            raise ValueError(f"Expected {expected_classes} classes, found {len(data.class_names)}")
        model_cfg = config["model"]
        encoder = build_encoder(
            expected_features, model_cfg["latent_dim"],
            resolve_encoder_config(model_cfg, "gate_encoder"),
        )
        model = build_model(
            config["architecture"], encoder, data.active_datasets, data.class_names, model_cfg
        )
        actual = parameter_count(model)
        expected = int(self.study["expected_total_parameters"])
        if actual != expected:
            raise ValueError(f"Expected {expected:,} parameters, found {actual:,}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _materialize_latent(self, seed: int, stage: str) -> None:
        if not self.execute:
            print(f"[latent-{stage}] seed {seed}: evaluate metrics; save PCA and UMAP", flush=True)
            return
        from evaluation.latent_space import (
            combine_latent_reports,
            evaluate_latent_checkpoints,
            load_latent_report,
            plot_class_focus,
            plot_latent_snapshots,
        )

        config = self._final_config(seed)
        context = self._context(seed)
        stage_dir = self._latent_dir(seed, stage)
        stage_a_dir = self._latent_dir(seed, "A")
        reference = load_latent_report(stage_a_dir) if stage == "B" else None
        if self._latent_complete(seed, stage):
            report = load_latent_report(stage_dir)
            print(f"[latent-{stage}] seed {seed}: reusing metrics", flush=True)
        else:
            report = evaluate_latent_checkpoints(
                config,
                context,
                split_name="val",
                output_dir=stage_dir,
                device=config["training"]["device"],
                stages=[stage],
                sample_manifest=stage_a_dir if stage != "A" else None,
                reference_report=reference,
                reuse_completed=True,
            )
        plot_latent_snapshots(report, stage_dir, method="pca", random_seed=seed)
        plot_latent_snapshots(report, stage_dir, method="umap", random_seed=seed)
        reports = [
            load_latent_report(self._latent_dir(seed, current))
            for current in ("A", "B", "C")
            if self._latent_complete(seed, current)
        ]
        cumulative_dir = os.path.join(self._result_dir(seed), "latent", "cumulative")
        cumulative = combine_latent_reports(reports, cumulative_dir)
        plot_latent_snapshots(cumulative, cumulative_dir, method="pca", random_seed=seed)
        umap_projection = plot_latent_snapshots(
            cumulative, cumulative_dir, method="umap", random_seed=seed
        )
        if stage == "C":
            for class_name in sorted(set(cumulative["manifest"]["class"].astype(str))):
                plot_class_focus(cumulative, umap_projection, class_name, cumulative_dir)

    def _materialize_training_report(self, seed: int) -> None:
        if not self.execute:
            print(
                f"[training-report] seed {seed}: combine epoch history, logs, and curves",
                flush=True,
            )
            return
        final = self._final_config(seed)
        training_dir = self._training_dir(seed)
        os.makedirs(training_dir, exist_ok=True)
        sources = []
        for role in ("gate_encoder", "private_encoder"):
            nested = self._stage_a_nested(seed, role)
            config = self._config(nested)
            sources.append({
                "stage": "A", "encoder_role": role,
                "checkpoint_dir": config["training"]["checkpoint_dir"],
                "learning_rate": config["training"]["stage_a"]["optimizer"]["lr"],
            })
        stage_b = self._config(self._stage_b_nested(seed))
        sources.append({
            "stage": "B", "encoder_role": "private_expert",
            "checkpoint_dir": stage_b["training"]["checkpoint_dir"],
            "learning_rate": stage_b["training"]["stage_b"]["optimizer"]["lr"],
        })
        sources.append({
            "stage": "C", "encoder_role": "full_model",
            "checkpoint_dir": final["training"]["checkpoint_dir"],
            "learning_rate": final["training"]["stage_c"]["optimizer"]["lr"],
        })

        histories = []
        log_rows = []
        for source in sources:
            log_path = os.path.join(source["checkpoint_dir"], "train.log")
            history = load_or_reconstruct_history(
                source["checkpoint_dir"], seed=seed, stage=source["stage"],
                encoder_role=source["encoder_role"],
                learning_rate=float(source["learning_rate"]),
            )
            if not history.empty:
                histories.append(history)
            log_rows.append({
                "seed": seed, "stage": source["stage"],
                "encoder_role": source["encoder_role"],
                "checkpoint_dir": source["checkpoint_dir"], "log_path": log_path,
                "log_exists": os.path.isfile(log_path),
                "log_size_bytes": os.path.getsize(log_path) if os.path.isfile(log_path) else 0,
                "structured_history_exists": os.path.isfile(os.path.join(
                    source["checkpoint_dir"], "Training_History.csv"
                )),
                "history_rows": int(len(history)),
            })
        history = (
            pd.concat(histories, ignore_index=True)
            if histories else pd.DataFrame(columns=HISTORY_COLUMNS)
        )
        stage_c = history[history["stage"] == "C"] if not history.empty else history
        if not stage_c.empty:
            missing_total = (history["stage"] == "C") & history["train_total_loss"].isna()
            stage_cfg = final["training"]["stage_c"]
            history.loc[missing_total, "train_total_loss"] = (
                history.loc[missing_total, "train_ce_loss"]
                + float(final["load_balance"]["lambda_balance"])
                * history.loc[missing_total, "train_balance_penalty"]
                + float(stage_cfg.get("lambda_dataset_aux", 0.0))
                * history.loc[missing_total, "train_dataset_aux_loss"]
                + float(stage_cfg.get("lambda_expert_anchor", 0.0))
                * history.loc[missing_total, "train_anchor_penalty"]
            )
        summary_path = os.path.join(
            final["training"]["checkpoint_dir"], "stage_c_training_summary.json"
        )
        selected_epoch = None
        if os.path.isfile(summary_path):
            with open(summary_path) as handle:
                selected_epoch = json.load(handle).get("best_epoch")
        _atomic_csv(history, os.path.join(training_dir, "Training_History.csv"))
        log_manifest = pd.DataFrame(log_rows)
        _atomic_csv(log_manifest, os.path.join(training_dir, "Training_Log_Manifest.csv"))
        figure_dir = os.path.join(training_dir, "training_figures")
        figures = plot_seed_training_history(
            history, figure_dir, selected_epoch=selected_epoch
        )
        present_stages = sorted(set(history["stage"].astype(str))) if not history.empty else []
        report = {
            "format_version": 1,
            "seed": seed,
            "selected_stage_c_epoch": selected_epoch,
            "present_stages": present_stages,
            "missing_stages": [stage for stage in ("A", "B", "C") if stage not in present_stages],
            "history_sources": sorted(set(history["history_source"].astype(str))) if not history.empty else [],
            "figure_files": figures,
            "log_manifest": os.path.join(training_dir, "Training_Log_Manifest.csv"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(report, os.path.join(training_dir, "Training_Report_Config.json"))
        print(
            f"[training-report] seed {seed}: rows={len(history)} figures={len(figures)} "
            f"directory={training_dir}",
            flush=True,
        )

    def _seed_complete(self, seed: int) -> bool:
        return (
            self._test_complete(seed)
            and self._latent_seed_complete(seed)
            and self._training_complete(seed)
        )

    def run_seed(self, seed: int) -> None:
        if self.execute:
            self._preflight(seed)
        for role in ("gate_encoder", "private_encoder"):
            nested = self._stage_a_nested(seed, role)
            path = self._stage_a_path(seed, role)
            if self.execute and os.path.isfile(path):
                print(f"[stage-a:{role}] seed {seed}: complete; skipping", flush=True)
            else:
                self._run(f"stage-a:{role}", nested)
        self._materialize_latent(seed, "A")

        if self.execute and os.path.isfile(self._stage_b_path(seed)):
            print(f"[stage-b] seed {seed}: complete; skipping", flush=True)
        else:
            self._run("stage-b", self._stage_b_nested(seed))
        self._materialize_latent(seed, "B")

        final = self._final_config(seed)
        stage_c_path = os.path.join(final["training"]["checkpoint_dir"], STAGE_C_FILE)
        if self.execute and os.path.isfile(stage_c_path):
            print(f"[stage-c] seed {seed}: complete; skipping", flush=True)
        else:
            self._run("stage-c", self._final_nested(seed, evaluate=False))
        self._materialize_latent(seed, "C")

        if self.execute and self._test_complete(seed):
            print(f"[locked-test] seed {seed}: complete; skipping", flush=True)
        else:
            self._run("locked-test", self._final_nested(seed, evaluate=True))
        self._materialize_training_report(seed)
        if self.execute and not self._seed_complete(seed):
            raise RuntimeError(
                f"seed {seed} did not produce all required task, latent, and training artifacts"
            )
        if self.execute:
            self.state["seeds"][str(seed)] = {
                "status": "complete",
                "result_dir": self._result_dir(seed),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()

    def aggregate(self) -> None:
        seeds = list(self.study["seeds"])
        if not all(self._seed_complete(seed) for seed in seeds):
            print("[summary] waiting for all three complete seeds", flush=True)
            return
        overall_all = []; dataset_all = []; class_all = []; resource_all = []
        validation_rows = []; latent_snapshots = []; latent_classes = []
        latent_probes = []; latent_probe_classes = []
        training_histories = []; training_logs = []
        for seed in seeds:
            result = self._result_dir(seed)
            overall = pd.read_csv(os.path.join(result, "Overall_Metrics.csv"))
            overall = overall[overall["origin"] == "ALL"].copy(); overall["seed"] = seed
            overall_all.append(overall)
            dataset = pd.read_csv(os.path.join(result, "Per_Dataset_Metrics.csv")); dataset["seed"] = seed
            dataset_all.append(dataset)
            per_class = pd.read_csv(os.path.join(result, "Per_Class_Metrics.csv"))
            native = per_class["is_native_class"]
            native = native if native.dtype == bool else native.astype(str).str.lower().eq("true")
            per_class = per_class[(per_class["origin"] == "ALL") & native].copy()
            per_class["seed"] = seed; class_all.append(per_class)
            resources = pd.read_csv(os.path.join(result, "Resource_Accounting.csv")); resources["seed"] = seed
            resource_all.append(resources)

            validation = pd.read_csv(os.path.join(result, "Validation_Overall_Metrics.csv")).iloc[0]
            validation_ds = pd.read_csv(os.path.join(result, "Validation_Per_Dataset_Metrics.csv"))
            validation_cls = pd.read_csv(os.path.join(result, "Validation_Per_Class_Metrics.csv"))
            rare = validation_cls[
                (validation_cls["support"] >= int(self.study["selection"]["rare_validation_min_support"]))
                & (validation_cls["support"] < int(self.study["selection"]["rare_validation_max_support"]))
            ]
            worst = validation_ds.sort_values("macro_f1").iloc[0]
            weakest = validation_cls[validation_cls["support"] > 0].sort_values("recall").iloc[0]
            validation_rows.append({
                "seed": seed,
                "validation_macro_f1": float(validation["macro_f1"]),
                "worst_dataset": str(worst["dataset"]),
                "worst_dataset_macro_f1": float(worst["macro_f1"]),
                "rare_classes": "|".join(rare["class"].astype(str)),
                "rare_class_count": int(len(rare)),
                "rare_recall_mean": float(rare["recall"].mean()) if len(rare) else np.nan,
                "weakest_class": str(weakest["class"]),
                "weakest_class_recall": float(weakest["recall"]),
            })

            latent_dir = os.path.join(result, "latent", "cumulative")
            for filename, target in (
                ("Latent_Snapshot_Metrics.csv", latent_snapshots),
                ("Latent_Per_Class.csv", latent_classes),
                ("Latent_Probe_Metrics.csv", latent_probes),
                ("Latent_Probe_Per_Class.csv", latent_probe_classes),
            ):
                frame = pd.read_csv(os.path.join(latent_dir, filename)); frame["seed"] = seed
                target.append(frame)
            training_dir = self._training_dir(seed)
            training_history = pd.read_csv(os.path.join(training_dir, "Training_History.csv"))
            training_history["seed"] = seed; training_histories.append(training_history)
            training_log = pd.read_csv(os.path.join(training_dir, "Training_Log_Manifest.csv"))
            training_log["seed"] = seed; training_logs.append(training_log)

        frames = {
            "overall": pd.concat(overall_all, ignore_index=True),
            "dataset": pd.concat(dataset_all, ignore_index=True),
            "class": pd.concat(class_all, ignore_index=True),
            "resource": pd.concat(resource_all, ignore_index=True),
            "guardrail": pd.DataFrame(validation_rows),
            "snapshot": pd.concat(latent_snapshots, ignore_index=True),
            "latent_class": pd.concat(latent_classes, ignore_index=True),
            "probe": pd.concat(latent_probes, ignore_index=True),
            "probe_class": pd.concat(latent_probe_classes, ignore_index=True),
            "training": pd.concat(training_histories, ignore_index=True),
            "training_logs": pd.concat(training_logs, ignore_index=True),
        }
        training_summary = summarize_training_history(frames["training"])
        outputs = {
            "Three_Seed_Overall.csv": frames["overall"],
            "Three_Seed_Overall_Summary.csv": _mean_sd(frames["overall"], []),
            "Three_Seed_Per_Dataset.csv": frames["dataset"],
            "Three_Seed_Per_Dataset_Summary.csv": _mean_sd(frames["dataset"], ["dataset"]),
            "Three_Seed_Per_Class.csv": frames["class"],
            "Three_Seed_Per_Class_Summary.csv": _mean_sd(frames["class"], ["class"]),
            "Three_Seed_Guardrails.csv": frames["guardrail"],
            "Three_Seed_Resource_Accounting.csv": frames["resource"],
            "Three_Seed_Selected_Epochs.csv": frames["resource"][[
                column for column in (
                    "seed", "Trial_ID", "A_selected_epoch", "B_selected_epoch",
                    "C_selected_epoch", "A_epochs_completed", "B_epochs_completed",
                    "C_epochs_completed",
                ) if column in frames["resource"].columns
            ]],
            "Latent_Snapshot_Metrics_3Seed.csv": frames["snapshot"],
            "Latent_Snapshot_Summary_3Seed.csv": _mean_sd(frames["snapshot"], ["snapshot", "stage", "encoder", "representation"]),
            "Latent_Per_Class_3Seed.csv": frames["latent_class"],
            "Latent_Per_Class_Summary_3Seed.csv": _mean_sd(frames["latent_class"], ["snapshot", "stage", "encoder", "class"]),
            "Latent_Probe_Metrics_3Seed.csv": frames["probe"],
            "Latent_Probe_Summary_3Seed.csv": _mean_sd(frames["probe"], ["snapshot", "stage", "encoder", "probe", "target"]),
            "Latent_Probe_Per_Class_3Seed.csv": frames["probe_class"],
            "Latent_Probe_Per_Class_Summary_3Seed.csv": _mean_sd(frames["probe_class"], ["snapshot", "stage", "encoder", "probe", "class"]),
            "Training_History_3Seed.csv": frames["training"],
            "Training_History_Summary_3Seed.csv": training_summary,
            "Training_Log_Manifest_3Seed.csv": frames["training_logs"],
        }
        for filename, frame in outputs.items():
            _atomic_csv(frame, os.path.join(self.summary_dir, filename))
        plot_inputs = {
            "snapshot_summary": outputs["Latent_Snapshot_Summary_3Seed.csv"],
            "probe_summary": outputs["Latent_Probe_Summary_3Seed.csv"],
            "probe_class_all": frames["probe_class"],
            "per_class_all": frames["latent_class"],
        }
        figures = _save_summary_plots(self.summary_dir, plot_inputs)
        training_figures = plot_cross_seed_training_history(
            training_summary, os.path.join(self.summary_dir, "training_figures")
        )
        manifest = {
            "format_version": 1,
            "protocol_hash": self.protocol_hash,
            "prefix": self.prefix,
            "seeds": seeds,
            "primary_metric": self.study["selection"]["primary_metric"],
            "result_dirs": {str(seed): self._result_dir(seed) for seed in seeds},
            "summary_files": sorted(outputs),
            "figure_files": figures,
            "training_figure_files": training_figures,
            "notebook_stream_log": os.path.join(
                self.summary_dir, "Notebook_Training_Stream.log"
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(manifest, os.path.join(self.summary_dir, "Final_Study_Manifest.json"))
        self.state["status"] = "complete"
        self.state["manifest"] = os.path.join(self.summary_dir, "Final_Study_Manifest.json")
        self._save_state()

    def run(self) -> None:
        print(
            f"[runner] mode={'EXECUTE' if self.execute else 'DRY_RUN'} prefix={self.prefix} "
            f"seeds={self.study['seeds']} summary_dir={self.summary_dir}",
            flush=True,
        )
        launched = 0
        for seed in self.study["seeds"]:
            print(f"[runner] seed {seed}: checking resumable artifacts", flush=True)
            incomplete = not self._seed_complete(seed)
            if self.execute and not incomplete:
                print(f"[runner] seed {seed}: all task and latent artifacts complete", flush=True)
                continue
            if self.execute and incomplete and self.max_new_seeds is not None and launched >= self.max_new_seeds:
                print(f"[runner] max-new-seeds={self.max_new_seeds} reached", flush=True)
                break
            self.run_seed(seed)
            if self.execute and incomplete:
                launched += 1
        if self.execute:
            self.aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--study-config", default="config/recommended_private_3seed.yaml")
    parser.add_argument("--prefix", default="nfv3_4way_recommended_private_bsupcon_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-seeds", type=int)
    args = parser.parse_args()
    runner = RecommendedPrivateStudy(
        base_config_path=args.config,
        study_config_path=args.study_config,
        prefix=args.prefix,
        execute=args.execute,
        max_new_seeds=args.max_new_seeds,
    )
    runner.run()


if __name__ == "__main__":
    main()
