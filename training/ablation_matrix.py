"""Resumable orchestrator for the finalized greedy MoE experiment suite.

Scientific settings live in ``config/greedy_moe_study.yaml``.  This module
turns that immutable protocol into Stage-A jobs plus evaluated Stage-B/C runs,
selects every winner from validation macro-F1, and records the decisions in an
atomic JSON state file.  Running without ``--execute`` is a side-effect-free
preview of the jobs that are currently unblocked.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .config import load_config


PHASES = ("0", "1", "2", "3a", "3b", "3c", "4")
STATE_FORMAT_VERSION = 1
STAGE_A_FILE = "stage_a_encoder.pt"
STAGE_C_FILE = "stage_c_full.pt"
STAGE_C_SUMMARY_FILE = "stage_c_training_summary.json"
T_975_DF2 = 4.302652729911275


@dataclass(frozen=True)
class RunSpec:
    phase: str
    condition: str
    seed: int
    architecture: str
    expert_hidden_dims: tuple[int, ...]
    representation_objective: str = "ce"
    representation_weight: float = 0.1
    adapter_rank: int = 16
    weight_decay: float = 0.0

    @property
    def identity_condition(self) -> str:
        if self.weight_decay == 0.0:
            return self.condition
        suffix = f"_wd{_weight_decay_label(self.weight_decay)}"
        return self.condition if suffix in self.condition else self.condition + suffix

    @property
    def run_id(self) -> str:
        return f"{self.identity_condition}:seed{self.seed}"

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["expert_hidden_dims"] = list(self.expert_hidden_dims)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunSpec":
        restored = dict(value)
        restored["expert_hidden_dims"] = tuple(restored["expert_hidden_dims"])
        return cls(**restored)


def load_study(path: str) -> dict[str, Any]:
    with open(path) as handle:
        study = yaml.safe_load(handle)
    if study.get("format_version") != 1:
        raise ValueError("greedy study config must use format_version: 1")
    if study.get("seeds") != [0, 1, 2]:
        raise ValueError("the finalized study requires paired seeds [0, 1, 2]")
    if study.get("selection", {}).get("metric") != "best_validation_macro_f1":
        raise ValueError("winner selection must use best_validation_macro_f1")
    weight_decays = study.get(
        "capacity_weight_decays",
        [study["backbone"]["training"].get("weight_decay", 0.0)],
    )
    if not weight_decays or any(float(value) < 0 for value in weight_decays):
        raise ValueError("capacity_weight_decays must contain non-negative values")
    latent = study.get("latent_reporting", {})
    if latent:
        if latent.get("split", "val") != "val":
            raise ValueError("greedy-study latent reporting must use the validation split")
        if not latent.get("required_artifacts"):
            raise ValueError("latent_reporting.required_artifacts must not be empty")
    return study


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def protocol_hash(study: dict[str, Any], base_config_path: str) -> str:
    with open(base_config_path, "rb") as handle:
        base_hash = hashlib.sha256(handle.read()).hexdigest()
    return _canonical_hash({"study": study, "base_config_sha256": base_hash})


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_format_override(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)


def flatten_overrides(node: dict[str, Any], prefix: str = "") -> list[str]:
    result: list[str] = []
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.extend(flatten_overrides(value, dotted))
        else:
            result.append(f"{dotted}={_format_override(value)}")
    return result


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_csv(path: str, frame: pd.DataFrame, *, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def _sample_sd(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.std(items, ddof=1)) if len(items) > 1 else 0.0


def _capacity_label(hidden_dims: Iterable[int]) -> str:
    values = list(hidden_dims)
    return "linear" if not values else "x".join(str(value) for value in values)


def _weight_label(weight: float) -> str:
    return format(float(weight), "g").replace(".", "p")


def _weight_decay_label(weight_decay: float) -> str:
    return format(float(weight_decay), "g").replace(".", "p").replace("-", "m")


def _expert_parameter_count(hidden_dims: Iterable[int], latent_dim: int = 64, classes: int = 22) -> int:
    dims = [latent_dim, *list(hidden_dims), classes]
    return sum((left + 1) * right for left, right in zip(dims[:-1], dims[1:]))


class GreedyStudyRunner:
    def __init__(
        self,
        *,
        base_config_path: str,
        study_config_path: str,
        prefix: str,
        summary_dir: str | None = None,
        execute: bool = False,
        max_new_runs: int | None = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.study_config_path = study_config_path
        self.prefix = prefix
        self.execute = execute
        self.max_new_runs = max_new_runs
        self.new_runs = 0
        self.study = load_study(study_config_path)
        base = load_config(base_config_path)
        self.summary_dir = summary_dir or os.path.join(
            base["OUTPUT_DIR"], "results", f"{prefix}_summary"
        )
        self.state_path = os.path.join(self.summary_dir, "study_state.json")
        self.protocol_hash = protocol_hash(self.study, base_config_path)
        self.state = self._load_state()

    @property
    def tolerance(self) -> float:
        return float(self.study["selection"].get("exact_tolerance", 1e-12))

    @property
    def seeds(self) -> list[int]:
        return [int(value) for value in self.study["seeds"]]

    @property
    def capacity_weight_decays(self) -> list[float]:
        return [
            float(value) for value in self.study.get(
                "capacity_weight_decays",
                [self.study["backbone"]["training"].get("weight_decay", 0.0)],
            )
        ]

    @property
    def latent_policy(self) -> dict[str, Any]:
        return self.study.get("latent_reporting", {})

    @property
    def latent_enabled(self) -> bool:
        return bool(self.latent_policy.get("enabled", False))

    def _load_state(self) -> dict[str, Any]:
        if not os.path.isfile(self.state_path):
            return {
                "format_version": STATE_FORMAT_VERSION,
                "protocol_hash": self.protocol_hash,
                "prefix": self.prefix,
                "study_config": os.path.abspath(self.study_config_path),
                "base_config": os.path.abspath(self.base_config_path),
                "runs": {},
                "decisions": {},
                "phases": {},
            }
        with open(self.state_path) as handle:
            state = json.load(handle)
        if state.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"Unsupported study-state format in {self.state_path}")
        if state.get("protocol_hash") != self.protocol_hash:
            raise ValueError(
                "Study protocol changed after execution began. Choose a new --prefix or restore "
                f"the original study/base config. State: {self.state_path}"
            )
        return state

    def _save_state(self) -> None:
        if self.execute:
            _atomic_json(self.state_path, self.state)

    def _base_nested(self) -> dict[str, Any]:
        return copy.deepcopy(self.study["backbone"])

    def _condition_nested(self, spec: RunSpec) -> dict[str, Any]:
        nested = self._base_nested()
        nested["seed"] = spec.seed
        nested["architecture"] = spec.architecture
        nested["data"]["split_seed"] = spec.seed
        nested["model"]["expert"]["hidden_dims"] = list(spec.expert_hidden_dims)
        nested["model"]["adapter"]["rank"] = spec.adapter_rank
        nested["training"]["weight_decay"] = spec.weight_decay
        representation = nested["training"]["representation"]
        representation["objective"] = spec.representation_objective
        representation["weight"] = spec.representation_weight
        representation["sampling"] = "legacy"
        representation["class_weighting"] = "legacy"
        return nested

    def _stage_a_name(self, spec: RunSpec) -> str:
        objective = spec.representation_objective
        weight = _weight_label(spec.representation_weight)
        decay = "" if spec.weight_decay == 0.0 else f"_wd{_weight_decay_label(spec.weight_decay)}"
        return f"{self.prefix}_stagea_{objective}_w{weight}{decay}_seed{spec.seed}_v1"

    def _run_name(self, spec: RunSpec) -> str:
        return f"{self.prefix}_{spec.identity_condition}_seed{spec.seed}_v1"

    def _command(self, overrides: list[str], *, latent_only: bool = False) -> list[str]:
        command = [
            sys.executable, "-u", "-m", "training.ooc_run",
            "--config", self.base_config_path,
        ]
        if latent_only:
            command.append("--latent-only")
        for override in overrides:
            command.extend(["--set", override])
        return command

    def _stage_a_overrides(self, spec: RunSpec) -> list[str]:
        nested = self._condition_nested(spec)
        # Stage A is architecture/capacity independent.  Canonicalizing these
        # irrelevant fields guarantees reuse across phases.
        nested["architecture"] = "moe_dataset_soft"
        nested["model"]["expert"]["hidden_dims"] = []
        nested["training"]["stages"] = ["A"]
        nested["training"]["run_final_evaluation"] = False
        nested["run_name"] = self._stage_a_name(spec)
        return flatten_overrides(nested)

    def _stage_a_path(self, spec: RunSpec) -> str:
        config = load_config(self.base_config_path, self._stage_a_overrides(spec))
        return os.path.join(config["training"]["checkpoint_dir"], STAGE_A_FILE)

    def _default_latent_plot(self, spec: RunSpec) -> bool:
        threshold = int(self.latent_policy.get("plot_seed_zero_after_phase", 99))
        phase_number = int(spec.phase[0]) if spec.phase and spec.phase[0].isdigit() else -1
        return spec.seed == 0 and phase_number > threshold

    def _run_overrides(
        self,
        spec: RunSpec,
        *,
        latent_stages: Iterable[str] | None = None,
        latent_plot: bool | None = None,
    ) -> list[str]:
        nested = self._condition_nested(spec)
        nested["run_name"] = self._run_name(spec)
        nested["training"]["stages"] = ["B", "C"]
        nested["training"]["run_final_evaluation"] = True
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(spec)
        if self.latent_enabled:
            latent = nested.setdefault("evaluation", {}).setdefault("latent", {})
            latent["enabled"] = True
            # Fixed (not seed-derived) so conditions sharing a split use the
            # same stratified rows and remain numerically comparable.
            latent["random_seed"] = int(self.latent_policy.get("random_seed", 0))
            latent["split"] = self.latent_policy.get("split", "val")
            latent["stages"] = list(latent_stages or self.latent_policy.get("stages", ["C"]))
            latent["plot"] = (
                self._default_latent_plot(spec) if latent_plot is None else bool(latent_plot)
            )
            latent["plot_method"] = self.latent_policy.get("plot_method", "umap")
        return flatten_overrides(nested)

    def _resolved(self, spec: RunSpec) -> dict[str, Any]:
        return load_config(self.base_config_path, self._run_overrides(spec))

    def _paths(self, spec: RunSpec) -> tuple[str, str]:
        config = self._resolved(spec)
        return config["training"]["checkpoint_dir"], config["evaluation"]["output_dir"]

    def _missing_primary_artifacts(self, spec: RunSpec) -> list[str]:
        checkpoint_dir, result_dir = self._paths(spec)
        missing = [
            filename for filename in self.study["required_artifacts"]
            if not os.path.isfile(os.path.join(result_dir, filename))
        ]
        for filename in (STAGE_C_FILE, STAGE_C_SUMMARY_FILE):
            if not os.path.isfile(os.path.join(checkpoint_dir, filename)):
                missing.append(f"checkpoints/{filename}")
        return missing

    def _missing_latent_artifacts(
        self,
        spec: RunSpec,
        *,
        stages: Iterable[str] | None = None,
        plot: bool | None = None,
    ) -> list[str]:
        if not self.latent_enabled:
            return []
        requested_stages = [
            str(stage).upper()
            for stage in (stages or self.latent_policy.get("stages", ["C"]))
        ]
        requested_plot = self._default_latent_plot(spec) if plot is None else bool(plot)
        latent_dir = self._latent_dir(spec)
        missing = [
            f"latent/{filename}"
            for filename in self.latent_policy["required_artifacts"]
            if not os.path.isfile(os.path.join(latent_dir, filename))
        ]
        config_path = os.path.join(latent_dir, "Latent_Report_Config.json")
        report_config = None
        if os.path.isfile(config_path):
            try:
                with open(config_path) as handle:
                    report_config = json.load(handle)
            except (OSError, ValueError):
                missing.append("latent/Latent_Report_Config.json:invalid")
        stored_stages = set(report_config.get("stages", [])) if report_config is not None else set()
        if report_config is not None and not set(requested_stages).issubset(stored_stages):
            missing.append(
                "latent/Latent_Report_Config.json:stages=" + json.dumps(requested_stages)
            )
        if requested_plot and report_config is not None:
            method = self.latent_policy.get("plot_method", "umap")
            figure_dir = os.path.join(latent_dir, "latent_figures")
            for snapshot_id in report_config.get("snapshots", []):
                filename = f"{snapshot_id.replace('::', '_')}__{method}.png"
                if not os.path.isfile(os.path.join(figure_dir, filename)):
                    missing.append(f"latent/latent_figures/{filename}")
        return list(dict.fromkeys(missing))

    def _missing_artifacts(self, spec: RunSpec) -> list[str]:
        return [
            *self._missing_primary_artifacts(spec),
            *self._missing_latent_artifacts(spec),
        ]

    def _is_complete(self, spec: RunSpec) -> bool:
        return not self._missing_artifacts(spec)

    def _validation_score(self, spec: RunSpec) -> float:
        checkpoint_dir, _ = self._paths(spec)
        path = os.path.join(checkpoint_dir, STAGE_C_SUMMARY_FILE)
        with open(path) as handle:
            summary = json.load(handle)
        if summary.get("selection_mode") != "best_val":
            raise ValueError(f"{spec.run_id} did not use validation checkpoint selection")
        score = summary.get("best_validation_macro_f1")
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"{spec.run_id} has no finite validation macro-F1")
        return float(score)

    def _overall_row(self, spec: RunSpec) -> dict[str, Any]:
        _, result_dir = self._paths(spec)
        frame = pd.read_csv(os.path.join(result_dir, "Overall_Metrics.csv"))
        if "origin" in frame.columns:
            frame = frame[frame["origin"] == "ALL"]
        if len(frame) != 1:
            raise ValueError(f"Expected one overall row for {spec.run_id}; found {len(frame)}")
        return frame.iloc[0].to_dict()

    def _record_complete(self, spec: RunSpec) -> None:
        checkpoint_dir, result_dir = self._paths(spec)
        record = {
            "spec": spec.serializable(),
            "run_name": self._run_name(spec),
            "checkpoint_dir": checkpoint_dir,
            "result_dir": result_dir,
            "stage_a_checkpoint": self._stage_a_path(spec),
            "validation_macro_f1": self._validation_score(spec),
            "overall_test_metrics": self._overall_row(spec),
            "status": "complete",
        }
        self.state["runs"][spec.run_id] = record
        self._save_state()
        self._harvest_latent_metrics(spec)

    _LATENT_METRIC_FILES = {
        "snapshot": (
            "Latent_Snapshot_Metrics.csv",
            "Latent_Snapshot_Metrics_By_Run.csv",
            ["condition", "architecture", "snapshot", "stage", "encoder", "representation", "equivalent_to"],
        ),
        "per_class": (
            "Latent_Per_Class.csv",
            "Latent_Per_Class_By_Run.csv",
            ["condition", "architecture", "snapshot", "stage", "encoder", "class"],
        ),
        "probe": (
            "Latent_Probe_Metrics.csv",
            "Latent_Probe_Metrics_By_Run.csv",
            ["condition", "architecture", "snapshot", "stage", "encoder", "probe", "target"],
        ),
        "probe_per_class": (
            "Latent_Probe_Per_Class.csv",
            "Latent_Probe_Per_Class_By_Run.csv",
            ["condition", "architecture", "snapshot", "stage", "encoder", "probe", "class"],
        ),
    }

    def _latent_dir(self, spec: RunSpec) -> str:
        _, result_dir = self._paths(spec)
        return os.path.join(result_dir, "latent")

    def _harvest_latent_metrics(self, spec: RunSpec) -> None:
        """Upsert every latent metric table into run-keyed study summaries."""
        latent_dir = self._latent_dir(spec)
        if not os.path.isfile(os.path.join(latent_dir, "Latent_Report_Config.json")):
            return
        identity = {
            "run_id": spec.run_id, "phase": spec.phase, "condition": spec.condition,
            "seed": spec.seed, "architecture": spec.architecture,
            "latent_figures_dir": os.path.join(latent_dir, "latent_figures"),
        }
        for kind, (filename, summary_filename, group_columns) in self._LATENT_METRIC_FILES.items():
            path = os.path.join(latent_dir, filename)
            if not os.path.isfile(path):
                continue
            frame = pd.read_csv(path)
            for key, value in reversed(list(identity.items())):
                frame.insert(0, key, value)
            summary_path = os.path.join(self.summary_dir, summary_filename)
            combined = frame
            if os.path.isfile(summary_path):
                existing = pd.read_csv(summary_path)
                existing = existing[existing["run_id"] != spec.run_id]
                combined = pd.concat([existing, frame], ignore_index=True)
            _atomic_csv(summary_path, combined)
            self._write_latent_aggregate(combined, kind, group_columns)

    def _write_latent_aggregate(
        self,
        frame: pd.DataFrame,
        kind: str,
        group_columns: list[str],
        *,
        output_prefix: str = "Latent",
    ) -> str:
        available_groups = [column for column in group_columns if column in frame.columns]
        excluded = {
            "seed", "split_seed", "run_id", "phase", "latent_figures_dir",
            *available_groups,
        }
        numeric = [
            column for column in frame.select_dtypes(include=[np.number]).columns
            if column not in excluded
        ]
        rows = []
        grouped = (
            frame.groupby(available_groups, dropna=False, sort=False)
            if available_groups else [((), frame)]
        )
        for key, local in grouped:
            values = key if isinstance(key, tuple) else (key,)
            identity = dict(zip(available_groups, values))
            for metric in numeric:
                observed = pd.to_numeric(local[metric], errors="coerce").dropna()
                if observed.empty:
                    continue
                rows.append({
                    **identity,
                    "metric": metric,
                    "n_seeds": int(local.loc[observed.index, "seed"].nunique()),
                    "mean": float(observed.mean()),
                    "sd": _sample_sd(observed),
                })
        output = os.path.join(
            self.summary_dir, f"{output_prefix}_{kind.title()}_Summary.csv"
        )
        _atomic_csv(output, pd.DataFrame(rows))
        return output

    def _print_job(self, spec: RunSpec) -> None:
        primary_missing = self._missing_primary_artifacts(spec)
        latent_missing = self._missing_latent_artifacts(spec)
        missing = [*primary_missing, *latent_missing]
        status = "complete" if not missing else "pending: " + ", ".join(missing)
        print(f"[{spec.phase}:{spec.run_id}] {status}")
        if primary_missing:
            if not os.path.isfile(self._stage_a_path(spec)):
                print("  Stage A:", shlex.join(self._command(self._stage_a_overrides(spec))))
            print("  Run:", shlex.join(self._command(self._run_overrides(spec))))
        elif latent_missing:
            print(
                "  Latent:",
                shlex.join(self._command(self._run_overrides(spec), latent_only=True)),
            )

    def _budget_available(self) -> bool:
        return self.max_new_runs is None or self.new_runs < self.max_new_runs

    def ensure_specs(self, specs: list[RunSpec]) -> None:
        for spec in specs:
            self._print_job(spec)
            if self._is_complete(spec):
                if self.execute:
                    self._record_complete(spec)
                continue
            primary_missing = self._missing_primary_artifacts(spec)
            if not self.execute or (primary_missing and not self._budget_available()):
                continue
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if primary_missing:
                if not os.path.isfile(self._stage_a_path(spec)):
                    subprocess.run(self._command(self._stage_a_overrides(spec)), check=True, env=env)
                subprocess.run(self._command(self._run_overrides(spec)), check=True, env=env)
            else:
                subprocess.run(
                    self._command(self._run_overrides(spec), latent_only=True),
                    check=True,
                    env=env,
                )
            missing = self._missing_artifacts(spec)
            if missing:
                raise RuntimeError(f"{spec.run_id} finished without required artifacts: {missing}")
            if primary_missing:
                self.new_runs += 1
            self._record_complete(spec)

    def ensure_latent_specs(
        self,
        specs: Iterable[RunSpec],
        *,
        stages: Iterable[str],
        plot: bool,
    ) -> bool:
        """Upgrade existing runs to an explicit latent policy without retraining."""
        unique = {spec.run_id: spec for spec in specs}
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        all_complete = True
        for spec in unique.values():
            missing = self._missing_latent_artifacts(spec, stages=stages, plot=plot)
            if not missing:
                if self.execute:
                    self._harvest_latent_metrics(spec)
                continue
            all_complete = False
            command = self._command(
                self._run_overrides(spec, latent_stages=stages, latent_plot=plot),
                latent_only=True,
            )
            print(f"[latent-upgrade:{spec.run_id}] pending: {', '.join(missing)}")
            print("  Latent:", shlex.join(command))
            if not self.execute:
                continue
            subprocess.run(command, check=True, env=env)
            remaining = self._missing_latent_artifacts(spec, stages=stages, plot=plot)
            if remaining:
                raise RuntimeError(
                    f"{spec.run_id} latent upgrade finished without required artifacts: {remaining}"
                )
            self._harvest_latent_metrics(spec)
        return all_complete or self.execute

    def _all_complete(self, specs: Iterable[RunSpec]) -> bool:
        return all(self._is_complete(spec) for spec in specs)

    def _require_decision(self, name: str, earlier_phase: str) -> dict[str, Any]:
        decision = self.state["decisions"].get(name)
        if decision is None:
            raise RuntimeError(
                f"Phase prerequisite missing: complete phase {earlier_phase} before this phase "
                f"(no {name!r} decision in {self.state_path})."
            )
        return decision

    def _specs_from_ids(self, run_ids: Iterable[str]) -> list[RunSpec]:
        specs = []
        for run_id in run_ids:
            record = self.state["runs"].get(run_id)
            if not record:
                raise RuntimeError(f"Study state has no completed run {run_id!r}")
            specs.append(RunSpec.from_dict(record["spec"]))
        return specs

    def _mean_validation(self, specs: Iterable[RunSpec]) -> float:
        return float(np.mean([self._validation_score(spec) for spec in specs]))

    def _best_by_score(self, specs: Iterable[RunSpec], *, tie_key) -> RunSpec:
        ordered = sorted(specs, key=tie_key)
        best = ordered[0]
        best_score = self._validation_score(best)
        for candidate in ordered[1:]:
            score = self._validation_score(candidate)
            if score > best_score + self.tolerance:
                best, best_score = candidate, score
        return best

    def _write_phase_summary(self, phase: str, specs: list[RunSpec]) -> None:
        rows = []
        for spec in specs:
            overall = self._overall_row(spec)
            rows.append({
                "phase": phase,
                "run_id": spec.run_id,
                "condition": spec.condition,
                "seed": spec.seed,
                "split_seed": spec.seed,
                "architecture": spec.architecture,
                "expert_hidden_dims": json.dumps(list(spec.expert_hidden_dims)),
                "representation_objective": spec.representation_objective,
                "representation_weight": spec.representation_weight,
                "weight_decay": spec.weight_decay,
                "validation_macro_f1__selection_metric": self._validation_score(spec),
                "test_macro_f1__reporting_only": float(overall["macro_f1"]),
                "test_accuracy__reporting_only": float(overall["accuracy"]),
                "test_weighted_f1__reporting_only": float(overall["weighted_f1"]),
                "run_name": self._run_name(spec),
            })
        runs = pd.DataFrame(rows)
        grouped_rows = []
        for condition, frame in runs.groupby("condition", sort=False):
            grouped_rows.append({
                "phase": phase,
                "condition": condition,
                "n_seeds": len(frame),
                "weight_decay": frame["weight_decay"].iloc[0],
                "validation_macro_f1_mean__selection_metric": frame["validation_macro_f1__selection_metric"].mean(),
                "validation_macro_f1_sd": frame["validation_macro_f1__selection_metric"].std(ddof=1) if len(frame) > 1 else 0.0,
                "test_macro_f1_mean__reporting_only": frame["test_macro_f1__reporting_only"].mean(),
                "test_macro_f1_sd__reporting_only": frame["test_macro_f1__reporting_only"].std(ddof=1) if len(frame) > 1 else 0.0,
            })
        _atomic_csv(os.path.join(self.summary_dir, f"Ablation_Phase{phase}_Runs.csv"), runs)
        _atomic_csv(
            os.path.join(self.summary_dir, f"Ablation_Phase{phase}_Summary.csv"),
            pd.DataFrame(grouped_rows),
        )

    def baseline_specs(self) -> list[RunSpec]:
        weight_decay = float(self.study["backbone"]["training"].get("weight_decay", 0.0))
        return [
            RunSpec(
                "0", "baseline_linear", seed, "moe_dataset_soft", (),
                weight_decay=weight_decay,
            )
            for seed in self.seeds
        ]

    def capacity_coarse_specs(self) -> list[RunSpec]:
        baseline_weight_decay = self.baseline_specs()[0].weight_decay
        distinguish_decay = len(self.capacity_weight_decays) > 1
        specs = []
        for values in self.study["capacity_candidates"]:
            hidden = tuple(values)
            for weight_decay in self.capacity_weight_decays:
                if not hidden and weight_decay == baseline_weight_decay:
                    continue
                condition = f"capacity_{_capacity_label(values)}"
                if distinguish_decay:
                    condition += f"_wd{_weight_decay_label(weight_decay)}"
                specs.append(
                    RunSpec(
                        "1", condition, 0, "moe_dataset_soft", hidden,
                        weight_decay=weight_decay,
                    )
                )
        return specs

    def architecture_specs(self) -> list[RunSpec]:
        capacity = self._require_decision("capacity", "1")
        hidden = tuple(capacity["expert_hidden_dims"])
        weight_decay = float(capacity.get("weight_decay", 0.0))
        values = []
        for seed in self.seeds:
            values.append(RunSpec(
                "2", "private_encoders", seed, "moe_dataset_private_encoders", hidden,
                weight_decay=weight_decay,
            ))
            values.append(RunSpec(
                "2", "adapters_r16", seed, "moe_dataset_adapters", hidden,
                adapter_rank=16, weight_decay=weight_decay,
            ))
        return values

    def supcon_sweep_specs(self) -> list[RunSpec]:
        architecture = self._require_decision("architecture", "2")
        hidden = tuple(architecture["expert_hidden_dims"])
        return [
            RunSpec(
                "3a", f"supcon_w{_weight_label(float(weight))}", 0,
                architecture["architecture"], hidden, "supcon", float(weight),
                int(architecture.get("adapter_rank", 16)),
                float(architecture.get("weight_decay", 0.0)),
            )
            for weight in self.study["supcon_weights"]
        ]

    def run_phase_0(self) -> None:
        specs = self.baseline_specs()
        self.ensure_specs(specs)
        if self._all_complete(specs):
            self.state["decisions"]["baseline"] = {
                "weight_decay": specs[0].weight_decay,
                "run_ids": [spec.run_id for spec in specs],
            }
            self.state["phases"]["0"] = {"status": "complete"}
            self._write_phase_summary("0", specs)
            self._save_state()

    def run_phase_1(self) -> None:
        self._require_decision("baseline", "0")
        coarse = self.capacity_coarse_specs()
        self.ensure_specs(coarse)
        if not self._all_complete(coarse):
            print("[phase 1] confirmations remain blocked until every seed-0 capacity candidate completes")
            return
        winner = self._best_by_score(
            coarse,
            tie_key=lambda spec: (
                _expert_parameter_count(spec.expert_hidden_dims), spec.weight_decay,
            ),
        )
        self.state["decisions"]["capacity_coarse"] = {
            "expert_hidden_dims": list(winner.expert_hidden_dims),
            "weight_decay": winner.weight_decay,
            "seed0_run_id": winner.run_id,
            "validation_macro_f1": self._validation_score(winner),
        }
        confirmations = [
            RunSpec(
                "1", winner.condition, seed, winner.architecture, winner.expert_hidden_dims,
                weight_decay=winner.weight_decay,
            )
            for seed in self.seeds[1:]
        ]
        self.ensure_specs(confirmations)
        candidate = [winner, *confirmations]
        if not self._all_complete(candidate):
            return
        baseline = self.baseline_specs()
        candidate_mean = self._mean_validation(candidate)
        baseline_mean = self._mean_validation(baseline)
        selected = candidate if candidate_mean > baseline_mean + self.tolerance else baseline
        selected_hidden = list(winner.expert_hidden_dims) if selected is candidate else []
        selected_weight_decay = winner.weight_decay if selected is candidate else baseline[0].weight_decay
        self.state["decisions"]["capacity"] = {
            "expert_hidden_dims": selected_hidden,
            "weight_decay": selected_weight_decay,
            "condition": selected[0].condition,
            "run_ids": [spec.run_id for spec in selected],
            "validation_macro_f1_mean": self._mean_validation(selected),
            "confirmed_candidate_hidden_dims": list(winner.expert_hidden_dims),
            "confirmed_candidate_weight_decay": winner.weight_decay,
            "candidate_validation_mean": candidate_mean,
            "baseline_validation_mean": baseline_mean,
        }
        self.state["phases"]["1"] = {"status": "complete"}
        self._write_phase_summary("1", [*baseline, *coarse, *confirmations])
        self._save_state()

    def run_phase_2(self) -> None:
        capacity = self._require_decision("capacity", "1")
        candidates = self.architecture_specs()
        self.ensure_specs(candidates)
        if not self._all_complete(candidates):
            return
        incumbent = self._specs_from_ids(capacity["run_ids"])
        private = [spec for spec in candidates if spec.architecture == "moe_dataset_private_encoders"]
        adapters = [spec for spec in candidates if spec.architecture == "moe_dataset_adapters"]
        choices = [("shared", incumbent), ("private", private), ("adapters", adapters)]
        best_name, best_specs = choices[0]
        best_mean = self._mean_validation(best_specs)
        for name, specs in choices[1:]:
            score = self._mean_validation(specs)
            if score > best_mean + self.tolerance:
                best_name, best_specs, best_mean = name, specs, score
        self.state["decisions"]["architecture"] = {
            "choice": best_name,
            "architecture": best_specs[0].architecture,
            "expert_hidden_dims": list(best_specs[0].expert_hidden_dims),
            "adapter_rank": best_specs[0].adapter_rank,
            "weight_decay": best_specs[0].weight_decay,
            "run_ids": [spec.run_id for spec in best_specs],
            "validation_macro_f1_mean": best_mean,
            "candidate_means": {name: self._mean_validation(specs) for name, specs in choices},
        }
        self.state["phases"]["2"] = {"status": "complete"}
        self._write_phase_summary("2", [*incumbent, *candidates])
        self._save_state()

    def run_phase_3a(self) -> None:
        self._require_decision("architecture", "2")
        specs = self.supcon_sweep_specs()
        self.ensure_specs(specs)
        if not self._all_complete(specs):
            return
        winner = self._best_by_score(specs, tie_key=lambda spec: spec.representation_weight)
        self.state["decisions"]["supcon_weight"] = {
            "weight": winner.representation_weight,
            "seed0_run_id": winner.run_id,
            "validation_macro_f1": self._validation_score(winner),
        }
        self.state["phases"]["3a"] = {"status": "complete"}
        self._write_phase_summary("3a", specs)
        self._save_state()

    def _confirmed_supcon_specs(self) -> list[RunSpec]:
        architecture = self._require_decision("architecture", "2")
        selected = self._require_decision("supcon_weight", "3a")
        seed0 = self._specs_from_ids([selected["seed0_run_id"]])[0]
        return [seed0, *[
            RunSpec(
                "3b", seed0.condition, seed, architecture["architecture"],
                tuple(architecture["expert_hidden_dims"]), "supcon", float(selected["weight"]),
                int(architecture.get("adapter_rank", 16)),
                float(architecture.get("weight_decay", 0.0)),
            )
            for seed in self.seeds[1:]
        ]]

    def run_phase_3b(self) -> None:
        specs = self._confirmed_supcon_specs()
        self.ensure_specs(specs[1:])
        if not self._all_complete(specs):
            return
        architecture = self._require_decision("architecture", "2")
        incumbent = self._specs_from_ids(architecture["run_ids"])
        supcon_mean = self._mean_validation(specs)
        ce_mean = self._mean_validation(incumbent)
        selected = specs if supcon_mean > ce_mean + self.tolerance else incumbent
        objective = "supcon" if selected is specs else "ce"
        self.state["decisions"]["representation_preliminary"] = {
            "objective": objective,
            "weight": float(self.state["decisions"]["supcon_weight"]["weight"]),
            "weight_decay": float(architecture.get("weight_decay", 0.0)),
            "run_ids": [spec.run_id for spec in selected],
            "ce_validation_mean": ce_mean,
            "supcon_validation_mean": supcon_mean,
        }
        self.state["decisions"]["supcon_confirmed"] = {
            "run_ids": [spec.run_id for spec in specs],
            "validation_macro_f1_mean": supcon_mean,
        }
        self.state["phases"]["3b"] = {"status": "complete"}
        self._write_phase_summary("3b", [*incumbent, *specs])
        self._save_state()

    def _balanced_specs(self) -> list[RunSpec]:
        architecture = self._require_decision("architecture", "2")
        selected = self._require_decision("supcon_weight", "3a")
        weight = float(selected["weight"])
        return [
            RunSpec(
                "3c", f"balanced_supcon_w{_weight_label(weight)}", seed,
                architecture["architecture"], tuple(architecture["expert_hidden_dims"]),
                "balanced_supcon", weight, int(architecture.get("adapter_rank", 16)),
                float(architecture.get("weight_decay", 0.0)),
            )
            for seed in self.seeds
        ]

    def run_phase_3c(self) -> None:
        preliminary = self._require_decision("representation_preliminary", "3b")
        balanced = self._balanced_specs()
        self.ensure_specs(balanced)
        if not self._all_complete(balanced):
            return
        architecture = self._require_decision("architecture", "2")
        ce = self._specs_from_ids(architecture["run_ids"])
        supcon = self._specs_from_ids(self.state["decisions"]["supcon_confirmed"]["run_ids"])
        order = self.study["selection"]["representation_tie_order"]
        choices = {"ce": ce, "supcon": supcon, "balanced_supcon": balanced}
        winner = order[0]
        winner_mean = self._mean_validation(choices[winner])
        for objective in order[1:]:
            score = self._mean_validation(choices[objective])
            if score > winner_mean + self.tolerance:
                winner, winner_mean = objective, score
        self.state["decisions"]["representation"] = {
            "objective": winner,
            "weight": float(self.state["decisions"]["supcon_weight"]["weight"]),
            "weight_decay": float(architecture.get("weight_decay", 0.0)),
            "run_ids": [spec.run_id for spec in choices[winner]],
            "validation_macro_f1_mean": winner_mean,
            "candidate_means": {name: self._mean_validation(specs) for name, specs in choices.items()},
            "preliminary_objective": preliminary["objective"],
        }
        self.state["phases"]["3c"] = {"status": "complete"}
        self._write_phase_summary("3c", [*ce, *supcon, *balanced])
        self._save_state()

    def _paired_metric_summary(
        self,
        baseline: pd.DataFrame,
        winner: pd.DataFrame,
        *,
        key_columns: list[str],
        metrics: list[str],
    ) -> pd.DataFrame:
        left = baseline.set_index([*key_columns, "seed"])
        right = winner.set_index([*key_columns, "seed"])
        rows = []
        keys = sorted(set(left.index.droplevel("seed"))) if key_columns else [()]
        for key in keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            selector = key_tuple if key_columns else slice(None)
            left_group = left.loc[selector]
            right_group = right.loc[selector]
            if isinstance(left_group, pd.Series):
                left_group = left_group.to_frame().T
            if isinstance(right_group, pd.Series):
                right_group = right_group.to_frame().T
            left_group = left_group.sort_index()
            right_group = right_group.sort_index()
            for metric in metrics:
                base_values = left_group[metric].astype(float).to_numpy()
                winner_values = right_group[metric].astype(float).to_numpy()
                delta = winner_values - base_values
                delta_sd = _sample_sd(delta)
                half_width = T_975_DF2 * delta_sd / math.sqrt(len(delta)) if len(delta) > 1 else 0.0
                row = {column: value for column, value in zip(key_columns, key_tuple)}
                row.update({
                    "metric": metric,
                    "n_seeds": len(delta),
                    "baseline_mean": float(base_values.mean()),
                    "baseline_sd": _sample_sd(base_values),
                    "winner_mean": float(winner_values.mean()),
                    "winner_sd": _sample_sd(winner_values),
                    "paired_delta_mean": float(delta.mean()),
                    "paired_delta_sd": delta_sd,
                    "paired_delta_ci95_low": float(delta.mean() - half_width),
                    "paired_delta_ci95_high": float(delta.mean() + half_width),
                    "interval_note": "paired Student-t interval; descriptive only for n=3",
                })
                rows.append(row)
        return pd.DataFrame(rows)

    def _load_seed_frames(self, specs: list[RunSpec], filename: str) -> pd.DataFrame:
        frames = []
        for spec in specs:
            _, result_dir = self._paths(spec)
            frame = pd.read_csv(os.path.join(result_dir, filename))
            frame["seed"] = spec.seed
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _write_final_report(self, baseline: list[RunSpec], winner: list[RunSpec]) -> None:
        overall_base = self._load_seed_frames(baseline, "Overall_Metrics.csv")
        overall_win = self._load_seed_frames(winner, "Overall_Metrics.csv")
        if "origin" in overall_base.columns:
            overall_base = overall_base[overall_base["origin"] == "ALL"]
            overall_win = overall_win[overall_win["origin"] == "ALL"]
        metrics = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
        _atomic_csv(
            os.path.join(self.summary_dir, "Final_Overall_Summary.csv"),
            self._paired_metric_summary(overall_base, overall_win, key_columns=[], metrics=metrics),
        )

        dataset_base = self._load_seed_frames(baseline, "Per_Dataset_Metrics.csv")
        dataset_win = self._load_seed_frames(winner, "Per_Dataset_Metrics.csv")
        _atomic_csv(
            os.path.join(self.summary_dir, "Final_Per_Dataset_Summary.csv"),
            self._paired_metric_summary(
                dataset_base, dataset_win, key_columns=["dataset"],
                metrics=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
            ),
        )

        class_base = self._load_seed_frames(baseline, "Per_Class_Metrics.csv")
        class_win = self._load_seed_frames(winner, "Per_Class_Metrics.csv")
        if "origin" in class_base.columns:
            class_base = class_base[class_base["origin"] == "ALL"]
            class_win = class_win[class_win["origin"] == "ALL"]
        _atomic_csv(
            os.path.join(self.summary_dir, "Final_Per_Class_Summary.csv"),
            self._paired_metric_summary(
                class_base, class_win, key_columns=["class"],
                metrics=["precision", "recall", "f1", "roc_auc_ovr", "pr_auc_ovr"],
            ),
        )

        for name, specs in (("Baseline", baseline), ("Winner", winner)):
            matrices = []
            for spec in specs:
                _, result_dir = self._paths(spec)
                matrices.append(pd.read_csv(os.path.join(result_dir, "Confusion_Matrix.csv"), index_col=0))
            combined = sum(matrices[1:], matrices[0].copy()) if len(matrices) > 1 else matrices[0]
            _atomic_csv(
                os.path.join(self.summary_dir, f"Final_Confusion_{name}.csv"),
                combined,
                index=True,
            )

        resource_base = self._load_seed_frames(baseline, "Resource_Accounting.csv")
        resource_win = self._load_seed_frames(winner, "Resource_Accounting.csv")
        resource_metrics = [
            value for value in (
                "total_parameters", "active_parameters_per_sample_mean",
                "forward_macs_per_sample_mean", "total_optimizer_steps", "total_wall_seconds",
            ) if value in resource_base.columns and value in resource_win.columns
        ]
        _atomic_csv(
            os.path.join(self.summary_dir, "Final_Resource_Summary.csv"),
            self._paired_metric_summary(resource_base, resource_win, key_columns=[], metrics=resource_metrics),
        )

        baseline_decision = self.state["decisions"]["baseline"]
        capacity_decision = self.state["decisions"]["capacity"]
        architecture_decision = self.state["decisions"]["architecture"]
        representation_decision = self.state["decisions"]["representation"]
        _atomic_csv(
            os.path.join(self.summary_dir, "Final_Decision_Summary.csv"),
            pd.DataFrame([{
                "selection_metric": self.study["selection"]["metric"],
                "baseline_validation_macro_f1_mean": self._mean_validation(
                    self._specs_from_ids(baseline_decision["run_ids"])
                ),
                "selected_expert_hidden_dims": json.dumps(
                    capacity_decision["expert_hidden_dims"]
                ),
                "selected_weight_decay": float(capacity_decision.get("weight_decay", 0.0)),
                "weight_decay_scope": "bounded_comparison_stage_not_final_hpo",
                "capacity_validation_macro_f1_mean": float(
                    capacity_decision["validation_macro_f1_mean"]
                ),
                "selected_architecture_choice": architecture_decision["choice"],
                "selected_architecture": architecture_decision["architecture"],
                "selected_adapter_rank": int(architecture_decision.get("adapter_rank", 16)),
                "architecture_validation_macro_f1_mean": float(
                    architecture_decision["validation_macro_f1_mean"]
                ),
                "selected_representation_objective": representation_decision["objective"],
                "selected_representation_weight": float(representation_decision["weight"]),
                "winner_validation_macro_f1_mean": float(
                    representation_decision["validation_macro_f1_mean"]
                ),
            }]),
        )

        latent_report_files = [
            f"Final_Latent_{kind.title()}_Summary.csv"
            for kind in self._LATENT_METRIC_FILES
            if os.path.isfile(
                os.path.join(self.summary_dir, f"Final_Latent_{kind.title()}_Summary.csv")
            )
        ]
        manifest = {
            "format_version": 1,
            "protocol_hash": self.protocol_hash,
            "baseline_run_ids": [spec.run_id for spec in baseline],
            "winner_run_ids": [spec.run_id for spec in winner],
            "decisions": self.state["decisions"],
            "report_files": [
                "Final_Decision_Summary.csv",
                "Final_Overall_Summary.csv", "Final_Per_Dataset_Summary.csv",
                "Final_Per_Class_Summary.csv", "Final_Resource_Summary.csv",
                "Final_Confusion_Baseline.csv", "Final_Confusion_Winner.csv",
                *latent_report_files,
            ],
            "insurance_probes": "deferred",
        }
        _atomic_json(os.path.join(self.summary_dir, "Final_Study_Manifest.json"), manifest)

    def _write_final_latent_summaries(
        self,
        baseline: list[RunSpec],
        winner: list[RunSpec],
    ) -> None:
        roles = (
            ("baseline", {spec.run_id for spec in baseline}),
            ("winner", {spec.run_id for spec in winner}),
        )
        for kind, (_source, summary_filename, group_columns) in self._LATENT_METRIC_FILES.items():
            path = os.path.join(self.summary_dir, summary_filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"missing harvested latent metrics: {path}")
            source = pd.read_csv(path)
            selected = []
            for role, run_ids in roles:
                local = source[source["run_id"].isin(run_ids)].copy()
                local.insert(0, "comparison_role", role)
                selected.append(local)
            combined = pd.concat(selected, ignore_index=True)
            self._write_latent_aggregate(
                combined,
                kind,
                ["comparison_role", *group_columns],
                output_prefix="Final_Latent",
            )

    def _write_latent_final_report(self, winner: list[RunSpec]) -> None:
        """Best-effort: highlight the worst-performing classes across every
        A/B/C latent snapshot for the phase-4 seed-0 winner. Never raises --
        latent reporting is diagnostic, not a gate on phase-4 completion."""
        seed0 = next((spec for spec in winner if spec.seed == 0), None)
        per_class_path = os.path.join(self.summary_dir, "Final_Per_Class_Summary.csv")
        if seed0 is None or not os.path.isfile(per_class_path):
            return
        latent_dir = self._latent_dir(seed0)
        if not os.path.isfile(os.path.join(latent_dir, "Latent_Report_Config.json")):
            return
        try:
            from evaluation.latent_space import load_latent_report, plot_class_focus, plot_latent_snapshots
            per_class = pd.read_csv(per_class_path)
            worst_classes = (
                per_class[per_class["metric"] == "f1"]
                .sort_values("winner_mean")["class"]
                .head(2)
            )
            report = load_latent_report(latent_dir)
            projections = plot_latent_snapshots(report, latent_dir, method="umap")
            present = set(report["manifest"]["class"])
            for class_name in worst_classes:
                if class_name in present:
                    plot_class_focus(report, projections, class_name, latent_dir)
        except Exception as exc:  # noqa: BLE001 -- diagnostic-only, must not block phase 4
            print(f"[greedy-study] latent final report skipped: {type(exc).__name__}: {exc}")

    def run_phase_4(self) -> None:
        baseline_decision = self._require_decision("baseline", "0")
        representation = self._require_decision("representation", "3c")
        baseline = self._specs_from_ids(baseline_decision["run_ids"])
        winner = self._specs_from_ids(representation["run_ids"])
        # Core C-stage latent reports are required for v2. Existing checkpoints
        # are backfilled through --latent-only without retraining or test inference.
        self.ensure_specs([*baseline, *winner])
        if not self._all_complete([*baseline, *winner]):
            return
        if self.latent_enabled:
            final_stages = self.latent_policy.get("final_stages", ["A", "B", "C"])
            final_plot = bool(self.latent_policy.get("final_plot_all_seeds", True))
            if not self.ensure_latent_specs(
                [*baseline, *winner], stages=final_stages, plot=final_plot
            ):
                return
            self._write_final_latent_summaries(baseline, winner)
        self.state["decisions"]["final"] = {
            "run_ids": [spec.run_id for spec in winner],
            "baseline_run_ids": [spec.run_id for spec in baseline],
            "objective": representation["objective"],
            "weight_decay": float(representation.get("weight_decay", 0.0)),
            "insurance_probes": "deferred",
        }
        self._write_final_report(baseline, winner)
        self._write_latent_final_report(winner)
        self.state["phases"]["4"] = {"status": "complete"}
        self._save_state()

    def run_phase(self, phase: str) -> None:
        dispatch = {
            "0": self.run_phase_0,
            "1": self.run_phase_1,
            "2": self.run_phase_2,
            "3a": self.run_phase_3a,
            "3b": self.run_phase_3b,
            "3c": self.run_phase_3c,
            "4": self.run_phase_4,
        }
        dispatch[phase]()
        print(f"[greedy-study] state={self.state_path}")
        print(f"[greedy-study] new evaluated runs this invocation={self.new_runs}")


def expected_evaluated_run_count(study: dict[str, Any]) -> int:
    """Count the finalized evaluated runs, excluding reusable Stage-A jobs."""
    baseline_weight_decay = float(study["backbone"]["training"].get("weight_decay", 0.0))
    weight_decays = [
        float(value) for value in study.get(
            "capacity_weight_decays", [baseline_weight_decay]
        )
    ]
    capacity_challengers = sum(
        1
        for values in study["capacity_candidates"]
        for weight_decay in weight_decays
        if list(values) or weight_decay != baseline_weight_decay
    )
    return (
        len(study["seeds"])  # phase 0
        + capacity_challengers + (len(study["seeds"]) - 1)  # phase 1
        + 2 * len(study["seeds"])  # phase 2
        + len(study["supcon_weights"])  # phase 3a
        + (len(study["seeds"]) - 1)  # phase 3b reuses seed 0
        + len(study["seeds"])  # phase 3c
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--study-config", default="config/greedy_moe_study.yaml")
    parser.add_argument("--prefix", default="nfv3_4way_greedy_v1")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--summary-dir")
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be a positive integer")
    runner = GreedyStudyRunner(
        base_config_path=args.config,
        study_config_path=args.study_config,
        prefix=args.prefix,
        summary_dir=args.summary_dir,
        execute=args.execute,
        max_new_runs=args.max_new_runs,
    )
    runner.run_phase(args.phase)


if __name__ == "__main__":
    main()
