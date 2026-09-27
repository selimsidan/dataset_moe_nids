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

    @property
    def run_id(self) -> str:
        return f"{self.condition}:seed{self.seed}"

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
        representation = nested["training"]["representation"]
        representation["objective"] = spec.representation_objective
        representation["weight"] = spec.representation_weight
        representation["sampling"] = "legacy"
        representation["class_weighting"] = "legacy"
        return nested

    def _stage_a_name(self, spec: RunSpec) -> str:
        objective = spec.representation_objective
        weight = _weight_label(spec.representation_weight)
        return f"{self.prefix}_stagea_{objective}_w{weight}_seed{spec.seed}_v1"

    def _run_name(self, spec: RunSpec) -> str:
        return f"{self.prefix}_{spec.condition}_seed{spec.seed}_v1"

    def _command(self, overrides: list[str]) -> list[str]:
        command = [
            sys.executable, "-u", "-m", "training.ooc_run",
            "--config", self.base_config_path,
        ]
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

    def _run_overrides(self, spec: RunSpec) -> list[str]:
        nested = self._condition_nested(spec)
        nested["run_name"] = self._run_name(spec)
        nested["training"]["stages"] = ["B", "C"]
        nested["training"]["run_final_evaluation"] = True
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(spec)
        return flatten_overrides(nested)

    def _resolved(self, spec: RunSpec) -> dict[str, Any]:
        return load_config(self.base_config_path, self._run_overrides(spec))

    def _paths(self, spec: RunSpec) -> tuple[str, str]:
        config = self._resolved(spec)
        return config["training"]["checkpoint_dir"], config["evaluation"]["output_dir"]

    def _missing_artifacts(self, spec: RunSpec) -> list[str]:
        checkpoint_dir, result_dir = self._paths(spec)
        missing = [
            filename for filename in self.study["required_artifacts"]
            if not os.path.isfile(os.path.join(result_dir, filename))
        ]
        for filename in (STAGE_C_FILE, STAGE_C_SUMMARY_FILE):
            if not os.path.isfile(os.path.join(checkpoint_dir, filename)):
                missing.append(f"checkpoints/{filename}")
        return missing

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

    def _print_job(self, spec: RunSpec) -> None:
        missing = self._missing_artifacts(spec)
        status = "complete" if not missing else "pending: " + ", ".join(missing)
        print(f"[{spec.phase}:{spec.run_id}] {status}")
        if missing:
            if not os.path.isfile(self._stage_a_path(spec)):
                print("  Stage A:", shlex.join(self._command(self._stage_a_overrides(spec))))
            print("  Run:", shlex.join(self._command(self._run_overrides(spec))))

    def _budget_available(self) -> bool:
        return self.max_new_runs is None or self.new_runs < self.max_new_runs

    def ensure_specs(self, specs: list[RunSpec]) -> None:
        for spec in specs:
            self._print_job(spec)
            if self._is_complete(spec):
                if self.execute:
                    self._record_complete(spec)
                continue
            if not self.execute or not self._budget_available():
                continue
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if not os.path.isfile(self._stage_a_path(spec)):
                subprocess.run(self._command(self._stage_a_overrides(spec)), check=True, env=env)
            subprocess.run(self._command(self._run_overrides(spec)), check=True, env=env)
            missing = self._missing_artifacts(spec)
            if missing:
                raise RuntimeError(f"{spec.run_id} finished without required artifacts: {missing}")
            self.new_runs += 1
            self._record_complete(spec)

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
                "condition": spec.condition,
                "seed": spec.seed,
                "split_seed": spec.seed,
                "architecture": spec.architecture,
                "expert_hidden_dims": json.dumps(list(spec.expert_hidden_dims)),
                "representation_objective": spec.representation_objective,
                "representation_weight": spec.representation_weight,
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
        return [RunSpec("0", "baseline_linear", seed, "moe_dataset_soft", ()) for seed in self.seeds]

    def capacity_coarse_specs(self) -> list[RunSpec]:
        return [
            RunSpec("1", f"capacity_{_capacity_label(values)}", 0, "moe_dataset_soft", tuple(values))
            for values in self.study["capacity_candidates"]
        ]

    def architecture_specs(self) -> list[RunSpec]:
        capacity = self._require_decision("capacity", "1")
        hidden = tuple(capacity["expert_hidden_dims"])
        values = []
        for seed in self.seeds:
            values.append(RunSpec("2", "private_encoders", seed, "moe_dataset_private_encoders", hidden))
            values.append(RunSpec("2", "adapters_r16", seed, "moe_dataset_adapters", hidden, adapter_rank=16))
        return values

    def supcon_sweep_specs(self) -> list[RunSpec]:
        architecture = self._require_decision("architecture", "2")
        hidden = tuple(architecture["expert_hidden_dims"])
        return [
            RunSpec(
                "3a", f"supcon_w{_weight_label(float(weight))}", 0,
                architecture["architecture"], hidden, "supcon", float(weight),
            )
            for weight in self.study["supcon_weights"]
        ]

    def run_phase_0(self) -> None:
        specs = self.baseline_specs()
        self.ensure_specs(specs)
        if self._all_complete(specs):
            self.state["decisions"]["baseline"] = {"run_ids": [spec.run_id for spec in specs]}
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
            tie_key=lambda spec: _expert_parameter_count(spec.expert_hidden_dims),
        )
        self.state["decisions"]["capacity_coarse"] = {
            "expert_hidden_dims": list(winner.expert_hidden_dims),
            "seed0_run_id": winner.run_id,
            "validation_macro_f1": self._validation_score(winner),
        }
        confirmations = [
            RunSpec("1", winner.condition, seed, winner.architecture, winner.expert_hidden_dims)
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
        self.state["decisions"]["capacity"] = {
            "expert_hidden_dims": selected_hidden,
            "condition": selected[0].condition,
            "run_ids": [spec.run_id for spec in selected],
            "validation_macro_f1_mean": self._mean_validation(selected),
            "confirmed_candidate_hidden_dims": list(winner.expert_hidden_dims),
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

        manifest = {
            "format_version": 1,
            "protocol_hash": self.protocol_hash,
            "baseline_run_ids": [spec.run_id for spec in baseline],
            "winner_run_ids": [spec.run_id for spec in winner],
            "decisions": self.state["decisions"],
            "report_files": [
                "Final_Overall_Summary.csv", "Final_Per_Dataset_Summary.csv",
                "Final_Per_Class_Summary.csv", "Final_Resource_Summary.csv",
                "Final_Confusion_Baseline.csv", "Final_Confusion_Winner.csv",
            ],
            "insurance_probes": "deferred",
        }
        _atomic_json(os.path.join(self.summary_dir, "Final_Study_Manifest.json"), manifest)

    def run_phase_4(self) -> None:
        baseline_decision = self._require_decision("baseline", "0")
        representation = self._require_decision("representation", "3c")
        baseline = self._specs_from_ids(baseline_decision["run_ids"])
        winner = self._specs_from_ids(representation["run_ids"])
        # Normally complete already; this regenerates missing reports from the
        # original checkpoints without creating duplicate run identities.
        self.ensure_specs(winner)
        if not self._all_complete(winner):
            return
        self.state["decisions"]["final"] = {
            "run_ids": [spec.run_id for spec in winner],
            "baseline_run_ids": [spec.run_id for spec in baseline],
            "objective": representation["objective"],
            "insurance_probes": "deferred",
        }
        self._write_final_report(baseline, winner)
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
    return (
        len(study["seeds"])  # phase 0
        + len(study["capacity_candidates"]) + (len(study["seeds"]) - 1)  # phase 1
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
