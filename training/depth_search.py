"""Adaptive targeted multi-depth study for the full NF-v3 pipeline.

The v3 runner is deliberately separate from :mod:`training.ablation_matrix`:
v2 remains an immutable paper protocol, while this state machine supports
role-specific encoders, reusable A/B artifacts, validation-only selection,
conditional expansion, and a hard search budget.
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
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd
import yaml

from .checkpoint import (
    BASELINE_MODEL_FILE,
    BASELINE_STAGE_B_FILE,
    STAGE_A_FILE,
    STAGE_B_FILE,
    STAGE_C_FILE,
)
from .config import load_config


PHASES = ("1", "2", "3", "4")
STATE_VERSION = 1
VALIDATION_ARTIFACTS = {
    "Validation_Overall_Metrics.csv",
    "Validation_Per_Dataset_Metrics.csv",
    "Validation_Per_Class_Metrics.csv",
    "Validation_Expert_Owned_StageB.csv",
    "Validation_Expert_Owned_StageC.csv",
}


def _hash(value: Any, length: int = 16) -> str:
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


def read_validation_artifact(directory: str, filename: str) -> pd.DataFrame:
    """Selection-time CSV reader that structurally excludes test reports."""
    if filename not in VALIDATION_ARTIFACTS:
        raise ValueError(f"selection cannot read non-validation artifact {filename!r}")
    return pd.read_csv(os.path.join(directory, filename))


@dataclass
class DepthRunSpec:
    phase: str
    condition: str
    seed: int
    architecture: str = "moe_dataset_soft"
    encoder: dict[str, Any] = field(default_factory=dict)
    private_encoder: dict[str, Any] | None = None
    expert_hidden_dims: list[int] = field(default_factory=list)
    latent_dim: int = 64
    adapter_rank: int = 16
    gate_hidden_dims: list[int] = field(default_factory=list)
    gate_supervision: str = "none"
    stage_c_unfreeze: str = "all"
    representation_objective: str = "ce"
    representation_weight: float = 0.1
    weight_decay: float = 0.0001
    stage_c_lr: float = 0.001
    lambda_balance: float = 0.1
    lambda_expert_anchor: float = 0.0

    @property
    def run_id(self) -> str:
        return f"{self.condition}:seed{self.seed}"

    def clone(self, **updates) -> "DepthRunSpec":
        value = copy.deepcopy(asdict(self))
        value.update(updates)
        return DepthRunSpec(**value)


class TargetedDepthStudy:
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
        with open(study_config_path) as handle:
            self.study = yaml.safe_load(handle)
        if self.study.get("format_version") != 3:
            raise ValueError("targeted depth study requires format_version: 3")
        if self.study.get("seeds") != [0, 1, 2]:
            raise ValueError("targeted depth study requires paired seeds [0, 1, 2]")
        base = load_config(base_config_path)
        self.base_config = base
        self.summary_dir = summary_dir or os.path.join(
            base["OUTPUT_DIR"], "results", f"{prefix}_summary"
        )
        self.state_path = os.path.join(self.summary_dir, "study_state.json")
        self.protocol_hash = self._protocol_hash()
        self.state = self._load_state()
        self._previewed: set[str] = set()

    @property
    def ceiling(self) -> int:
        return int(self.study.get("hard_configuration_ceiling", 36))

    @property
    def selection(self) -> dict[str, Any]:
        return self.study["selection"]

    @property
    def encoders(self) -> dict[str, dict[str, Any]]:
        return self.study["encoders"]

    def _protocol_hash(self) -> str:
        with open(self.base_config_path, "rb") as handle:
            base_sha = hashlib.sha256(handle.read()).hexdigest()
        return _hash({"study": self.study, "base_sha256": base_sha}, 64)

    def _load_state(self) -> dict[str, Any]:
        if not os.path.isfile(self.state_path):
            return {
                "format_version": STATE_VERSION,
                "protocol_hash": self.protocol_hash,
                "prefix": self.prefix,
                "runs": {},
                "decisions": {},
                "phases": {},
            }
        with open(self.state_path) as handle:
            state = json.load(handle)
        if state.get("protocol_hash") != self.protocol_hash:
            raise ValueError("v3 protocol changed after execution began; use a new prefix")
        return state

    def _save_state(self) -> None:
        if not self.execute:
            return
        os.makedirs(self.summary_dir, exist_ok=True)
        temporary = self.state_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(self.state, handle, indent=2, sort_keys=True)
        os.replace(temporary, self.state_path)

    def _base_nested(self) -> dict[str, Any]:
        return copy.deepcopy(self.study["backbone"])

    def _spec_nested(self, spec: DepthRunSpec) -> dict[str, Any]:
        nested = self._base_nested()
        nested["seed"] = spec.seed
        nested["data"]["split_seed"] = spec.seed
        nested["architecture"] = spec.architecture
        model = nested["model"]
        model["latent_dim"] = spec.latent_dim
        model["encoder"] = copy.deepcopy(spec.encoder)
        model["gate_encoder"] = copy.deepcopy(spec.encoder)
        if spec.private_encoder is not None:
            model["private_encoder"] = copy.deepcopy(spec.private_encoder)
        else:
            model.pop("private_encoder", None)
        model["expert"]["hidden_dims"] = list(spec.expert_hidden_dims)
        model["adapter"]["rank"] = spec.adapter_rank
        model["gate"]["hidden_dims"] = list(spec.gate_hidden_dims)
        training = nested["training"]
        training["weight_decay"] = spec.weight_decay
        training["stage_a"]["optimizer"]["weight_decay"] = spec.weight_decay
        training["stage_b"]["optimizer"]["weight_decay"] = spec.weight_decay
        training["stage_c"]["optimizer"] = {
            "lr": spec.stage_c_lr,
            "weight_decay": spec.weight_decay,
        }
        training["stage_c_unfreeze"] = spec.stage_c_unfreeze
        training["stage_c"]["gate_supervision"] = spec.gate_supervision
        training["stage_c"]["lambda_expert_anchor"] = spec.lambda_expert_anchor
        training["representation"]["objective"] = spec.representation_objective
        training["representation"]["weight"] = spec.representation_weight
        nested["load_balance"]["lambda_balance"] = spec.lambda_balance
        return nested

    def _resolved(self, nested: dict[str, Any]) -> dict[str, Any]:
        return load_config(self.base_config_path, flatten_overrides(nested))

    def _command(self, nested: dict[str, Any]) -> list[str]:
        command = [sys.executable, "-u", "-m", "training.ooc_run", "--config", self.base_config_path]
        for override in flatten_overrides(nested):
            command.extend(["--set", override])
        return command

    def _run_command(self, label: str, nested: dict[str, Any]) -> None:
        command = self._command(nested)
        printable = shlex.join(command)
        if not self.execute:
            key = f"{label}:{printable}"
            if key not in self._previewed:
                print(f"[{label}] {printable}")
                self._previewed.add(key)
            return
        print(f"[{label}] {printable}", flush=True)
        subprocess.run(command, check=True)

    def _stage_a_identity(
        self, spec: DepthRunSpec, encoder: dict[str, Any], role: str
    ) -> dict[str, Any]:
        return {
            "protocol_hash": self.protocol_hash,
            "seed": spec.seed,
            "encoder_role": role,
            "encoder": encoder,
            "latent_dim": spec.latent_dim,
            "data": {
                "active_datasets": self.study["backbone"]["data"]["active_datasets"],
                "split": self.study["backbone"]["data"].get(
                    "split", self.base_config["data"].get("split")
                ),
            },
            "representation": {
                "objective": spec.representation_objective,
                "weight": spec.representation_weight,
                "sampling": self.study["backbone"]["training"]["representation"]["sampling"],
                "class_weighting": self.study["backbone"]["training"]["representation"]["class_weighting"],
            },
            "optimizer": {
                **self.study["backbone"]["training"]["stage_a"]["optimizer"],
                "weight_decay": spec.weight_decay,
            },
        }

    def _stage_a_nested(
        self, spec: DepthRunSpec, encoder: dict[str, Any], role: str
    ) -> dict[str, Any]:
        identity = self._stage_a_identity(spec, encoder, role)
        nested = self._spec_nested(spec)
        nested["architecture"] = "moe_dataset_soft"
        nested["model"]["encoder"] = copy.deepcopy(encoder)
        nested["model"].pop("gate_encoder", None)
        nested["model"].pop("private_encoder", None)
        nested["model"]["expert"]["hidden_dims"] = []
        nested["training"]["stages"] = ["A"]
        nested["training"]["run_final_evaluation"] = False
        nested["run_name"] = f"{self.prefix}_cache_a_{_hash(identity)}"
        return nested

    def _stage_a_path(
        self, spec: DepthRunSpec, encoder: dict[str, Any], role: str
    ) -> str:
        cfg = self._resolved(self._stage_a_nested(spec, encoder, role))
        return os.path.join(cfg["training"]["checkpoint_dir"], STAGE_A_FILE)

    def _ensure_stage_a(
        self, spec: DepthRunSpec, encoder: dict[str, Any], role: str
    ) -> bool:
        path = self._stage_a_path(spec, encoder, role)
        if os.path.isfile(path):
            return True
        self._run_command("stage-a", self._stage_a_nested(spec, encoder, role))
        return (not self.execute) or os.path.isfile(path)

    @staticmethod
    def _primary_encoder_role(spec: DepthRunSpec) -> str:
        return "gate" if spec.architecture == "moe_dataset_private_encoders" else "shared"

    def _stage_b_identity(self, spec: DepthRunSpec) -> dict[str, Any]:
        identity = {
            "protocol_hash": self.protocol_hash,
            "seed": spec.seed,
            "architecture": spec.architecture,
            "encoder": spec.encoder,
            "private_encoder": spec.private_encoder,
            "latent_dim": spec.latent_dim,
            "expert_hidden_dims": spec.expert_hidden_dims,
            "adapter_rank": spec.adapter_rank,
            "weight_decay": spec.weight_decay,
            "stage_a": self._stage_a_path(
                spec, spec.encoder, self._primary_encoder_role(spec)
            ),
            "optimizer": {
                **self.study["backbone"]["training"]["stage_b"]["optimizer"],
                "weight_decay": spec.weight_decay,
            },
        }
        if spec.architecture == "moe_dataset_private_encoders":
            identity["private_stage_a"] = self._stage_a_path(
                spec, spec.private_encoder or spec.encoder, "private"
            )
        return identity

    def _stage_b_nested(self, spec: DepthRunSpec) -> dict[str, Any]:
        nested = self._spec_nested(spec)
        nested["training"]["stages"] = ["B"]
        nested["training"]["run_final_evaluation"] = False
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(
            spec, spec.encoder, self._primary_encoder_role(spec)
        )
        if spec.architecture == "moe_dataset_private_encoders":
            nested["training"]["private_stage_a_checkpoint"] = self._stage_a_path(
                spec, spec.private_encoder or spec.encoder, "private"
            )
        nested["run_name"] = f"{self.prefix}_cache_b_{_hash(self._stage_b_identity(spec))}"
        return nested

    def _stage_b_path(self, spec: DepthRunSpec) -> str:
        cfg = self._resolved(self._stage_b_nested(spec))
        filename = (
            BASELINE_STAGE_B_FILE
            if spec.architecture in {"plain_pooled", "matched_dense"}
            else STAGE_B_FILE
        )
        return os.path.join(cfg["training"]["checkpoint_dir"], filename)

    def _ensure_stage_b(self, spec: DepthRunSpec) -> bool:
        if not self._ensure_stage_a(
            spec, spec.encoder, self._primary_encoder_role(spec)
        ):
            return False
        if spec.architecture == "moe_dataset_private_encoders":
            if not self._ensure_stage_a(
                spec, spec.private_encoder or spec.encoder, "private"
            ):
                return False
        path = self._stage_b_path(spec)
        if os.path.isfile(path):
            return True
        self._run_command("stage-b", self._stage_b_nested(spec))
        return (not self.execute) or os.path.isfile(path)

    def _run_name(self, spec: DepthRunSpec) -> str:
        identity = asdict(spec); identity.pop("phase", None)
        identity["protocol_hash"] = self.protocol_hash
        return f"{self.prefix}_{spec.condition}_seed{spec.seed}_{_hash(identity, 10)}"

    def _run_nested(self, spec: DepthRunSpec, *, final_test: bool = False) -> dict[str, Any]:
        nested = self._spec_nested(spec)
        nested["run_name"] = self._run_name(spec)
        nested["training"]["stage_a_checkpoint"] = self._stage_a_path(
            spec, spec.encoder, self._primary_encoder_role(spec)
        )
        if spec.architecture == "moe_dataset_private_encoders":
            nested["training"]["private_stage_a_checkpoint"] = self._stage_a_path(
                spec, spec.private_encoder or spec.encoder, "private"
            )
        if spec.architecture in {"plain_pooled", "matched_dense"}:
            nested["training"]["dense_stage_b_checkpoint"] = self._stage_b_path(spec)
            nested["training"]["stages"] = ["C"]
        else:
            nested["training"]["stage_b_checkpoint"] = self._stage_b_path(spec)
            nested["training"]["stages"] = ["C"]
        nested["training"]["run_final_evaluation"] = bool(final_test)
        return nested

    def _paths(self, spec: DepthRunSpec) -> tuple[str, str]:
        cfg = self._resolved(self._run_nested(spec))
        return cfg["training"]["checkpoint_dir"], cfg["evaluation"]["output_dir"]

    def _complete(self, spec: DepthRunSpec) -> bool:
        checkpoint_dir, result_dir = self._paths(spec)
        model_file = BASELINE_MODEL_FILE if spec.architecture in {"plain_pooled", "matched_dense"} else STAGE_C_FILE
        required = [
            os.path.join(checkpoint_dir, model_file),
            *[
                os.path.join(result_dir, name)
                for name in self.study["required_validation_artifacts"]
            ],
        ]
        return all(os.path.isfile(path) for path in required)

    def ensure_specs(self, specs: Iterable[DepthRunSpec]) -> bool:
        for spec in specs:
            if self._complete(spec):
                self._record(spec)
                continue
            if self.max_new_runs is not None and self.new_runs >= self.max_new_runs:
                return False
            if len(self.state["runs"]) >= self.ceiling:
                raise RuntimeError(f"hard configuration ceiling ({self.ceiling}) reached")
            if not self._ensure_stage_b(spec):
                continue
            self._run_command("evaluate", self._run_nested(spec))
            if self.execute:
                self.new_runs += 1
                if not self._complete(spec):
                    raise RuntimeError(f"run {spec.run_id} completed without validation artifacts")
                self._record(spec)
        return all(self._complete(spec) for spec in specs)

    def _record(self, spec: DepthRunSpec) -> None:
        _, result_dir = self._paths(spec)
        overall = read_validation_artifact(result_dir, "Validation_Overall_Metrics.csv").iloc[0]
        self.state["runs"][spec.run_id] = {
            "spec": asdict(spec),
            "run_name": self._run_name(spec),
            "validation_macro_f1": float(overall["macro_f1"]),
            "total_parameters": int(overall.get("total_parameters", 0)),
            "forward_macs_per_sample_mean": int(
                overall.get("forward_macs_per_sample_mean", 0)
            ),
            "result_dir": result_dir,
        }
        self._save_state()

    def _profile(self, spec: DepthRunSpec) -> tuple[float, float, float, int, int]:
        _, result_dir = self._paths(spec)
        overall = read_validation_artifact(result_dir, "Validation_Overall_Metrics.csv").iloc[0]
        per_dataset = read_validation_artifact(result_dir, "Validation_Per_Dataset_Metrics.csv")
        per_class = read_validation_artifact(result_dir, "Validation_Per_Class_Metrics.csv")
        rare = per_class[
            (per_class["support"] >= int(self.selection["rare_validation_min_support"]))
            & (per_class["support"] < int(self.selection.get("rare_validation_max_support", 200)))
        ]
        rare_recall = float(rare["recall"].mean()) if len(rare) else 0.0
        return (
            float(overall["macro_f1"]),
            float(per_dataset["macro_f1"].min()),
            rare_recall,
            int(overall.get("total_parameters", 0)),
            int(overall.get("forward_macs_per_sample_mean", 0)),
        )

    def _rank_seed0(self, specs: list[DepthRunSpec]) -> list[DepthRunSpec]:
        near_tie = float(self.selection["near_tie"])
        best = max(self._profile(spec)[0] for spec in specs)
        return sorted(
            specs,
            key=lambda spec: (
                0 if self._profile(spec)[0] >= best - near_tie else 1,
                -self._profile(spec)[1]
                if self._profile(spec)[0] >= best - near_tie
                else -self._profile(spec)[0],
                -self._profile(spec)[2]
                if self._profile(spec)[0] >= best - near_tie
                else 0.0,
                -self._profile(spec)[0],
                self._profile(spec)[3],
                self._profile(spec)[4],
            ),
        )

    def _mean_score(self, specs: list[DepthRunSpec]) -> float:
        return sum(self._profile(spec)[0] for spec in specs) / len(specs)

    def _consistent_regression(
        self, candidate: list[DepthRunSpec], reference: list[DepthRunSpec]
    ) -> bool:
        dataset_limit = float(self.selection["dataset_regression_limit"])
        recall_limit = float(self.selection["rare_recall_regression_limit"])
        min_support = int(self.selection["rare_validation_min_support"])
        max_support = int(self.selection.get("rare_validation_max_support", 200))
        dataset_deltas: dict[str, list[float]] = {}
        class_deltas: dict[str, list[float]] = {}
        by_seed_candidate = {spec.seed: spec for spec in candidate}
        by_seed_reference = {spec.seed: spec for spec in reference}
        for seed in sorted(set(by_seed_candidate) & set(by_seed_reference)):
            _, cand_dir = self._paths(by_seed_candidate[seed])
            _, ref_dir = self._paths(by_seed_reference[seed])
            cand_ds = read_validation_artifact(cand_dir, "Validation_Per_Dataset_Metrics.csv").set_index("dataset")
            ref_ds = read_validation_artifact(ref_dir, "Validation_Per_Dataset_Metrics.csv").set_index("dataset")
            for name in cand_ds.index.intersection(ref_ds.index):
                dataset_deltas.setdefault(str(name), []).append(
                    float(cand_ds.loc[name, "macro_f1"] - ref_ds.loc[name, "macro_f1"])
                )
            cand_cls = read_validation_artifact(cand_dir, "Validation_Per_Class_Metrics.csv").set_index("class")
            ref_cls = read_validation_artifact(ref_dir, "Validation_Per_Class_Metrics.csv").set_index("class")
            for name in cand_cls.index.intersection(ref_cls.index):
                support = min(float(cand_cls.loc[name, "support"]), float(ref_cls.loc[name, "support"]))
                if support < min_support or support >= max_support:
                    continue
                class_deltas.setdefault(str(name), []).append(
                    float(cand_cls.loc[name, "recall"] - ref_cls.loc[name, "recall"])
                )
        if any(sum(value < 0 for value in values) >= 2 and sum(values) / len(values) < -dataset_limit
               for values in dataset_deltas.values()):
            return True
        return any(
            sum(value < 0 for value in values) >= 2 and sum(values) / len(values) < -recall_limit
            for values in class_deltas.values()
        )

    def _robust_group_choice(
        self, groups: list[list[DepthRunSpec]]
    ) -> list[DepthRunSpec]:
        groups = sorted(groups, key=lambda group: -self._mean_score(group))
        best = groups[0]
        near_tie = float(self.selection["near_tie"])
        tied = [group for group in groups if self._mean_score(group) >= self._mean_score(best) - near_tie]
        survivors = [
            group for group in tied
            if not any(
                self._consistent_regression(group, other)
                for other in tied if other is not group
            )
        ] or tied
        def key(group):
            profiles = [self._profile(spec) for spec in group]
            return (
                min(profile[1] for profile in profiles),
                sum(profile[2] for profile in profiles) / len(profiles),
                self._mean_score(group),
                -sum(profile[3] for profile in profiles) / len(profiles),
                -sum(profile[4] for profile in profiles) / len(profiles),
            )
        return max(survivors, key=key)

    def _owned_expert_degradation(self, spec: DepthRunSpec) -> float | None:
        if spec.architecture not in {"moe_dataset_soft", "moe_dataset_private_encoders"}:
            return None
        stage_b_path = os.path.join(
            os.path.dirname(self._stage_b_path(spec)), "Validation_Expert_Owned_StageB.csv"
        )
        _, result_dir = self._paths(spec)
        stage_c_path = os.path.join(result_dir, "Validation_Expert_Owned_StageC.csv")
        if not os.path.isfile(stage_b_path) or not os.path.isfile(stage_c_path):
            return None
        before = read_validation_artifact(
            os.path.dirname(stage_b_path), os.path.basename(stage_b_path)
        ).set_index("dataset")
        after = read_validation_artifact(
            os.path.dirname(stage_c_path), os.path.basename(stage_c_path)
        ).set_index("dataset")
        common = before.index.intersection(after.index)
        if len(common) == 0:
            return None
        return float((before.loc[common, "macro_f1"] - after.loc[common, "macro_f1"]).mean())

    def _base_spec(self, condition: str, encoder_name: str, expert: list[int], **updates) -> DepthRunSpec:
        value = DepthRunSpec(
            phase="1", condition=condition, seed=0,
            encoder=copy.deepcopy(self.encoders[encoder_name]),
            expert_hidden_dims=list(expert),
        )
        return value.clone(**updates)

    def phase1_base_specs(self) -> list[DepthRunSpec]:
        return [
            self._base_spec("shared_shallow_linear_wd0", "shallow", [], weight_decay=0.0),
            self._base_spec("shared_shallow_linear_wd1e4", "shallow", []),
            self._base_spec("shared_shallow_h64", "shallow", [64]),
            self._base_spec("shared_shallow_h256x128", "shallow", [256, 128]),
            self._base_spec("shared_plain_deep_h64", "plain_deep", [64]),
            self._base_spec("shared_res2_bn_h64", "residual_batchnorm", [64]),
            self._base_spec("shared_res2_ln_h64", "residual_layernorm", [64]),
        ]

    def _spec_from_state(self, value: dict[str, Any]) -> DepthRunSpec:
        return DepthRunSpec(**copy.deepcopy(value))

    def run_phase_1(self) -> None:
        base = self.phase1_base_specs()
        if not self.ensure_specs(base):
            print("[phase 1] base structural screen is incomplete")
            return
        encoder_group = [spec for spec in base if spec.condition in {
            "shared_shallow_h64", "shared_plain_deep_h64", "shared_res2_bn_h64", "shared_res2_ln_h64"
        }]
        head_group = [spec for spec in base if spec.condition in {
            "shared_shallow_linear_wd1e4", "shared_shallow_h64", "shared_shallow_h256x128"
        }]
        best_encoder = self._rank_seed0(encoder_group)[0]
        best_head = self._rank_seed0(head_group)[0]
        adaptive = []
        if best_encoder.encoder != best_head.encoder or best_encoder.expert_hidden_dims != best_head.expert_hidden_dims:
            adaptive.append(best_encoder.clone(
                condition="shared_cross_best_encoder_head",
                expert_hidden_dims=best_head.expert_hidden_dims,
            ))
        joint = adaptive[0] if adaptive else (
            best_encoder if best_encoder.expert_hidden_dims == best_head.expert_hidden_dims else best_head
        )
        adaptive.append(joint.clone(condition="shared_best_latent128", latent_dim=128))
        encoder_gain = self._profile(best_encoder)[0] - self._profile(
            next(spec for spec in encoder_group if spec.condition == "shared_shallow_h64")
        )[0]
        expert_gain = self._profile(best_head)[0] - self._profile(
            next(spec for spec in head_group if spec.condition == "shared_shallow_h64")
        )[0]
        threshold = float(self.selection["expansion_min_gain"])
        if max(encoder_gain, expert_gain) >= threshold:
            if expert_gain > encoder_gain and best_head.expert_hidden_dims == [256, 128]:
                adaptive.append(joint.clone(
                    condition="shared_expand_expert_h256x128x64",
                    expert_hidden_dims=[256, 128, 64],
                ))
            elif best_encoder.encoder.get("kind", "mlp") == "residual_mlp":
                expanded = copy.deepcopy(best_encoder.encoder); expanded["blocks"] = 4
                adaptive.append(joint.clone(condition="shared_expand_res4", encoder=expanded))
            elif best_encoder.encoder == self.encoders["plain_deep"]:
                expanded = copy.deepcopy(best_encoder.encoder); expanded["hidden_dims"] = [512, 256, 128]
                adaptive.append(joint.clone(condition="shared_expand_plain", encoder=expanded))
        adaptive = adaptive[:3]
        if not self.ensure_specs(adaptive):
            print("[phase 1] adaptive structural trials are incomplete")
            return
        winner = self._rank_seed0([*base, *adaptive])[0]
        self.state["decisions"]["structural"] = asdict(winner)
        deeper_candidates = [
            spec for spec in [*base, *adaptive]
            if spec.encoder != self.encoders["shallow"]
        ]
        self.state["decisions"]["deep_encoder"] = copy.deepcopy(
            self._rank_seed0(deeper_candidates)[0].encoder
        )
        self.state["decisions"]["selected_expert_hidden_dims"] = list(
            self._rank_seed0(head_group)[0].expert_hidden_dims
        )
        self.state["phases"]["1"] = {"status": "complete"}
        self._save_state()

    def _require(self, decision: str, phase: str) -> dict[str, Any]:
        value = self.state["decisions"].get(decision)
        if value is None:
            raise RuntimeError(f"phase {phase} is blocked until decision {decision!r} exists")
        return value

    def run_phase_2(self) -> None:
        structural = self._spec_from_state(self._require("structural", "1"))
        shallow = copy.deepcopy(self.encoders["shallow"])
        deep = copy.deepcopy(self._require("deep_encoder", "1"))
        head = list(self._require("selected_expert_hidden_dims", "1"))
        private = [
            structural.clone(phase="2", condition=f"private_g{g}_p{p}", seed=0,
                architecture="moe_dataset_private_encoders",
                encoder=copy.deepcopy(shallow if g == "s" else deep),
                private_encoder=copy.deepcopy(shallow if p == "s" else deep),
                expert_hidden_dims=head)
            for g, p in (("s", "s"), ("s", "d"), ("d", "s"), ("d", "d"))
        ]
        if not self.ensure_specs(private):
            print("[phase 2] private 2x2 depth matrix is incomplete")
            return
        best_private = self._rank_seed0(private)[0]
        alternate_head = [] if head else [64]
        private_head = best_private.clone(
            condition=best_private.condition + "_head_alt",
            expert_hidden_dims=alternate_head,
        )
        adapters = [
            structural.clone(
                phase="2", condition=f"adapter_{depth}_r{rank}", seed=0,
                architecture="moe_dataset_adapters",
                encoder=copy.deepcopy(shallow if depth == "s" else deep),
                private_encoder=None, adapter_rank=rank,
            )
            for depth in ("s", "d") for rank in (16, 32)
        ]
        controls = [
            structural.clone(phase="2", condition="plain_pooled_selected", seed=0,
                architecture="plain_pooled", private_encoder=None),
            structural.clone(phase="2", condition="matched_dense_selected", seed=0,
                architecture="matched_dense", private_encoder=None),
        ]
        seed0 = [structural, *private, private_head, *adapters, *controls]
        if not self.ensure_specs([private_head, *adapters, *controls]):
            print("[phase 2] family interactions are incomplete")
            return
        top3 = self._rank_seed0(seed0)[:3]
        seed1 = [spec.clone(phase="2", seed=1) for spec in top3]
        if not self.ensure_specs(seed1):
            print("[phase 2] seed-1 promotion is incomplete")
            return
        pairs = [(base, promoted) for base, promoted in zip(top3, seed1)]
        pairs.sort(key=lambda pair: -self._mean_score(list(pair)))
        top2_pairs = pairs[:2]
        seed2 = [pair[0].clone(phase="2", seed=2) for pair in top2_pairs]
        if not self.ensure_specs(seed2):
            print("[phase 2] seed-2 finalist confirmation is incomplete")
            return
        finalists = []
        for (seed0_spec, seed1_spec), seed2_spec in zip(top2_pairs, seed2):
            finalists.append([seed0_spec, seed1_spec, seed2_spec])
        finalists.sort(key=lambda group: -self._mean_score(group))
        self.state["decisions"]["leader"] = [asdict(spec) for spec in finalists[0]]
        self.state["decisions"]["runner_up"] = [asdict(spec) for spec in finalists[1]]
        self.state["phases"]["2"] = {"status": "complete"}
        self._save_state()

    def run_phase_3(self) -> None:
        leader_group = [self._spec_from_state(value) for value in self._require("leader", "2")]
        leader = next(spec for spec in leader_group if spec.seed == 0)
        candidates = [leader]
        # Reserve six identities for tuned seeds 1-2 plus three-seed shallow
        # and dense final references. Conditional refinement never consumes
        # those slots, so the global 36-ID ceiling remains enforceable.
        refinement_budget = max(0, min(8, self.ceiling - 6 - len(self.state["runs"])))
        initial = []
        if leader.architecture.startswith("moe_"):
            initial.append(leader.clone(
                phase="3", condition=leader.condition + "_gate64", gate_hidden_dims=[64]
            ))
        initial.extend([
            leader.clone(phase="3", condition=leader.condition + "_lastlayer", stage_c_unfreeze="last_layer"),
            leader.clone(phase="3", condition=leader.condition + "_supcon01",
                representation_objective="supcon", representation_weight=0.1),
            leader.clone(phase="3", condition=leader.condition + "_supcon03",
                representation_objective="supcon", representation_weight=0.3),
            leader.clone(phase="3", condition=leader.condition + "_lr3e4", stage_c_lr=0.0003),
        ])
        initial = initial[:refinement_budget]
        if not self.ensure_specs(initial):
            print("[phase 3] initial finalist refinements are incomplete")
            return
        candidates.extend(initial)
        remaining = refinement_budget - len(initial)
        gate_best = leader
        gate64 = next((spec for spec in candidates if spec.gate_hidden_dims == [64]), None)
        if gate64 is not None and self._profile(gate64)[0] > self._profile(leader)[0]:
            gate_best = gate64
        if leader.architecture.startswith("moe_") and remaining > 0:
            light_aux = gate_best.clone(
                phase="3", condition=gate_best.condition + "_lightaux", gate_supervision="light_aux"
            )
            if not self.ensure_specs([light_aux]):
                return
            candidates.append(light_aux)
            remaining -= 1
        supcons = [spec for spec in candidates if spec.representation_objective == "supcon"]
        if supcons and remaining > 0:
            supcon_best = max(supcons, key=lambda spec: self._profile(spec)[0])
            balanced = supcon_best.clone(
                phase="3", condition=leader.condition + f"_balanced{supcon_best.representation_weight:g}",
                representation_objective="balanced_supcon",
            )
            if not self.ensure_specs([balanced]):
                return
            candidates.append(balanced)
            remaining -= 1
        if remaining > 0:
            threshold = float(self.selection["expansion_min_gain"])
            if gate64 is not None and self._profile(gate64)[0] >= self._profile(leader)[0] + threshold:
                diagnostic = gate64.clone(
                    phase="3", condition=leader.condition + "_gate64x32", gate_hidden_dims=[64, 32]
                )
            elif (self._owned_expert_degradation(leader) or 0.0) > 0.02:
                diagnostic = leader.clone(
                    phase="3", condition=leader.condition + "_anchor1e4", lambda_expert_anchor=0.0001
                )
            else:
                diagnostic = leader.clone(
                    phase="3", condition=leader.condition + "_balance001", lambda_balance=0.01
                )
            if not self.ensure_specs([diagnostic]):
                return
            candidates.append(diagnostic)
        tuned = self._rank_seed0(candidates)[0]
        self.state["decisions"]["tuned"] = asdict(tuned)
        self.state["phases"]["3"] = {"status": "complete"}
        self._save_state()

    def _test_artifacts_complete(self, spec: DepthRunSpec) -> bool:
        _, result_dir = self._paths(spec)
        required = ["Overall_Metrics.csv", "Per_Dataset_Metrics.csv", "Per_Class_Metrics.csv"]
        return all(os.path.isfile(os.path.join(result_dir, name)) for name in required)

    def _ensure_test(self, spec: DepthRunSpec) -> None:
        if self._test_artifacts_complete(spec):
            return
        self._run_command("final-test", self._run_nested(spec, final_test=True))
        if self.execute and not self._test_artifacts_complete(spec):
            raise RuntimeError(f"final test artifacts missing for {spec.run_id}")

    def run_phase_4(self) -> None:
        tuned0 = self._spec_from_state(self._require("tuned", "3"))
        tuned = [tuned0, tuned0.clone(phase="4", seed=1), tuned0.clone(phase="4", seed=2)]
        if not self.ensure_specs(tuned[1:]):
            print("[phase 4] tuned winner confirmation is incomplete")
            return
        runner = [self._spec_from_state(value) for value in self._require("runner_up", "2")]
        winner_group = self._robust_group_choice([tuned, runner])
        self.state["decisions"]["winner"] = [asdict(spec) for spec in winner_group]
        baseline0 = self.phase1_base_specs()[0]
        baseline = [baseline0, baseline0.clone(phase="4", seed=1), baseline0.clone(phase="4", seed=2)]
        if not self.ensure_specs(baseline[1:]):
            print("[phase 4] three-seed shallow reference is incomplete")
            return
        phase2_runs = [
            self._spec_from_state(record["spec"])
            for record in self.state["runs"].values()
            if record["spec"]["phase"] == "2" and record["spec"]["architecture"] in {"plain_pooled", "matched_dense"}
        ]
        if not phase2_runs:
            raise RuntimeError("no dense control completed in phase 2")
        dense0 = self._rank_seed0([spec for spec in phase2_runs if spec.seed == 0])[0]
        dense = [dense0, dense0.clone(phase="4", seed=1), dense0.clone(phase="4", seed=2)]
        if not self.ensure_specs(dense[1:]):
            print("[phase 4] three-seed dense reference is incomplete")
            return
        for spec in [*baseline, *dense, *winner_group]:
            self._ensure_test(spec)
        self.state["phases"]["4"] = {"status": "complete"}
        self._save_state()

    def run(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        getattr(self, f"run_phase_{phase}")()
        print(json.dumps({
            "phase": phase,
            "status": self.state["phases"].get(phase, {}).get("status", "incomplete"),
            "new_runs": self.new_runs,
            "recorded_configurations": len(self.state["runs"]),
            "ceiling": self.ceiling,
            "state_path": self.state_path,
        }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--study-config", default="config/targeted_depth_search_v3.yaml")
    parser.add_argument("--prefix", default="nfv3_4way_depth_v3")
    parser.add_argument("--summary-dir")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    TargetedDepthStudy(
        base_config_path=args.config,
        study_config_path=args.study_config,
        prefix=args.prefix,
        summary_dir=args.summary_dir,
        execute=args.execute,
        max_new_runs=args.max_new_runs,
    ).run(args.phase)


if __name__ == "__main__":
    main()
