"""Checkpoint I/O for the three training stages. Each stage is independently
resumable/re-runnable: Stage C can load an existing Stage A+B checkpoint
without re-running A or B, per config `training.stages`.

Adapted from moe_nids/training/checkpoint.py: Stage B here persists a
dataset-expert bank (full DatasetExpertBank or AdapterExpertBank, tagged by
`bank_kind`) instead of a class-expert bank, and Stage C persists the gate
alongside dataset_names (not just class names), since the gate output
dimension is `len(active_datasets)`, derived at runtime.
"""
from __future__ import annotations

import os
import pickle
import hashlib
import json
import time

import torch

from models.encoder import resolve_encoder_config
from models.representation_losses import representation_config

STAGE_A_FILE = "stage_a_encoder.pt"
STAGE_B_FILE = "stage_b_expert_bank.pt"
STAGE_C_FILE = "stage_c_full.pt"
HARMONIZER_FILE = "harmonizer.pkl"
HARD_ROUTER_FILE = "hard_two_stage_router.pt"
HARD_CLASSIFIERS_FILE = "hard_two_stage_classifiers.pt"
BASELINE_MODEL_FILE = "baseline_full.pt"
BASELINE_STAGE_B_FILE = "baseline_stage_b.pt"


def _atomic_torch_save(value, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def _verify_torch_checkpoint(path: str) -> None:
    """Reopen a newly committed checkpoint before its progress file is cleared."""
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Checkpoint verification failed for {path}: {exc}") from exc


def save_harmonizer(checkpoint_dir: str, harmonizer) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, HARMONIZER_FILE)
    temporary = path + ".tmp"
    with open(temporary, "wb") as f:
        pickle.dump(harmonizer, f)
    os.replace(temporary, path)


def load_harmonizer(checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, HARMONIZER_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No fitted harmonizer at {path}. Run at least Stage A (or prepare_datasets) first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def split_signature_for_data(data) -> str:
    """Stable paired-run signature without materializing additional arrays."""
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "class_names": list(data.class_names),
        "active_datasets": list(data.active_datasets),
        "feature_width": int(data.train.features.shape[1]),
    }, sort_keys=True).encode())
    for split_name in ("train", "val", "test"):
        split = getattr(data, split_name)
        digest.update(split_name.encode())
        digest.update(memoryview(split.class_idx).cast("B"))
        digest.update(memoryview(split.dataset_idx).cast("B"))
        for start in range(0, len(split.features), 65_536):
            block = split.features[start : start + 65_536]
            digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def stage_a_metadata(
    config: dict,
    data,
    *,
    split_signature: str | None = None,
    feature_columns: list[str] | None = None,
    encoder_role: str = "encoder",
    encoder_config: dict | None = None,
) -> dict:
    encoder_cfg = encoder_config or resolve_encoder_config(config["model"], encoder_role)
    if feature_columns is None:
        harmonizer = getattr(data, "harmonizer", None)
        feature_columns = list(getattr(harmonizer, "output_columns", []))
        if not feature_columns:
            feature_columns = [f"feature_{index}" for index in range(data.train.features.shape[1])]
    encoder_metadata = {
        "input_dim": int(data.train.features.shape[1]),
        "hidden_dims": list(encoder_cfg.get("hidden_dims", [])),
        "latent_dim": int(config["model"]["latent_dim"]),
        "activation": encoder_cfg.get("activation", "relu"),
        "dropout": float(encoder_cfg.get("dropout", 0.1)),
    }
    if encoder_cfg.get("kind", "mlp") != "mlp":
        encoder_metadata.update({
            "kind": encoder_cfg["kind"],
            "width": int(encoder_cfg.get("width", 128)),
            "blocks": int(encoder_cfg.get("blocks", 2)),
            "expansion": float(encoder_cfg.get("expansion", 2.0)),
            "normalization": encoder_cfg.get("normalization", "layernorm"),
            "dropout_second": float(encoder_cfg.get("dropout_second", 0.0)),
        })
    metadata = {
        "format_version": 2,
        "seed": int(config.get("seed", 0)),
        "encoder": encoder_metadata,
        "class_names": list(data.class_names),
        "active_datasets": list(data.active_datasets),
        "feature_columns": list(feature_columns),
        "split_signature": split_signature or split_signature_for_data(data),
    }
    role_aware = (
        "stage_a_encoder_role" in config.get("training", {})
        or bool(config.get("training", {}).get("representation_by_role"))
    )
    if role_aware:
        metadata["encoder_role"] = encoder_role
    return metadata


def save_stage_a(
    checkpoint_dir: str,
    encoder_state: dict,
    class_names: list[str],
    metadata: dict | None = None,
    representation_state: dict | None = None,
    representation_config: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, STAGE_A_FILE)
    _atomic_torch_save(
        {
            "encoder_state": encoder_state,
            "class_names": class_names,
            "metadata": metadata,
            "representation_state": representation_state,
            "representation_config": representation_config,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_a(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, STAGE_A_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage A checkpoint at {path}. Run Stage A first.")
    return torch.load(path, map_location="cpu")


def resolve_stage_a_path(config: dict, encoder_role: str = "encoder") -> str:
    baseline_cfg = config.get("training", {}).get("baseline", {})
    training_cfg = config.get("training", {})
    if encoder_role == "private_encoder":
        source = training_cfg.get("private_stage_a_checkpoint")
    else:
        source = None
    source = source or training_cfg.get("stage_a_checkpoint") or baseline_cfg.get("stage_a_checkpoint")
    if source:
        return source if source.endswith(".pt") else os.path.join(source, STAGE_A_FILE)
    return os.path.join(config["training"]["checkpoint_dir"], STAGE_A_FILE)


def load_validated_stage_a(
    config: dict,
    expected_metadata: dict,
    encoder_role: str = "encoder",
) -> dict:
    path = resolve_stage_a_path(config, encoder_role)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage-A encoder checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu")
    actual = checkpoint.get("metadata")
    if actual is None:
        raise ValueError(
            f"Stage-A checkpoint {path} predates the fairness metadata contract; regenerate it"
        )
    mismatches = {
        key: {"expected": expected_metadata.get(key), "actual": actual.get(key)}
        for key in expected_metadata
        if actual.get(key) != expected_metadata.get(key)
    }
    if mismatches:
        raise ValueError(f"Incompatible Stage-A checkpoint {path}: {mismatches}")
    role_aware = bool(config.get("training", {}).get("representation_by_role"))
    configured_representation = (
        representation_config(config, encoder_role)
        if role_aware
        else config.get("training", {}).get("representation", {})
    )
    configured_objective = configured_representation.get("objective", "ce")
    configured_sampling = configured_representation.get("sampling", "legacy")
    if role_aware or (configured_objective, configured_sampling) != ("ce", "legacy"):
        checkpoint_representation = checkpoint.get("representation_config")
        if checkpoint_representation is None:
            raise ValueError(
                f"Stage-A checkpoint {path} does not record a representation objective; "
                "regenerate it for this non-legacy configuration"
            )
        mismatched_representation = {
            key: {"expected": value, "actual": checkpoint_representation.get(key)}
            for key, value in configured_representation.items()
            if checkpoint_representation.get(key) != value
        }
        if mismatched_representation:
            raise ValueError(
                f"Incompatible Stage-A representation checkpoint {path}: "
                f"{mismatched_representation}"
            )
    return checkpoint


def save_stage_b(
    checkpoint_dir: str,
    expert_bank_state: dict,
    dataset_names: list[str],
    bank_kind: str,
    training_summary: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, STAGE_B_FILE)
    _atomic_torch_save(
        {
            "expert_bank_state": expert_bank_state,
            "dataset_names": dataset_names,
            "bank_kind": bank_kind,
            "training_summary": training_summary,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_b(checkpoint_dir: str) -> dict:
    path = checkpoint_dir if checkpoint_dir.endswith(".pt") else os.path.join(checkpoint_dir, STAGE_B_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage B checkpoint at {path}. Run Stage B first.")
    return torch.load(path, map_location="cpu")


def resolve_stage_b_path(config: dict) -> str:
    source = config.get("training", {}).get("stage_b_checkpoint")
    if source:
        return source if source.endswith(".pt") else os.path.join(source, STAGE_B_FILE)
    return os.path.join(config["training"]["checkpoint_dir"], STAGE_B_FILE)


def load_configured_stage_b(
    config: dict,
    *,
    expected_dataset_names: list[str] | None = None,
    expected_bank_kind: str | None = None,
) -> dict:
    """Load the selected Stage-B artifact and reject obvious cache mix-ups.

    Historical checkpoints remain readable when no expectations are supplied.
    V3 call sites provide both expectations before attempting to load weights,
    which produces a useful error instead of a later state-dict shape failure.
    """
    checkpoint = load_stage_b(resolve_stage_b_path(config))
    mismatches = {}
    if expected_dataset_names is not None:
        actual = list(checkpoint.get("dataset_names", []))
        expected = list(expected_dataset_names)
        if actual != expected:
            mismatches["dataset_names"] = {"expected": expected, "actual": actual}
    if expected_bank_kind is not None:
        actual = checkpoint.get("bank_kind")
        if actual != expected_bank_kind:
            mismatches["bank_kind"] = {
                "expected": expected_bank_kind,
                "actual": actual,
            }
    if mismatches:
        raise ValueError(
            f"Incompatible Stage-B checkpoint {resolve_stage_b_path(config)}: {mismatches}"
        )
    return checkpoint


def save_stage_c(
    checkpoint_dir: str,
    model_state: dict,
    class_names: list[str],
    dataset_names: list[str],
    bank_kind: str,
    training_summary: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, STAGE_C_FILE)
    _atomic_torch_save(
        {
            "model_state": model_state,
            "class_names": class_names,
            "dataset_names": dataset_names,
            "bank_kind": bank_kind,
            "training_summary": training_summary,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_c(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, STAGE_C_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage C checkpoint at {path}. Run Stage C first.")
    return torch.load(path, map_location="cpu")


def stage_complete(checkpoint_dir: str, stage: str) -> bool:
    filename = {"A": STAGE_A_FILE, "B": STAGE_B_FILE, "C": STAGE_C_FILE}[stage]
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(path):
        return False
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint] Stage {stage} final checkpoint is unreadable: {path}: {exc}")
        return False
    return True


def hard_stage_complete(checkpoint_dir: str, stage: str) -> bool:
    filename = {"router": HARD_ROUTER_FILE, "classifiers": HARD_CLASSIFIERS_FILE}[stage]
    return os.path.isfile(os.path.join(checkpoint_dir, filename))


def _progress_path(checkpoint_dir: str, stage: str) -> str:
    return os.path.join(checkpoint_dir, f"stage_{stage.lower()}_progress.pt")


def save_progress(checkpoint_dir: str, stage: str, state: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_torch_save(state, _progress_path(checkpoint_dir, stage))


def load_progress(checkpoint_dir: str, stage: str) -> dict | None:
    path = _progress_path(checkpoint_dir, stage)
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        quarantine = f"{path}.corrupt-{time.time_ns()}"
        os.replace(path, quarantine)
        print(
            f"[checkpoint] unreadable Stage {stage} progress was quarantined at "
            f"{quarantine}; restarting the stage from its last valid input: {exc}",
            flush=True,
        )
        return None


def clear_progress(checkpoint_dir: str, stage: str) -> None:
    path = _progress_path(checkpoint_dir, stage)
    if os.path.isfile(path):
        os.remove(path)
