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

import torch

STAGE_A_FILE = "stage_a_encoder.pt"
STAGE_B_FILE = "stage_b_expert_bank.pt"
STAGE_C_FILE = "stage_c_full.pt"
HARMONIZER_FILE = "harmonizer.pkl"


def save_harmonizer(checkpoint_dir: str, harmonizer) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, HARMONIZER_FILE), "wb") as f:
        pickle.dump(harmonizer, f)


def load_harmonizer(checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, HARMONIZER_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No fitted harmonizer at {path}. Run at least Stage A (or prepare_datasets) first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_stage_a(checkpoint_dir: str, encoder_state: dict, class_names: list[str]) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({"encoder_state": encoder_state, "class_names": class_names}, os.path.join(checkpoint_dir, STAGE_A_FILE))


def load_stage_a(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, STAGE_A_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage A checkpoint at {path}. Run Stage A first.")
    return torch.load(path, map_location="cpu")


def save_stage_b(checkpoint_dir: str, expert_bank_state: dict, dataset_names: list[str], bank_kind: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {"expert_bank_state": expert_bank_state, "dataset_names": dataset_names, "bank_kind": bank_kind},
        os.path.join(checkpoint_dir, STAGE_B_FILE),
    )


def load_stage_b(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, STAGE_B_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage B checkpoint at {path}. Run Stage B first.")
    return torch.load(path, map_location="cpu")


def save_stage_c(checkpoint_dir: str, model_state: dict, class_names: list[str], dataset_names: list[str], bank_kind: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {"model_state": model_state, "class_names": class_names, "dataset_names": dataset_names, "bank_kind": bank_kind},
        os.path.join(checkpoint_dir, STAGE_C_FILE),
    )


def load_stage_c(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, STAGE_C_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage C checkpoint at {path}. Run Stage C first.")
    return torch.load(path, map_location="cpu")


def stage_complete(checkpoint_dir: str, stage: str) -> bool:
    filename = {"A": STAGE_A_FILE, "B": STAGE_B_FILE, "C": STAGE_C_FILE}[stage]
    return os.path.isfile(os.path.join(checkpoint_dir, filename))


def _progress_path(checkpoint_dir: str, stage: str) -> str:
    return os.path.join(checkpoint_dir, f"stage_{stage.lower()}_progress.pt")


def save_progress(checkpoint_dir: str, stage: str, state: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, _progress_path(checkpoint_dir, stage))


def load_progress(checkpoint_dir: str, stage: str) -> dict | None:
    path = _progress_path(checkpoint_dir, stage)
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def clear_progress(checkpoint_dir: str, stage: str) -> None:
    path = _progress_path(checkpoint_dir, stage)
    if os.path.isfile(path):
        os.remove(path)
