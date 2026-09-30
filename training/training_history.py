"""Atomic epoch histories and learning curves for long-running OOC studies."""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd


TRAINING_HISTORY_FILE = "Training_History.csv"
HISTORY_KEYS = ["seed", "stage", "encoder_role", "dataset", "epoch"]
HISTORY_COLUMNS = [
    *HISTORY_KEYS,
    "objective",
    "history_source",
    "learning_rate",
    "rows",
    "replay_rows",
    "optimizer_steps",
    "examples_seen",
    "epoch_seconds",
    "train_total_loss",
    "train_ce_loss",
    "train_owned_ce_loss",
    "train_replay_ce_loss",
    "train_representation_loss",
    "train_balance_penalty",
    "train_dataset_aux_loss",
    "train_anchor_penalty",
    "train_reliability_penalty",
    "val_macro_f1",
    "best_val_macro_f1",
    "improved",
    "patience_left",
]


def training_history_enabled(config: dict) -> bool:
    return bool(config.get("training", {}).get("save_epoch_history", False))


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    if frame.empty:
        return frame[HISTORY_COLUMNS]
    for column in ("stage", "encoder_role", "dataset", "objective", "history_source"):
        frame[column] = frame[column].fillna("").astype(str)
    for column in ("seed", "epoch"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame = frame[HISTORY_COLUMNS]
    return frame.sort_values(HISTORY_KEYS, kind="stable").reset_index(drop=True)


def history_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, TRAINING_HISTORY_FILE)


def read_training_history(checkpoint_dir: str) -> pd.DataFrame:
    path = history_path(checkpoint_dir)
    if not os.path.isfile(path):
        return _empty_history()
    return _normalize_history(pd.read_csv(path))


def append_training_history(config: dict, row: dict[str, Any]) -> None:
    """Atomically upsert one completed epoch when history is opted in."""
    if not training_history_enabled(config):
        return
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    current = read_training_history(checkpoint_dir)
    value = {column: row.get(column, np.nan) for column in HISTORY_COLUMNS}
    value["seed"] = int(row.get("seed", config.get("seed", 0)))
    value["history_source"] = row.get("history_source", "structured")
    updated = pd.concat([current, pd.DataFrame([value])], ignore_index=True)
    updated = updated.drop_duplicates(HISTORY_KEYS, keep="last")
    updated = _normalize_history(updated)
    path = history_path(checkpoint_dir)
    temporary = path + ".tmp"
    updated.to_csv(temporary, index=False)
    os.replace(temporary, path)


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STAGE_A = re.compile(
    rf"\[Stage A/ooc\] epoch (?P<epoch>\d+): objective=(?P<objective>\S+) "
    rf"CE=(?P<ce>{_FLOAT}) metric=(?P<metric>{_FLOAT}) "
    rf"total=(?P<total>{_FLOAT}) rows=(?P<rows>[\d,]+)"
)
_STAGE_B = re.compile(
    rf"\[Stage B/ooc:(?P<dataset>.+?)\] epoch (?P<epoch>\d+): "
    rf"CE=(?P<ce>{_FLOAT}).*?(?:owned_)?rows=(?P<rows>[\d,]+)"
)
_STAGE_C = re.compile(
    rf"\[Stage C/ooc\] epoch (?P<epoch>\d+): CE=(?P<ce>{_FLOAT}) "
    rf"balance=(?P<balance>{_FLOAT}) aux=(?P<aux>{_FLOAT}) "
    rf"anchor=(?P<anchor>{_FLOAT}).* rows=(?P<rows>[\d,]+)"
)
_STAGE_C_VAL = re.compile(
    rf"\[Stage C/ooc\] epoch (?P<epoch>\d+): val_macro_f1=(?P<val>{_FLOAT}) "
    rf"best=(?P<best>{_FLOAT}) patience_left=(?P<patience>-?\d+)"
)


def reconstruct_training_history(
    log_path: str,
    *,
    seed: int,
    stage: str,
    encoder_role: str,
    learning_rate: float | None = None,
) -> pd.DataFrame:
    """Recover plot-compatible epoch metrics from historical ``train.log``."""
    if not os.path.isfile(log_path):
        return _empty_history()
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    with open(log_path, errors="replace") as handle:
        for line in handle:
            match = _STAGE_A.search(line) if stage == "A" else None
            if match:
                values = match.groupdict()
                epoch = int(values["epoch"])
                rows[(epoch, "ALL")] = {
                    "seed": seed, "stage": "A", "encoder_role": encoder_role,
                    "dataset": "ALL", "epoch": epoch,
                    "objective": values["objective"], "history_source": "legacy_log",
                    "learning_rate": learning_rate,
                    "rows": int(values["rows"].replace(",", "")),
                    "train_ce_loss": float(values["ce"]),
                    "train_representation_loss": float(values["metric"]),
                    "train_total_loss": float(values["total"]),
                }
                continue
            match = _STAGE_B.search(line) if stage == "B" else None
            if match:
                values = match.groupdict(); epoch = int(values["epoch"])
                dataset = values["dataset"]
                rows[(epoch, dataset)] = {
                    "seed": seed, "stage": "B", "encoder_role": encoder_role,
                    "dataset": dataset, "epoch": epoch, "objective": "ce",
                    "history_source": "legacy_log", "learning_rate": learning_rate,
                    "rows": int(values["rows"].replace(",", "")),
                    "train_ce_loss": float(values["ce"]),
                    "train_total_loss": float(values["ce"]),
                }
                continue
            match = _STAGE_C.search(line) if stage == "C" else None
            if match:
                values = match.groupdict(); epoch = int(values["epoch"])
                rows[(epoch, "ALL")] = {
                    "seed": seed, "stage": "C", "encoder_role": encoder_role,
                    "dataset": "ALL", "epoch": epoch, "objective": "joint",
                    "history_source": "legacy_log", "learning_rate": learning_rate,
                    "rows": int(values["rows"].replace(",", "")),
                    "train_ce_loss": float(values["ce"]),
                    "train_balance_penalty": float(values["balance"]),
                    "train_dataset_aux_loss": float(values["aux"]),
                    "train_anchor_penalty": float(values["anchor"]),
                }
                continue
            match = _STAGE_C_VAL.search(line) if stage == "C" else None
            if match:
                values = match.groupdict(); epoch = int(values["epoch"])
                row = rows.setdefault((epoch, "ALL"), {
                    "seed": seed, "stage": "C", "encoder_role": encoder_role,
                    "dataset": "ALL", "epoch": epoch, "objective": "joint",
                    "history_source": "legacy_log", "learning_rate": learning_rate,
                })
                row.update({
                    "val_macro_f1": float(values["val"]),
                    "best_val_macro_f1": float(values["best"]),
                    "patience_left": int(values["patience"]),
                })
    return _normalize_history(pd.DataFrame(rows.values())) if rows else _empty_history()


def load_or_reconstruct_history(
    checkpoint_dir: str,
    *,
    seed: int,
    stage: str,
    encoder_role: str,
    learning_rate: float | None = None,
) -> pd.DataFrame:
    reconstructed = reconstruct_training_history(
        os.path.join(checkpoint_dir, "train.log"), seed=seed, stage=stage,
        encoder_role=encoder_role, learning_rate=learning_rate,
    )
    structured = read_training_history(checkpoint_dir)
    if not structured.empty:
        structured = structured[
            (structured["stage"] == stage)
            & (structured["encoder_role"] == encoder_role)
        ].copy()
    if structured.empty:
        return reconstructed
    combined = pd.concat([reconstructed, structured], ignore_index=True)
    combined = combined.drop_duplicates(HISTORY_KEYS, keep="last")
    return _normalize_history(combined)


def summarize_training_history(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage", "encoder_role", "dataset", "epoch", "objective"]
    numeric = [
        column for column in history.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "epoch"}
    ]
    grouped = history.groupby(keys, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        value if isinstance(value, str) else "__".join(part for part in value if part)
        for value in grouped.columns
    ]
    return grouped


def _save_figure(fig, output_dir: str, filename: str) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed_training_history(
    history: pd.DataFrame,
    output_dir: str,
    *,
    selected_epoch: int | None = None,
) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    stage_a = history[history["stage"] == "A"]
    if not stage_a.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for role, group in stage_a.groupby("encoder_role"):
            group = group.sort_values("epoch")
            axes[0].plot(group["epoch"], group["train_ce_loss"], marker="o", label=role)
            axes[1].plot(group["epoch"], group["train_total_loss"], marker="o", label=role)
            if group["train_representation_loss"].notna().any():
                axes[1].plot(
                    group["epoch"], group["train_representation_loss"], linestyle="--",
                    label=f"{role} representation",
                )
        axes[0].set_title("Stage A classification loss"); axes[0].set_ylabel("CE")
        axes[1].set_title("Stage A total and representation losses")
        for axis in axes:
            axis.set_xlabel("Epoch"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_A_Learning_Curves.png"))

    stage_b = history[history["stage"] == "B"]
    if not stage_b.empty:
        fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for dataset, group in stage_b.groupby("dataset"):
            group = group.sort_values("epoch")
            axis.plot(group["epoch"], group["train_ce_loss"], marker="o", label=dataset)
        axis.set_title("Stage B specialist training loss"); axis.set_xlabel("Epoch")
        axis.set_ylabel("Weighted CE"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_B_Learning_Curves.png"))

    stage_c = history[history["stage"] == "C"].sort_values("epoch")
    if not stage_c.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for column, label in (
            ("train_total_loss", "total"), ("train_ce_loss", "CE"),
            ("train_balance_penalty", "balance"),
            ("train_dataset_aux_loss", "dataset auxiliary"),
            ("train_anchor_penalty", "anchor"),
            ("train_reliability_penalty", "class reliability"),
        ):
            if stage_c[column].notna().any():
                axes[0].plot(stage_c["epoch"], stage_c[column], marker="o", label=label)
        axes[0].set_title("Stage C training objectives"); axes[0].set_ylabel("Loss")
        axes[1].plot(
            stage_c["epoch"], stage_c["val_macro_f1"], marker="o", label="validation macro-F1"
        )
        axes[1].plot(
            stage_c["epoch"], stage_c["best_val_macro_f1"], linestyle="--", label="best so far"
        )
        axes[1].set_title("Stage C validation performance"); axes[1].set_ylim(0, 1)
        if selected_epoch is not None:
            for axis in axes:
                axis.axvline(selected_epoch, color="black", linestyle=":", label="selected epoch")
        for axis in axes:
            axis.set_xlabel("Epoch"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_C_Learning_Curves.png"))
    return written


def plot_cross_seed_training_history(summary: pd.DataFrame, output_dir: str) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    specifications = (
        ("A", "encoder_role", "train_total_loss", "Stage A total loss", "Stage_A_3Seed.png"),
        ("B", "dataset", "train_ce_loss", "Stage B specialist CE", "Stage_B_3Seed.png"),
        ("C", "dataset", "val_macro_f1", "Stage C validation macro-F1", "Stage_C_3Seed.png"),
    )
    for stage, series_column, metric, title, filename in specifications:
        subset = summary[summary["stage"] == stage]
        mean_column = f"{metric}__mean"; std_column = f"{metric}__std"
        if subset.empty or mean_column not in subset or subset[mean_column].isna().all():
            continue
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for label, group in subset.groupby(series_column):
            group = group.sort_values("epoch")
            mean = group[mean_column].to_numpy(dtype=float)
            std = group[std_column].fillna(0).to_numpy(dtype=float)
            epochs = group["epoch"].to_numpy(dtype=int)
            axis.plot(epochs, mean, marker="o", label=str(label))
            axis.fill_between(epochs, mean - std, mean + std, alpha=0.18)
        axis.set_title(f"{title}, mean ± sample SD")
        axis.set_xlabel("Epoch"); axis.set_ylabel(metric.replace("_", " "))
        if metric == "val_macro_f1":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, filename))
    return written
