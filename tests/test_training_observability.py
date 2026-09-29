from __future__ import annotations

import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from training.logging_utils import run_streaming_logged
from training.training_history import (
    append_training_history,
    load_or_reconstruct_history,
    plot_cross_seed_training_history,
    plot_seed_training_history,
    read_training_history,
    reconstruct_training_history,
    summarize_training_history,
)


def test_streaming_command_mirrors_stdout_stderr_to_live_and_log(tmp_path):
    output = io.StringIO(); log_path = tmp_path / "stream.log"
    result = run_streaming_logged(
        [sys.executable, "-u", "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr)"],
        str(log_path), heartbeat_seconds=1, output_stream=output, label="mirror-test",
    )
    assert result.returncode == 0
    assert output.getvalue() == log_path.read_text()
    assert "stdout" in output.getvalue()
    assert "stderr" in output.getvalue()
    assert "completed return_code=0" in output.getvalue()


def test_streaming_command_emits_heartbeat_for_silent_child(tmp_path):
    output = io.StringIO()
    run_streaming_logged(
        [sys.executable, "-u", "-c", "import time; time.sleep(0.12); print('awake')"],
        str(tmp_path / "heartbeat.log"), heartbeat_seconds=0.03,
        output_stream=output, label="heartbeat-test",
    )
    assert "[heartbeat] heartbeat-test" in output.getvalue()
    assert "awake" in output.getvalue()


def test_streaming_command_reports_tail_and_raises_on_failure(tmp_path):
    output = io.StringIO()
    with pytest.raises(subprocess.CalledProcessError):
        run_streaming_logged(
            [sys.executable, "-u", "-c", "print('important failure context'); raise SystemExit(7)"],
            str(tmp_path / "failure.log"), heartbeat_seconds=1,
            output_stream=output, label="failure-test",
        )
    assert "failed return_code=7" in output.getvalue()
    assert "[tail] important failure context" in output.getvalue()


def test_epoch_history_is_opt_in_atomic_and_resume_safe(tmp_path):
    disabled = {"seed": 2, "training": {"checkpoint_dir": str(tmp_path)}}
    append_training_history(disabled, {
        "stage": "A", "encoder_role": "gate_encoder", "dataset": "ALL", "epoch": 1,
    })
    assert not (tmp_path / "Training_History.csv").exists()

    enabled = {
        "seed": 2,
        "training": {"checkpoint_dir": str(tmp_path), "save_epoch_history": True},
    }
    base = {
        "stage": "A", "encoder_role": "gate_encoder", "dataset": "ALL",
        "epoch": 1, "objective": "ce", "train_ce_loss": 0.8,
    }
    append_training_history(enabled, base)
    append_training_history(enabled, {**base, "train_ce_loss": 0.7})
    history = read_training_history(str(tmp_path))
    assert len(history) == 1
    assert history.iloc[0]["train_ce_loss"] == pytest.approx(0.7)
    assert history.iloc[0]["history_source"] == "structured"


def test_legacy_log_reconstruction_recovers_available_metrics(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "[Stage C/ooc] epoch 1: CE=0.900000 balance=0.100000 aux=0.200000 "
        "anchor=0.000000 policy=all gate_supervision=light_aux rows=1,024\n"
        "[Stage C/ooc] epoch 1: val_macro_f1=0.600000 best=0.600000 patience_left=5\n"
    )
    history = reconstruct_training_history(
        str(log), seed=0, stage="C", encoder_role="full_model", learning_rate=0.0003
    )
    assert len(history) == 1
    row = history.iloc[0]
    assert row["rows"] == 1024
    assert row["train_ce_loss"] == pytest.approx(0.9)
    assert row["val_macro_f1"] == pytest.approx(0.6)
    assert row["history_source"] == "legacy_log"


def test_training_plots_are_saved_for_seed_and_cross_seed(tmp_path):
    rows = []
    for seed in (0, 1, 2):
        rows.extend([
            {"seed": seed, "stage": "A", "encoder_role": "gate_encoder", "dataset": "ALL",
             "epoch": 1, "objective": "ce", "train_ce_loss": 0.8 + seed / 100,
             "train_total_loss": 0.8 + seed / 100, "history_source": "structured"},
            {"seed": seed, "stage": "B", "encoder_role": "private_expert", "dataset": "A",
             "epoch": 1, "objective": "ce", "train_ce_loss": 0.7 + seed / 100,
             "train_total_loss": 0.7 + seed / 100, "history_source": "structured"},
            {"seed": seed, "stage": "C", "encoder_role": "full_model", "dataset": "ALL",
             "epoch": 1, "objective": "joint", "train_ce_loss": 0.6 + seed / 100,
             "train_total_loss": 0.62 + seed / 100, "train_balance_penalty": 0.1,
             "train_dataset_aux_loss": 0.1, "train_anchor_penalty": 0.0,
             "val_macro_f1": 0.65 + seed / 100, "best_val_macro_f1": 0.65 + seed / 100,
             "history_source": "structured"},
        ])
    history = pd.DataFrame(rows)
    for column in (
        "learning_rate", "rows", "optimizer_steps", "examples_seen", "epoch_seconds",
        "train_representation_loss", "improved", "patience_left",
    ):
        history[column] = np.nan
    seed_figures = plot_seed_training_history(
        history[history["seed"] == 0], str(tmp_path / "seed"), selected_epoch=1
    )
    summary = summarize_training_history(history)
    cross_figures = plot_cross_seed_training_history(summary, str(tmp_path / "summary"))
    assert len(seed_figures) == 3
    assert len(cross_figures) == 3
    assert all(os.path.isfile(path) for path in [*seed_figures, *cross_figures])


def test_structured_history_overrides_reconstructed_duplicate(tmp_path):
    (tmp_path / "train.log").write_text(
        "[Stage A/ooc] epoch 1: objective=ce CE=0.900000 metric=0.000000 "
        "total=0.900000 rows=100\n"
    )
    config = {
        "seed": 0,
        "training": {"checkpoint_dir": str(tmp_path), "save_epoch_history": True},
    }
    append_training_history(config, {
        "stage": "A", "encoder_role": "gate_encoder", "dataset": "ALL", "epoch": 1,
        "objective": "ce", "train_ce_loss": 0.8, "train_total_loss": 0.8,
    })
    history = load_or_reconstruct_history(
        str(tmp_path), seed=0, stage="A", encoder_role="gate_encoder"
    )
    assert len(history) == 1
    assert history.iloc[0]["train_ce_loss"] == pytest.approx(0.8)
    assert history.iloc[0]["history_source"] == "structured"
