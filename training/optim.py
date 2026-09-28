"""Backward-compatible stage-specific optimizer configuration."""
from __future__ import annotations


def optimizer_hparams(config: dict, stage: str) -> tuple[float, float]:
    """Return ``(learning_rate, weight_decay)`` for training stage A/B/C.

    V3 stores overrides under ``training.stage_<letter>.optimizer``.  Older
    configs continue to use the historical global ``training.lr`` and
    ``training.weight_decay`` values.
    """
    key = f"stage_{stage.lower()}"
    training = config["training"]
    stage_cfg = training.get(key, {}) or {}
    optimizer = stage_cfg.get("optimizer", {}) or {}
    return (
        float(optimizer.get("lr", training["lr"])),
        float(optimizer.get("weight_decay", training.get("weight_decay", 0.0))),
    )
