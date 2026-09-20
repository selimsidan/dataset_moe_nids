"""Gate diagnostics: does the gate's learned partition line up with,
diverge from, or refine the ground-truth dataset boundary? Plus expert
utilization histograms and a gate-collapse check -- the direct empirical
read on whether the "soft, task-loss-driven gate" design decision is doing
something sensible, independent of downstream classification accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from models.moe import MoEDatasetNIDS


@dataclass
class GateCollapseReport:
    mean_utilization: dict[str, float]  # dataset_name -> mean gate weight across all test rows
    collapsed_experts: list[str]  # experts whose mean utilization is below `threshold`
    is_collapsed: bool


def compute_gate_weights(model: MoEDatasetNIDS, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Runs the gate over `features` in batches, returns (n, num_datasets)."""
    model.eval()
    weights = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            chunk = torch.from_numpy(features[start : start + batch_size])
            weights.append(model.gate_weights_for(chunk).numpy())
    return np.concatenate(weights, axis=0)


def gate_weight_by_true_dataset(gate_weights: np.ndarray, dataset_names: list[str], true_dataset_name: np.ndarray) -> pd.DataFrame:
    """Mean gate weight vector per TRUE source dataset -- a diagonal-heavy
    matrix means the gate mostly recovers dataset identity (expected, but
    watch for `training.stage_c.gate_supervision: hard` or `damex` making this
    trivially true); a more diffuse matrix means the gate is blending
    experts based on task-relevant signal instead of dataset fingerprints.
    """
    rows = []
    for name in sorted(set(true_dataset_name.tolist())):
        mask = true_dataset_name == name
        if mask.sum() == 0:
            continue
        mean_weights = gate_weights[mask].mean(axis=0)
        rows.append({"true_dataset": name, **{f"gate_weight__{d}": w for d, w in zip(dataset_names, mean_weights)}})
    return pd.DataFrame(rows)


def expert_utilization(gate_weights: np.ndarray, dataset_names: list[str]) -> dict[str, float]:
    mean_utilization = gate_weights.mean(axis=0)
    return {name: float(w) for name, w in zip(dataset_names, mean_utilization)}


def detect_gate_collapse(gate_weights: np.ndarray, dataset_names: list[str], threshold: float = 0.02) -> GateCollapseReport:
    """An expert is flagged as "collapsed" if its mean utilization across
    the whole eval set is below `threshold` (default 2% -- well under the
    1/num_datasets uniform share for any config with >= 7 datasets) --
    i.e. the gate has effectively stopped routing any weight to it.
    """
    utilization = expert_utilization(gate_weights, dataset_names)
    collapsed = [name for name, w in utilization.items() if w < threshold]
    return GateCollapseReport(mean_utilization=utilization, collapsed_experts=collapsed, is_collapsed=len(collapsed) > 0)
