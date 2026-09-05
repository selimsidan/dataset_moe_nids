"""Three-seed paired aggregation for the fairness comparison matrix."""
from __future__ import annotations

import math
import os

import pandas as pd


EXPECTED_SEEDS = (0, 1, 2)
T_975_DF2 = 4.302652729911275


def method_id(row: pd.Series) -> str:
    parts = [str(row["architecture"])]
    for column in ("routing_mode", "matching_axis", "encoder_init", "stage_b_warmstart", "selection_mode"):
        value = row.get(column)
        if pd.notna(value) and value not in (None, ""):
            parts.append(f"{column}={value}")
    return "|".join(parts)


def _validate_trials(trials: pd.DataFrame, expected_seeds: tuple[int, ...]) -> pd.DataFrame:
    required = {"Trial_ID", "architecture", "seed", "split_signature"}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Trials.csv is missing fairness metadata columns: {sorted(missing)}")
    trials = trials.copy()
    trials["seed"] = trials["seed"].astype(int)
    if set(trials["seed"]) != set(expected_seeds):
        raise ValueError(f"Expected exactly paired seeds {expected_seeds}; found {sorted(trials['seed'].unique())}")
    for seed, frame in trials.groupby("seed"):
        signatures = set(frame["split_signature"].dropna())
        if len(signatures) != 1:
            raise ValueError(f"Seed {seed} mixes split signatures: {sorted(signatures)}")
    trials["method_id"] = trials.apply(method_id, axis=1)
    counts = trials.groupby("method_id")["seed"].nunique()
    incomplete = counts[counts != len(expected_seeds)]
    if len(incomplete):
        raise ValueError(f"Methods missing paired seeds: {incomplete.to_dict()}")
    return trials


def aggregate_seed_results(
    trials: pd.DataFrame,
    overall: pd.DataFrame,
    per_dataset: pd.DataFrame,
    *,
    reference_method: str,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> pd.DataFrame:
    """Mean/SD and paired macro-F1 deltas; intervals are descriptive for n=3."""
    trials = _validate_trials(trials, expected_seeds)
    metadata = trials[["Trial_ID", "seed", "method_id"]]
    overall_values = overall.merge(metadata, on="Trial_ID", validate="many_to_one")
    overall_values["dataset"] = "ALL"
    dataset_values = per_dataset.merge(metadata, on="Trial_ID", validate="many_to_one")
    values = pd.concat(
        [overall_values[["seed", "method_id", "dataset", "macro_f1"]],
         dataset_values[["seed", "method_id", "dataset", "macro_f1"]]],
        ignore_index=True,
    )
    if reference_method not in set(values["method_id"]):
        raise ValueError(f"Reference method {reference_method!r} is absent")
    reference = values[values["method_id"] == reference_method][["seed", "dataset", "macro_f1"]].rename(
        columns={"macro_f1": "reference_macro_f1"}
    )
    rows = []
    for (current_method, dataset), frame in values.groupby(["method_id", "dataset"], sort=True):
        frame = frame.sort_values("seed")
        if tuple(frame["seed"]) != expected_seeds:
            raise ValueError(f"{current_method}/{dataset} does not contain exactly seeds {expected_seeds}")
        paired = frame.merge(reference, on=["seed", "dataset"], validate="one_to_one")
        delta = paired["macro_f1"] - paired["reference_macro_f1"]
        n = len(frame)
        sd = float(frame["macro_f1"].std(ddof=1))
        delta_sd = float(delta.std(ddof=1))
        half_width = T_975_DF2 * delta_sd / math.sqrt(n)
        rows.append({
            "method_id": current_method,
            "reference_method_id": reference_method,
            "dataset": dataset,
            "seeds": "|".join(str(value) for value in expected_seeds),
            "n_seeds": n,
            "macro_f1_mean": float(frame["macro_f1"].mean()),
            "macro_f1_sd": sd,
            "paired_delta_mean": float(delta.mean()),
            "paired_delta_sd": delta_sd,
            "paired_delta_ci95_low": float(delta.mean() - half_width),
            "paired_delta_ci95_high": float(delta.mean() + half_width),
            "interval_note": "paired Student-t interval; descriptive only for n=3",
        })
    return pd.DataFrame(rows)


def write_seed_summary(output_dir: str, summary: pd.DataFrame) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "Seed_Summary.csv")
    temporary = path + ".tmp"
    summary.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path
