"""Comparison report generator: architecture variant x per-class metric,
plus the Trial_ID-keyed tracker CSV shape (`Trials.csv` / `Overall_Metrics.csv`
/ `Per_Class_Metrics.csv`), ported from moe_nids/evaluation/report.py.

NEW here: `Per_Dataset_Metrics.csv` and `per_dataset_comparison_table`,
since the core empirical question this project answers ("which datasets get
rescued or hurt") is per-dataset, not just per-class.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from .bootstrap_ci import BootstrapCI
from .metrics import EvaluationResult
from .resource_accounting import write_resource_accounting


def comparison_table(results_by_variant: dict[str, EvaluationResult]) -> pd.DataFrame:
    """One row per class, one metric group per
    variant -- directly supports pivoting all architecture variants
    (moe_dataset_soft, moe_basic, moe_dataset_hard_gate, moe_dataset_damex, moe_dataset_adapters,
    plain_pooled, no_fusion, hard_two_stage) into one ablation table."""
    rows = []
    class_names = next(iter(results_by_variant.values())).class_names
    for class_name in class_names:
        row = {"class": class_name}
        for variant, result in results_by_variant.items():
            cm = next(c for c in result.per_class if c.class_name == class_name)
            row[f"{variant}__precision"] = cm.precision
            row[f"{variant}__recall"] = cm.recall
            row[f"{variant}__f1"] = cm.f1
            row[f"{variant}__roc_auc_ovr"] = cm.roc_auc_ovr
            row[f"{variant}__pr_auc_ovr"] = cm.pr_auc_ovr
            row[f"{variant}__support"] = cm.support
        rows.append(row)
    df = pd.DataFrame(rows)

    summary = {"class": "MACRO_AVG (headline)"}
    for variant, result in results_by_variant.items():
        summary[f"{variant}__precision"] = result.macro_precision
        summary[f"{variant}__recall"] = result.macro_recall
        summary[f"{variant}__f1"] = result.macro_f1
        summary[f"{variant}__roc_auc_ovr"] = result.roc_auc_ovr_macro
        summary[f"{variant}__pr_auc_ovr"] = result.pr_auc_ovr_macro
        summary[f"{variant}__support"] = sum(cm.support for cm in result.per_class)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    micro = {"class": "MICRO_AVG"}
    for variant, result in results_by_variant.items():
        micro[f"{variant}__precision"] = result.micro_precision
        micro[f"{variant}__recall"] = result.micro_recall
        micro[f"{variant}__f1"] = result.micro_f1
        micro[f"{variant}__roc_auc_ovr"] = result.roc_auc_ovr_micro
        micro[f"{variant}__pr_auc_ovr"] = result.pr_auc_ovr_micro
        micro[f"{variant}__support"] = sum(cm.support for cm in result.per_class)
    df = pd.concat([df, pd.DataFrame([micro])], ignore_index=True)

    ref = {"class": "WEIGHTED_AVG (reference only, never for model selection)"}
    for variant, result in results_by_variant.items():
        ref[f"{variant}__precision"] = result.weighted_precision
        ref[f"{variant}__recall"] = result.weighted_recall
        ref[f"{variant}__f1"] = result.weighted_f1
        ref[f"{variant}__roc_auc_ovr"] = result.roc_auc_ovr_weighted
        ref[f"{variant}__pr_auc_ovr"] = result.pr_auc_ovr_weighted
        ref[f"{variant}__support"] = sum(cm.support for cm in result.per_class)
    df = pd.concat([df, pd.DataFrame([ref])], ignore_index=True)
    return df


def per_dataset_comparison_table(results_by_variant_and_dataset: dict[str, dict[str, EvaluationResult]]) -> pd.DataFrame:
    """One row per dataset, one macro-F1 column per variant -- the headline
    "which datasets got rescued or hurt" table across all architecture
    variants.
    """
    dataset_names = sorted({d for by_ds in results_by_variant_and_dataset.values() for d in by_ds})
    rows = []
    for dataset_name in dataset_names:
        row = {"dataset": dataset_name}
        for variant, by_dataset in results_by_variant_and_dataset.items():
            result = by_dataset.get(dataset_name)
            row[f"{variant}__macro_f1"] = result.macro_f1 if result else float("nan")
            row[f"{variant}__weighted_f1_reference_only"] = result.weighted_f1 if result else float("nan")
            row[f"{variant}__macro_precision"] = result.macro_precision if result else float("nan")
            row[f"{variant}__macro_recall"] = result.macro_recall if result else float("nan")
            row[f"{variant}__micro_precision"] = result.micro_precision if result else float("nan")
            row[f"{variant}__micro_recall"] = result.micro_recall if result else float("nan")
            row[f"{variant}__micro_f1"] = result.micro_f1 if result else float("nan")
            row[f"{variant}__weighted_precision"] = result.weighted_precision if result else float("nan")
            row[f"{variant}__weighted_recall"] = result.weighted_recall if result else float("nan")
            row[f"{variant}__weighted_f1"] = result.weighted_f1 if result else float("nan")
            row[f"{variant}__accuracy"] = result.accuracy if result else float("nan")
            row[f"{variant}__balanced_accuracy"] = result.balanced_accuracy if result else float("nan")
            row[f"{variant}__roc_auc_ovr_macro"] = result.roc_auc_ovr_macro if result else float("nan")
            row[f"{variant}__roc_auc_ovr_weighted"] = result.roc_auc_ovr_weighted if result else float("nan")
            row[f"{variant}__roc_auc_ovr_micro"] = result.roc_auc_ovr_micro if result else float("nan")
            row[f"{variant}__pr_auc_ovr_macro"] = result.pr_auc_ovr_macro if result else float("nan")
            row[f"{variant}__pr_auc_ovr_weighted"] = result.pr_auc_ovr_weighted if result else float("nan")
            row[f"{variant}__pr_auc_ovr_micro"] = result.pr_auc_ovr_micro if result else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_tracker_csvs(
    output_dir: str,
    trial_id: str,
    config: dict,
    results_by_variant: dict[str, EvaluationResult],
    ci_by_variant: dict[str, list[BootstrapCI]] | None = None,
    per_dataset_by_variant: dict[str, dict[str, EvaluationResult]] | None = None,
    resource_rows: list[dict] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    trials_rows = [
        {
            "Trial_ID": trial_id,
            "timestamp_utc": timestamp,
            "architecture": variant,
            "routing_mode": config["model"].get("gate", {}).get("routing", "dense"),
            "gate_supervision": config["training"].get("stage_c", {}).get("gate_supervision"),
            "expert_update_policy": config["training"].get("stage_c", {}).get("expert_update_policy", "all"),
            "run_name": config.get("run_name"),
            "active_datasets": ",".join(config["data"]["active_datasets"]),
            "matching_axis": config.get("model", {}).get("dense_match", {}).get("axis") if variant == "matched_dense" else None,
            "encoder_init": (
                config.get("training", {}).get("baseline", {}).get("encoder_init")
                if variant in {"plain_pooled", "matched_dense", "no_fusion", "hard_two_stage"} else None
            ),
            "stage_b_warmstart": (
                config.get("training", {}).get("baseline", {}).get("stage_b_warmstart")
                if variant in {"plain_pooled", "matched_dense"} else None
            ),
            "selection_mode": config.get("training", {}).get("selection_mode", "fixed_epochs"),
            "seed": config.get("seed", 0),
            "split_signature": resource_rows[0].get("split_signature") if resource_rows else None,
            "stage_a_checkpoint_sha256": resource_rows[0].get("stage_a_checkpoint_sha256") if resource_rows else None,
        }
        for variant in results_by_variant
    ]
    pd.DataFrame(trials_rows).to_csv(os.path.join(output_dir, "Trials.csv"), index=False)

    overall_rows = [
        {
            "Trial_ID": trial_id,
            "architecture": variant,
            "macro_f1": result.macro_f1,
            "weighted_f1_reference_only": result.weighted_f1,
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "macro_precision": result.macro_precision,
            "macro_recall": result.macro_recall,
            "micro_precision": result.micro_precision,
            "micro_recall": result.micro_recall,
            "micro_f1": result.micro_f1,
            "weighted_precision": result.weighted_precision,
            "weighted_recall": result.weighted_recall,
            "weighted_f1": result.weighted_f1,
            "roc_auc_ovr_macro": result.roc_auc_ovr_macro,
            "roc_auc_ovr_weighted": result.roc_auc_ovr_weighted,
            "roc_auc_ovr_micro": result.roc_auc_ovr_micro,
            "pr_auc_ovr_macro": result.pr_auc_ovr_macro,
            "pr_auc_ovr_weighted": result.pr_auc_ovr_weighted,
            "pr_auc_ovr_micro": result.pr_auc_ovr_micro,
            "roc_auc_method": "exact",
            "pr_auc_method": "exact",
        }
        for variant, result in results_by_variant.items()
    ]
    pd.DataFrame(overall_rows).to_csv(os.path.join(output_dir, "Overall_Metrics.csv"), index=False)

    per_class_rows = []
    for variant, result in results_by_variant.items():
        ci_lookup = {}
        if ci_by_variant and variant in ci_by_variant:
            for ci in ci_by_variant[variant]:
                ci_lookup[(ci.class_name, ci.metric)] = ci
        for cm in result.per_class:
            recall_ci = ci_lookup.get((cm.class_name, "recall"))
            f1_ci = ci_lookup.get((cm.class_name, "f1"))
            per_class_rows.append(
                {
                    "Trial_ID": trial_id,
                    "architecture": variant,
                    "class": cm.class_name,
                    "support": cm.support,
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "f1": cm.f1,
                    "roc_auc_ovr": cm.roc_auc_ovr,
                    "pr_auc_ovr": cm.pr_auc_ovr,
                    "recall_ci_low": recall_ci.ci_low if recall_ci else None,
                    "recall_ci_high": recall_ci.ci_high if recall_ci else None,
                    "f1_ci_low": f1_ci.ci_low if f1_ci else None,
                    "f1_ci_high": f1_ci.ci_high if f1_ci else None,
                    "is_low_sample": recall_ci.is_low_sample if recall_ci else cm.support < config["evaluation"]["low_sample_threshold"],
                }
            )
    pd.DataFrame(per_class_rows).to_csv(os.path.join(output_dir, "Per_Class_Metrics.csv"), index=False)

    if per_dataset_by_variant:
        per_dataset_rows = []
        for variant, by_dataset in per_dataset_by_variant.items():
            for dataset_name, result in by_dataset.items():
                per_dataset_rows.append(
                    {
                        "Trial_ID": trial_id,
                        "architecture": variant,
                        "dataset": dataset_name,
                        "macro_f1": result.macro_f1,
                        "weighted_f1_reference_only": result.weighted_f1,
                        "accuracy": result.accuracy,
                        "balanced_accuracy": result.balanced_accuracy,
                        "macro_precision": result.macro_precision,
                        "macro_recall": result.macro_recall,
                        "micro_precision": result.micro_precision,
                        "micro_recall": result.micro_recall,
                        "micro_f1": result.micro_f1,
                        "weighted_precision": result.weighted_precision,
                        "weighted_recall": result.weighted_recall,
                        "weighted_f1": result.weighted_f1,
                        "roc_auc_ovr_macro": result.roc_auc_ovr_macro,
                        "roc_auc_ovr_weighted": result.roc_auc_ovr_weighted,
                        "roc_auc_ovr_micro": result.roc_auc_ovr_micro,
                        "pr_auc_ovr_macro": result.pr_auc_ovr_macro,
                        "pr_auc_ovr_weighted": result.pr_auc_ovr_weighted,
                        "pr_auc_ovr_micro": result.pr_auc_ovr_micro,
                        "roc_auc_method": "exact",
                        "pr_auc_method": "exact",
                    }
                )
        pd.DataFrame(per_dataset_rows).to_csv(os.path.join(output_dir, "Per_Dataset_Metrics.csv"), index=False)

    if resource_rows:
        write_resource_accounting(output_dir, resource_rows)
