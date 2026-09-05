"""Chunked predictions and comprehensive reports for full-data MoE runs."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.baselines import NoFusionModel
from evaluation.resource_accounting import resource_profile, stage_a_checkpoint_hash, write_resource_accounting
from training.checkpoint import resolve_stage_a_path
from training.out_of_core_data import OutOfCoreContext


def _update_roc_histograms(
    positive: np.ndarray,
    negative: np.ndarray,
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Accumulate bounded-memory one-vs-rest ROC score histograms."""
    bins = positive.shape[1]
    score_bins = np.clip((probabilities * (bins - 1)).astype(np.int64), 0, bins - 1)
    for class_index in range(positive.shape[0]):
        is_positive = truth == class_index
        positive[class_index] += np.bincount(
            score_bins[is_positive, class_index], minlength=bins
        )
        negative[class_index] += np.bincount(
            score_bins[~is_positive, class_index], minlength=bins
        )


def _auc_from_counts(positive: np.ndarray, negative: np.ndarray) -> float:
    positives = int(positive.sum())
    negatives = int(negative.sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    true_positive_rate = np.r_[0.0, np.cumsum(positive[::-1]) / positives]
    false_positive_rate = np.r_[0.0, np.cumsum(negative[::-1]) / negatives]
    # Spell out the trapezoid rule: np.trapz was removed in NumPy 2.x while
    # np.trapezoid is unavailable in older local environments.
    widths = np.diff(false_positive_rate)
    heights = (true_positive_rate[:-1] + true_positive_rate[1:]) * 0.5
    return float(np.sum(widths * heights))


def _roc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> dict:
    per_class = np.asarray(
        [_auc_from_counts(positive[i], negative[i]) for i in range(len(positive))],
        dtype=np.float64,
    )
    support = positive.sum(axis=1)
    valid = np.isfinite(per_class)
    return {
        "per_class": per_class,
        "roc_auc_ovr_macro": float(per_class[valid].mean()) if valid.any() else float("nan"),
        "roc_auc_ovr_weighted": (
            float(np.average(per_class[valid], weights=support[valid]))
            if valid.any() and support[valid].sum() else float("nan")
        ),
        "roc_auc_ovr_micro": _auc_from_counts(positive.sum(axis=0), negative.sum(axis=0)),
    }


def _overall_from_confusion(
    confusion: np.ndarray,
    native_class_indices: set[int] | None = None,
    roc_metrics: dict | None = None,
) -> dict[str, float | int]:
    confusion = confusion.astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    present = support > 0
    total = confusion.sum()
    accuracy = tp.sum() / total if total else 0.0
    weighted_precision = float((precision * support).sum() / total) if total else 0.0
    weighted_recall = float((recall * support).sum() / total) if total else 0.0
    weighted_f1 = float((f1 * support).sum() / total) if total else 0.0
    numerator = total * tp.sum() - float(np.dot(support, predicted))
    denominator = np.sqrt((total**2 - float(np.dot(predicted, predicted))) * (total**2 - float(np.dot(support, support))))
    expected = float(np.dot(support, predicted)) / total**2 if total else 0.0
    foreign_prediction_rate = 0.0
    if native_class_indices is not None and total:
        foreign_columns = [index for index in range(len(support)) if index not in native_class_indices]
        foreign_prediction_rate = float(confusion[:, foreign_columns].sum() / total) if foreign_columns else 0.0
    return {
        "rows": int(total),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "micro_precision": float(accuracy),
        "micro_recall": float(accuracy),
        "micro_f1": float(accuracy),
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "matthews_corrcoef": float(numerator / denominator) if denominator else 0.0,
        "cohen_kappa": float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else 0.0,
        "foreign_prediction_rate": foreign_prediction_rate,
        "classes_with_test_support": int(present.sum()),
        "roc_auc_ovr_macro": roc_metrics["roc_auc_ovr_macro"] if roc_metrics else float("nan"),
        "roc_auc_ovr_weighted": roc_metrics["roc_auc_ovr_weighted"] if roc_metrics else float("nan"),
        "roc_auc_ovr_micro": roc_metrics["roc_auc_ovr_micro"] if roc_metrics else float("nan"),
    }


def _class_rows(
    confusion: np.ndarray,
    class_names: list[str],
    origin: str,
    per_class_roc_auc: np.ndarray | None = None,
) -> list[dict]:
    total = int(confusion.sum())
    rows = []
    for index, label in enumerate(class_names):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum() - tp)
        fp = int(confusion[:, index].sum() - tp)
        tn = total - tp - fn - fp
        support = tp + fn; predicted_support = tp + fp
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / support if support else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        npv = tn / (tn + fn) if tn + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "origin": origin, "class": label, "is_native_class": bool(support),
            "support": support, "predicted_support": predicted_support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prevalence": support / total if total else 0.0,
            "precision": precision, "recall": recall, "specificity": specificity,
            "npv": npv, "f1": f1,
            "roc_auc_ovr": (
                float(per_class_roc_auc[index])
                if per_class_roc_auc is not None else float("nan")
            ),
            "false_positive_rate": 1.0 - specificity,
            "false_negative_rate": 1.0 - recall,
            "class_balanced_accuracy": (recall + specificity) / 2.0,
        })
    return rows


def evaluate_and_report_ooc(
    model: torch.nn.Module,
    context: OutOfCoreContext,
    config: dict,
    contract: dict,
) -> dict[str, pd.DataFrame]:
    data = context.data
    device = next(model.parameters()).device
    result_dir = config["evaluation"]["output_dir"]
    prediction_dir = os.path.join(result_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    chunk_rows = int(config["evaluation"].get("prediction_chunk_rows", 262_144))
    num_classes = len(data.class_names)
    has_expert_bank = hasattr(model, "expert_bank")
    has_hard_router_model = hasattr(model, "stage_b") and hasattr(model, "id_head")
    num_experts = len(data.active_datasets) if has_expert_bank or has_hard_router_model else 0
    roc_bins = int(config["evaluation"].get("roc_auc_bins", 4096))
    if roc_bins < 2:
        raise ValueError("evaluation.roc_auc_bins must be at least 2")
    confusion_by_origin = {}
    expert_confusion_by_origin = {}
    roc_hist_by_origin = {}
    expert_roc_hist_by_origin = {}
    native_indices_by_origin = {
        origin: {data.class_names.index(label) for label in context.prepared_by_dataset[origin].class_names}
        for origin in data.active_datasets
    }
    gate_sum_rows = []
    utilization = np.zeros(num_experts, dtype=np.int64)
    model.eval()

    for origin in data.active_datasets:
        bounds = data.test.dataset_slices[origin]
        amount = bounds.stop - bounds.start
        prediction_path = os.path.join(prediction_dir, f"{origin}.npy")
        predictions = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.int16, shape=(amount,))
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        expert_confusion = np.zeros((num_experts, num_classes, num_classes), dtype=np.int64)
        roc_positive = np.zeros((num_classes, roc_bins), dtype=np.int64)
        roc_negative = np.zeros((num_classes, roc_bins), dtype=np.int64)
        expert_roc_positive = np.zeros((num_experts, num_classes, roc_bins), dtype=np.int64)
        expert_roc_negative = np.zeros((num_experts, num_classes, roc_bins), dtype=np.int64)
        gate_sum = np.zeros(num_experts, dtype=np.float64)
        routing_correct = 0
        has_hard_dataset_router = False
        cursor = 0
        with torch.no_grad():
            for start in range(bounds.start, bounds.stop, chunk_rows):
                stop = min(start + chunk_rows, bounds.stop)
                features = np.asarray(data.test.features[start:stop], dtype=np.float32)
                truth = np.asarray(data.test.class_idx[start:stop], dtype=np.int64)
                tensor = torch.from_numpy(features).to(device, non_blocking=True)
                output = model(tensor, origin) if isinstance(model, NoFusionModel) else model(tensor)
                if "combined_probs" in output:
                    class_probabilities_tensor = output["combined_probs"]
                elif "logits" in output:
                    class_probabilities_tensor = F.softmax(output["logits"], dim=1)
                else:
                    raise KeyError("model output must contain combined_probs or logits")
                class_probabilities = class_probabilities_tensor.cpu().numpy()
                prediction = class_probabilities.argmax(axis=1).astype(np.int16)
                _update_roc_histograms(
                    roc_positive, roc_negative, truth, class_probabilities
                )
                # Sparse inference executes only the selected expert. The
                # all-expert matrix is computed explicitly here for offline
                # diagnostics and never participates in routing or prediction.
                if "dataset_pred" in output:
                    has_hard_dataset_router = True
                    routes = output["dataset_pred"].cpu().numpy()
                    routing_correct += int((routes == data.active_datasets.index(origin)).sum())
                    gates = np.eye(num_experts, dtype=np.float32)[routes]
                    expert_predictions = None
                    expert_probabilities = None
                elif has_expert_bank:
                    expert_logits = output.get("expert_logits")
                    if expert_logits is None:
                        expert_logits = model.expert_bank(output["z"])
                    expert_probabilities = F.softmax(expert_logits, dim=2).cpu().numpy()
                    expert_predictions = expert_probabilities.argmax(axis=2)
                    gates = output["gate_weights"].cpu().numpy()
                else:
                    routes = np.zeros(len(prediction), dtype=np.int64)
                    gates = np.zeros((len(prediction), 0), dtype=np.float32)
                    expert_predictions = None
                    expert_probabilities = None
                predictions[cursor : cursor + len(prediction)] = prediction
                confusion += np.bincount(
                    truth * num_classes + prediction, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
                for expert_index in range(num_experts):
                    selected = routes == expert_index if expert_predictions is None else slice(None)
                    selected_prediction = (
                        prediction[selected]
                        if expert_predictions is None
                        else expert_predictions[:, expert_index]
                    )
                    expert_confusion[expert_index] += np.bincount(
                        truth[selected] * num_classes + selected_prediction,
                        minlength=num_classes * num_classes,
                    ).reshape(num_classes, num_classes)
                    selected_truth = truth[selected]
                    selected_probabilities = (
                        class_probabilities[selected]
                        if expert_probabilities is None
                        else expert_probabilities[:, expert_index]
                    )
                    _update_roc_histograms(
                        expert_roc_positive[expert_index],
                        expert_roc_negative[expert_index],
                        selected_truth,
                        selected_probabilities,
                    )
                if num_experts:
                    gate_sum += gates.sum(axis=0)
                    utilization += np.bincount(gates.argmax(axis=1), minlength=num_experts)
                cursor += len(prediction)
        predictions.flush()
        confusion_by_origin[origin] = confusion
        expert_confusion_by_origin[origin] = expert_confusion
        roc_hist_by_origin[origin] = (roc_positive, roc_negative)
        expert_roc_hist_by_origin[origin] = (expert_roc_positive, expert_roc_negative)
        gate_sum_rows.append({
            "origin": origin, "test_rows": amount,
            "stage_a_dataset_accuracy": (
                float(routing_correct / amount) if has_hard_dataset_router and amount else np.nan
            ),
            **{
                f"mean_gate__{expert}": float(gate_sum[i] / amount)
                for i, expert in enumerate(data.active_datasets[:num_experts])
            },
        })
        print(
            f"[evaluation:{origin}] "
            f"macro_f1={_overall_from_confusion(confusion, native_indices_by_origin[origin])['macro_f1']:.6f} "
            f"roc_auc_ovr_macro={_roc_from_histograms(roc_positive, roc_negative)['roc_auc_ovr_macro']:.6f} "
            f"rows={amount:,}"
        )

    combined = sum(confusion_by_origin.values(), np.zeros((num_classes, num_classes), dtype=np.int64))
    combined_roc_positive = sum(
        (hist[0] for hist in roc_hist_by_origin.values()),
        np.zeros((num_classes, roc_bins), dtype=np.int64),
    )
    combined_roc_negative = sum(
        (hist[1] for hist in roc_hist_by_origin.values()),
        np.zeros((num_classes, roc_bins), dtype=np.int64),
    )
    overall_rows = []
    per_class_rows = []
    for origin, confusion in [*(confusion_by_origin.items()), ("ALL", combined)]:
        positive, negative = (
            roc_hist_by_origin[origin]
            if origin != "ALL" else (combined_roc_positive, combined_roc_negative)
        )
        roc_metrics = _roc_from_histograms(positive, negative)
        overall_rows.append({
            "run_name": config["run_name"], "architecture": config["architecture"],
            "combination_size": len(data.active_datasets), "origin": origin,
            **_overall_from_confusion(
                confusion,
                native_indices_by_origin[origin] if origin != "ALL" else set().union(*native_indices_by_origin.values()),
                roc_metrics,
            ),
            "roc_auc_method": f"histogram_approximation_{roc_bins}_bins",
        })
        per_class_rows.extend(
            _class_rows(confusion, data.class_names, origin, roc_metrics["per_class"])
        )

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(per_class_rows)
    per_class["roc_auc_method"] = f"histogram_approximation_{roc_bins}_bins"
    gate = pd.DataFrame(gate_sum_rows)
    expert_performance_rows = []
    for origin, expert_confusions in expert_confusion_by_origin.items():
        expert_positive, expert_negative = expert_roc_hist_by_origin[origin]
        for expert_index, expert_name in enumerate(data.active_datasets[:num_experts]):
            expert_roc = _roc_from_histograms(
                expert_positive[expert_index], expert_negative[expert_index]
            )
            expert_performance_rows.append({
                "origin": origin,
                "expert": expert_name,
                "is_assigned_expert": expert_name == origin,
                **_overall_from_confusion(
                    expert_confusions[expert_index], native_indices_by_origin[origin], expert_roc
                ),
                "roc_auc_method": f"histogram_approximation_{roc_bins}_bins",
            })
    expert_performance = pd.DataFrame(expert_performance_rows)
    utilization_frame = pd.DataFrame([
        {
            "expert": name, "hard_argmax_rows": int(utilization[i]),
            "hard_argmax_fraction": float(utilization[i] / utilization.sum()) if utilization.sum() else 0.0,
        }
        for i, name in enumerate(data.active_datasets[:num_experts])
    ])
    confusion_frame = pd.DataFrame(combined, index=data.class_names, columns=data.class_names)
    confusion_frame.index.name = "true_class"
    per_dataset = overall[overall["origin"] != "ALL"].copy()
    per_dataset["dataset"] = per_dataset["origin"]
    trial_id = f"{config['run_name']}-seed{config.get('seed', 0)}"
    split_signature = contract.get("preprocessing_signature") or contract.get("signature")
    stage_path = resolve_stage_a_path(config)
    baseline_cfg = (
        config.get("training", {}).get("baseline", {})
        if config["architecture"] in {"plain_pooled", "matched_dense", "no_fusion", "hard_two_stage"}
        else {}
    )
    trials = pd.DataFrame([{
        "Trial_ID": trial_id,
        "run_name": config["run_name"], "architecture": config["architecture"],
        "routing_mode": config["model"].get("gate", {}).get("routing", "dense"),
        "gate_supervision": config["training"]["stage_c"].get("gate_supervision", "light_aux"),
        "expert_update_policy": config["training"]["stage_c"].get("expert_update_policy", "all"),
        "seed": config.get("seed", 0), "combination_size": len(data.active_datasets),
        "active_datasets": "|".join(data.active_datasets),
        "contract_signature": contract["signature"],
        "split_signature": split_signature,
        "matching_axis": config.get("model", {}).get("dense_match", {}).get("axis") if config["architecture"] == "matched_dense" else None,
        "encoder_init": baseline_cfg.get("encoder_init"),
        "stage_b_warmstart": baseline_cfg.get("stage_b_warmstart"),
        "selection_mode": config.get("training", {}).get("selection_mode", "fixed_epochs"),
        "stage_a_checkpoint_sha256": stage_a_checkpoint_hash(stage_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }])
    overall.insert(0, "Trial_ID", trial_id)
    per_class.insert(0, "Trial_ID", trial_id)
    per_dataset = overall[overall["origin"] != "ALL"].copy()
    per_dataset["dataset"] = per_dataset["origin"]

    training_summary = dict(getattr(model, "training_summary", {}))
    if os.path.isfile(stage_path):
        stage_metadata = torch.load(stage_path, map_location="cpu").get("metadata") or {}
        if stage_metadata.get("training_summary"):
            training_summary["A"] = stage_metadata["training_summary"]
    resource_row = resource_profile(
        model,
        config,
        trial_id=trial_id,
        split_signature=split_signature,
        checkpoint_hash=stage_a_checkpoint_hash(stage_path),
        training_summary=training_summary,
    )

    for filename, frame, index in [
        ("Trials.csv", trials, False),
        ("Overall_Metrics.csv", overall, False),
        ("Per_Dataset_Metrics.csv", per_dataset, False),
        ("Per_Class_Metrics.csv", per_class, False),
        ("Gate_By_Dataset.csv", gate, False),
        ("Expert_Performance_By_Dataset.csv", expert_performance, False),
        ("Expert_Utilization.csv", utilization_frame, False),
        ("Confusion_Matrix.csv", confusion_frame, True),
    ]:
        temporary = os.path.join(result_dir, filename + ".tmp")
        frame.to_csv(temporary, index=index)
        os.replace(temporary, os.path.join(result_dir, filename))
    write_resource_accounting(result_dir, [resource_row])

    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "architecture": config["architecture"],
        "active_datasets": data.active_datasets,
        "class_names": data.class_names,
        "run_contract": contract,
        "prediction_files": {name: os.path.join(prediction_dir, f"{name}.npy") for name in data.active_datasets},
        "report_files": [
            "Trials.csv", "Overall_Metrics.csv", "Per_Dataset_Metrics.csv",
            "Per_Class_Metrics.csv", "Gate_By_Dataset.csv",
            "Expert_Performance_By_Dataset.csv", "Expert_Utilization.csv",
            "Confusion_Matrix.csv",
            "Resource_Accounting.csv",
        ],
    }
    manifest_path = os.path.join(result_dir, "manifest.json")
    temporary = manifest_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return {
        "overall": overall, "per_class": per_class, "gate": gate,
        "expert_performance": expert_performance,
        "utilization": utilization_frame, "confusion": confusion_frame,
    }
