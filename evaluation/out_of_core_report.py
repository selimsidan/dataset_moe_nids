"""Chunked predictions and comprehensive reports for full-data MoE runs."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from training.out_of_core_data import OutOfCoreContext


def _overall_from_confusion(
    confusion: np.ndarray, native_class_indices: set[int] | None = None
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
        "weighted_f1": weighted_f1,
        "micro_f1": float(accuracy),
        "matthews_corrcoef": float(numerator / denominator) if denominator else 0.0,
        "cohen_kappa": float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else 0.0,
        "foreign_prediction_rate": foreign_prediction_rate,
        "classes_with_test_support": int(present.sum()),
    }


def _class_rows(confusion: np.ndarray, class_names: list[str], origin: str) -> list[dict]:
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
    num_experts = len(data.active_datasets)
    confusion_by_origin = {}
    expert_confusion_by_origin = {}
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
        gate_sum = np.zeros(num_experts, dtype=np.float64)
        cursor = 0
        with torch.no_grad():
            for start in range(bounds.start, bounds.stop, chunk_rows):
                stop = min(start + chunk_rows, bounds.stop)
                features = np.asarray(data.test.features[start:stop], dtype=np.float32)
                truth = np.asarray(data.test.class_idx[start:stop], dtype=np.int64)
                output = model(torch.from_numpy(features).to(device, non_blocking=True))
                prediction = output["combined_probs"].argmax(dim=1).cpu().numpy().astype(np.int16)
                expert_predictions = output["expert_logits"].argmax(dim=2).cpu().numpy()
                gates = output["gate_weights"].cpu().numpy()
                predictions[cursor : cursor + len(prediction)] = prediction
                confusion += np.bincount(
                    truth * num_classes + prediction, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
                for expert_index in range(num_experts):
                    expert_confusion[expert_index] += np.bincount(
                        truth * num_classes + expert_predictions[:, expert_index],
                        minlength=num_classes * num_classes,
                    ).reshape(num_classes, num_classes)
                gate_sum += gates.sum(axis=0)
                utilization += np.bincount(gates.argmax(axis=1), minlength=num_experts)
                cursor += len(prediction)
        predictions.flush()
        confusion_by_origin[origin] = confusion
        expert_confusion_by_origin[origin] = expert_confusion
        gate_sum_rows.append({
            "origin": origin, "test_rows": amount,
            **{f"mean_gate__{expert}": float(gate_sum[i] / amount) for i, expert in enumerate(data.active_datasets)},
        })
        print(
            f"[evaluation:{origin}] "
            f"macro_f1={_overall_from_confusion(confusion, native_indices_by_origin[origin])['macro_f1']:.6f} "
            f"rows={amount:,}"
        )

    combined = sum(confusion_by_origin.values(), np.zeros((num_classes, num_classes), dtype=np.int64))
    overall_rows = []
    per_class_rows = []
    for origin, confusion in [*(confusion_by_origin.items()), ("ALL", combined)]:
        overall_rows.append({
            "run_name": config["run_name"], "architecture": config["architecture"],
            "combination_size": len(data.active_datasets), "origin": origin,
            **_overall_from_confusion(
                confusion,
                native_indices_by_origin[origin] if origin != "ALL" else set().union(*native_indices_by_origin.values()),
            ),
        })
        per_class_rows.extend(_class_rows(confusion, data.class_names, origin))

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(per_class_rows)
    gate = pd.DataFrame(gate_sum_rows)
    expert_performance_rows = []
    for origin, expert_confusions in expert_confusion_by_origin.items():
        for expert_index, expert_name in enumerate(data.active_datasets):
            expert_performance_rows.append({
                "origin": origin,
                "expert": expert_name,
                "is_assigned_expert": expert_name == origin,
                **_overall_from_confusion(
                    expert_confusions[expert_index], native_indices_by_origin[origin]
                ),
            })
    expert_performance = pd.DataFrame(expert_performance_rows)
    utilization_frame = pd.DataFrame([
        {
            "expert": name, "hard_argmax_rows": int(utilization[i]),
            "hard_argmax_fraction": float(utilization[i] / utilization.sum()) if utilization.sum() else 0.0,
        }
        for i, name in enumerate(data.active_datasets)
    ])
    confusion_frame = pd.DataFrame(combined, index=data.class_names, columns=data.class_names)
    confusion_frame.index.name = "true_class"
    per_dataset = overall[overall["origin"] != "ALL"].copy()
    trials = pd.DataFrame([{
        "run_name": config["run_name"], "architecture": config["architecture"],
        "seed": config.get("seed", 0), "combination_size": len(data.active_datasets),
        "active_datasets": "|".join(data.active_datasets),
        "contract_signature": contract["signature"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }])

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
