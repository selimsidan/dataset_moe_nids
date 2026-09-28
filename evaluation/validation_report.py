"""Validation-only selection artifacts for adaptive architecture studies."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from .resource_accounting import resource_profile


def _rows_from_confusion(confusion: np.ndarray, class_names: list[str]) -> tuple[dict, list[dict]]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=float),
        where=(precision + recall) != 0,
    )
    present = support > 0
    overall = {
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "accuracy": float(true_positive.sum() / support.sum()) if support.sum() else 0.0,
        "support": int(support.sum()),
    }
    per_class = [
        {
            "class": name,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(class_names)
    ]
    return overall, per_class


def _atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_validation_report_ooc(model, context, config: dict) -> dict[str, pd.DataFrame]:
    """Evaluate the selected Stage-C checkpoint on validation, never test."""
    split = context.data.val
    class_names = context.data.class_names
    dataset_names = context.data.active_datasets
    classes = len(class_names)
    device = torch.device(config["training"]["device"])
    chunk_rows = int(config["training"].get("validation_chunk_rows", 262_144))
    overall_confusion = np.zeros((classes, classes), dtype=np.int64)
    dataset_confusions = {
        index: np.zeros((classes, classes), dtype=np.int64)
        for index in range(len(dataset_names))
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(split.class_idx), chunk_rows):
            stop = min(start + chunk_rows, len(split.class_idx))
            features = torch.from_numpy(
                np.asarray(split.features[start:stop], dtype=np.float32)
            ).to(device, non_blocking=True)
            truth = np.asarray(split.class_idx[start:stop], dtype=np.int64)
            dataset_ids = np.asarray(split.dataset_idx[start:stop], dtype=np.int64)
            if hasattr(model, "predict"):
                prediction = model.predict(features).cpu().numpy()
            else:
                prediction = model(features)["logits"].argmax(dim=1).cpu().numpy()
            flat = truth * classes + prediction
            overall_confusion += np.bincount(
                flat, minlength=classes * classes
            ).reshape(classes, classes)
            for dataset_index in dataset_confusions:
                mask = dataset_ids == dataset_index
                if mask.any():
                    dataset_confusions[dataset_index] += np.bincount(
                        flat[mask], minlength=classes * classes
                    ).reshape(classes, classes)

    overall, per_class = _rows_from_confusion(overall_confusion, class_names)
    resources = resource_profile(model, config)
    overall["total_parameters"] = int(resources["total_parameters"])
    overall["forward_macs_per_sample_mean"] = int(
        resources["forward_macs_per_sample_mean"]
    )
    per_dataset = []
    for index, name in enumerate(dataset_names):
        row, _ = _rows_from_confusion(dataset_confusions[index], class_names)
        per_dataset.append({"dataset": name, **row})
    frames = {
        "overall": pd.DataFrame([overall]),
        "per_dataset": pd.DataFrame(per_dataset),
        "per_class": pd.DataFrame(per_class),
    }
    output_dir = config["evaluation"]["output_dir"]
    _atomic_csv(frames["overall"], os.path.join(output_dir, "Validation_Overall_Metrics.csv"))
    _atomic_csv(frames["per_dataset"], os.path.join(output_dir, "Validation_Per_Dataset_Metrics.csv"))
    _atomic_csv(frames["per_class"], os.path.join(output_dir, "Validation_Per_Class_Metrics.csv"))
    return frames


def write_owned_expert_validation_ooc(
    encoder,
    bank,
    context,
    config: dict,
    path: str,
) -> pd.DataFrame:
    """Measure each dataset-owned expert on its validation rows."""
    from training.model_utils import expert_forward_one

    split = context.data.val
    device = next(bank.parameters()).device
    chunk_rows = int(config["training"].get("validation_chunk_rows", 262_144))
    classes = len(context.data.class_names)
    rows = []
    encoder.eval(); bank.eval()
    with torch.no_grad():
        for dataset_index, name in enumerate(context.data.active_datasets):
            confusion = np.zeros((classes, classes), dtype=np.int64)
            bounds = split.dataset_slices[name]
            for start in range(bounds.start, bounds.stop, chunk_rows):
                stop = min(start + chunk_rows, bounds.stop)
                features = torch.from_numpy(
                    np.asarray(split.features[start:stop], dtype=np.float32)
                ).to(device, non_blocking=True)
                truth = np.asarray(split.class_idx[start:stop], dtype=np.int64)
                representation = features if getattr(bank, "expects_raw_input", False) else encoder(features)
                prediction = expert_forward_one(bank, dataset_index, representation).argmax(dim=1).cpu().numpy()
                confusion += np.bincount(
                    truth * classes + prediction, minlength=classes * classes
                ).reshape(classes, classes)
            overall, _ = _rows_from_confusion(confusion, context.data.class_names)
            rows.append({"dataset": name, **overall})
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, path)
    return frame
