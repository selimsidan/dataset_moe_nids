"""Chunked routing/blending/representation diagnostics for dataset-MoE.

The diagnostic evaluates two already-trained models on exactly the same test
rows under five counterfactual inference policies:

* ``moe_dense``: the ordinary learned soft mixture.
* ``moe_learned_top1``: the same MoE and checkpoint, selecting gate argmax.
* ``moe_oracle_dataset``: the same MoE, selecting the true dataset owner.
* ``hard_predicted_route``: the ordinary hard two-stage model.
* ``hard_oracle_dataset``: the hard model's independent classifier selected
  with the true dataset owner.

No policy is retrained here.  Oracle policies are diagnostics only and must
never be reported as deployable dataset-blind results.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


VARIANTS = (
    "moe_dense",
    "moe_learned_top1",
    "moe_oracle_dataset",
    "hard_predicted_route",
    "hard_oracle_dataset",
)


@dataclass
class RouterDiagnosticResult:
    overall: pd.DataFrame
    per_dataset: pd.DataFrame
    per_class: pd.DataFrame
    per_dataset_per_class: pd.DataFrame
    routing: pd.DataFrame
    route_conditioned: pd.DataFrame
    confidence_bins: pd.DataFrame
    expert_cross_dataset: pd.DataFrame
    pairwise_correctness: pd.DataFrame
    moe_route_confusion: pd.DataFrame
    hard_route_confusion: pd.DataFrame
    class_confusions: dict[str, pd.DataFrame]


def _add_confusion(confusion: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> None:
    classes = confusion.shape[0]
    confusion += np.bincount(
        truth * classes + prediction,
        minlength=classes * classes,
    ).reshape(classes, classes)


def _metrics(confusion: np.ndarray) -> dict[str, float | int]:
    matrix = confusion.astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    present = support > 0
    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    return {
        "rows": int(total),
        "accuracy": accuracy,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "micro_f1": accuracy,
        "weighted_f1": float(np.dot(f1, support) / total) if total else 0.0,
    }


def _per_class_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict]:
    matrix = confusion.astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    return [
        {
            "class": name,
            "support": int(support[index]),
            "predicted_support": int(predicted[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(class_names)
    ]


def _origin_ranges(test_split, dataset_names: list[str]):
    slices = getattr(test_split, "dataset_slices", None)
    if slices:
        for dataset_index, name in enumerate(dataset_names):
            bounds = slices[name]
            yield dataset_index, name, bounds
        return
    names = np.asarray(test_split.dataset_name)
    for dataset_index, name in enumerate(dataset_names):
        yield dataset_index, name, np.flatnonzero(names == name)


def _row_chunks(rows, chunk_rows: int):
    if isinstance(rows, slice):
        if rows.step not in (None, 1):
            raise ValueError("out-of-core dataset slices must be contiguous")
        start = 0 if rows.start is None else int(rows.start)
        stop = int(rows.stop)
        for chunk_start in range(start, stop, chunk_rows):
            yield slice(chunk_start, min(chunk_start + chunk_rows, stop))
        return
    for offset in range(0, len(rows), chunk_rows):
        yield rows[offset : offset + chunk_rows]


def _take(array, row_ids):
    if isinstance(row_ids, slice):
        return np.asarray(array[row_ids])
    if not len(row_ids):
        return np.asarray(array[0:0])
    # Contiguous slices avoid materializing large index arrays in out-of-core mode.
    if len(row_ids) == 1 or np.all(np.diff(row_ids) == 1):
        return np.asarray(array[int(row_ids[0]) : int(row_ids[-1]) + 1])
    return np.asarray(array[row_ids])


def _selected_probabilities(expert_probabilities: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    row_ids = torch.arange(expert_probabilities.shape[0], device=expert_probabilities.device)
    return expert_probabilities[row_ids, routes]


def run_router_diagnostics(
    moe: torch.nn.Module,
    hard: torch.nn.Module,
    data,
    *,
    output_dir: str | None = None,
    chunk_rows: int = 262_144,
) -> RouterDiagnosticResult:
    """Evaluate routing counterfactuals without retraining either model.

    ``data`` is a ``PreparedData`` instance from either the in-memory or
    out-of-core pipeline.  Both models must use the same ordered dataset and
    class vocabularies as ``data``.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    dataset_names = list(data.active_datasets)
    class_names = list(data.class_names)
    if list(moe.dataset_names) != dataset_names:
        raise ValueError("MoE dataset order does not match the prepared data")
    if list(hard.dataset_names) != dataset_names:
        raise ValueError("Hard-model dataset order does not match the prepared data")
    if int(moe.expert_bank.num_classes) != len(class_names) or int(hard.num_classes) != len(class_names):
        raise ValueError("Model class vocabulary size does not match the prepared data")

    moe_device = next(moe.parameters()).device
    hard_device = next(hard.parameters()).device
    moe.eval()
    hard.eval()

    overall_confusions = {
        variant: np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for variant in VARIANTS
    }
    per_dataset_confusions = {
        (variant, name): np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for variant in VARIANTS for name in dataset_names
    }
    expert_cross_confusions = {
        (origin, expert): np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for origin in dataset_names for expert in dataset_names
    }
    moe_route_confusion = np.zeros((len(dataset_names), len(dataset_names)), dtype=np.int64)
    hard_route_confusion = np.zeros_like(moe_route_confusion)
    route_stats = {
        name: {
            "rows": 0,
            "moe_correct": 0,
            "hard_correct": 0,
            "moe_confidence_sum": 0.0,
            "moe_entropy_sum": 0.0,
        }
        for name in dataset_names
    }
    pair_counts = {
        (left, right): {"both_correct": 0, "left_only": 0, "right_only": 0, "both_wrong": 0}
        for left in VARIANTS for right in VARIANTS if VARIANTS.index(left) < VARIANTS.index(right)
    }
    route_conditioned_confusions = {
        (variant, condition): np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for variant in ("moe_dense", "moe_learned_top1", "hard_predicted_route")
        for condition in ("route_correct", "route_wrong")
    }
    confidence_edges = np.asarray([0.0, 0.5, 0.7, 0.9, 1.0000001], dtype=np.float64)
    confidence_stats = [
        {"rows": 0, "route_correct": 0, "dense_correct": 0, "top1_correct": 0, "oracle_correct": 0}
        for _ in range(len(confidence_edges) - 1)
    ]

    with torch.no_grad():
        for dataset_index, origin, origin_rows in _origin_ranges(data.test, dataset_names):
            for row_ids in _row_chunks(origin_rows, chunk_rows):
                features_np = _take(data.test.features, row_ids).astype(np.float32, copy=False)
                truth = _take(data.test.class_idx, row_ids).astype(np.int64, copy=False)
                features_moe = torch.from_numpy(features_np).to(moe_device, non_blocking=True)
                features_hard = (
                    features_moe.to(hard_device)
                    if hard_device != moe_device
                    else features_moe
                )

                gate_weights = moe.gate_weights_for(features_moe)
                expert_logits = moe.all_expert_logits(features_moe)
                expert_probabilities = F.softmax(expert_logits, dim=-1)
                moe_routes = gate_weights.argmax(dim=1)
                oracle_routes = torch.full_like(moe_routes, dataset_index)

                dense_probabilities = torch.einsum(
                    "bd,bdc->bc", gate_weights, expert_probabilities
                )
                top1_probabilities = _selected_probabilities(expert_probabilities, moe_routes)
                oracle_moe_probabilities = _selected_probabilities(
                    expert_probabilities, oracle_routes
                )

                hard_output = hard(features_hard)
                hard_probabilities = F.softmax(hard_output["logits"], dim=1)
                hard_oracle_probabilities = F.softmax(
                    hard.stage_b(features_hard, origin)["logits"], dim=1
                )

                prediction = {
                    "moe_dense": dense_probabilities.argmax(dim=1).cpu().numpy(),
                    "moe_learned_top1": top1_probabilities.argmax(dim=1).cpu().numpy(),
                    "moe_oracle_dataset": oracle_moe_probabilities.argmax(dim=1).cpu().numpy(),
                    "hard_predicted_route": hard_probabilities.argmax(dim=1).cpu().numpy(),
                    "hard_oracle_dataset": hard_oracle_probabilities.argmax(dim=1).cpu().numpy(),
                }
                moe_routes_np = moe_routes.cpu().numpy()
                hard_routes_np = hard_output["dataset_pred"].cpu().numpy()
                true_routes = np.full(len(truth), dataset_index, dtype=np.int64)
                _add_confusion(moe_route_confusion, true_routes, moe_routes_np)
                _add_confusion(hard_route_confusion, true_routes, hard_routes_np)

                confidence = gate_weights.max(dim=1).values
                entropy = -(gate_weights * gate_weights.clamp_min(1e-12).log()).sum(dim=1)
                stats = route_stats[origin]
                stats["rows"] += len(truth)
                stats["moe_correct"] += int((moe_routes_np == dataset_index).sum())
                stats["hard_correct"] += int((hard_routes_np == dataset_index).sum())
                stats["moe_confidence_sum"] += float(confidence.sum().item())
                stats["moe_entropy_sum"] += float(entropy.sum().item())

                expert_predictions = expert_probabilities.argmax(dim=2).cpu().numpy()
                for expert_index, expert_name in enumerate(dataset_names):
                    _add_confusion(
                        expert_cross_confusions[(origin, expert_name)],
                        truth,
                        expert_predictions[:, expert_index],
                    )

                correct = {}
                for variant in VARIANTS:
                    _add_confusion(overall_confusions[variant], truth, prediction[variant])
                    _add_confusion(
                        per_dataset_confusions[(variant, origin)], truth, prediction[variant]
                    )
                    correct[variant] = prediction[variant] == truth
                moe_route_is_correct = moe_routes_np == dataset_index
                hard_route_is_correct = hard_routes_np == dataset_index
                for variant in ("moe_dense", "moe_learned_top1"):
                    _add_confusion(
                        route_conditioned_confusions[(variant, "route_correct")],
                        truth[moe_route_is_correct], prediction[variant][moe_route_is_correct],
                    )
                    _add_confusion(
                        route_conditioned_confusions[(variant, "route_wrong")],
                        truth[~moe_route_is_correct], prediction[variant][~moe_route_is_correct],
                    )
                _add_confusion(
                    route_conditioned_confusions[("hard_predicted_route", "route_correct")],
                    truth[hard_route_is_correct], prediction["hard_predicted_route"][hard_route_is_correct],
                )
                _add_confusion(
                    route_conditioned_confusions[("hard_predicted_route", "route_wrong")],
                    truth[~hard_route_is_correct], prediction["hard_predicted_route"][~hard_route_is_correct],
                )
                confidence_np = confidence.cpu().numpy()
                confidence_bin = np.clip(
                    np.digitize(confidence_np, confidence_edges[1:-1], right=False),
                    0, len(confidence_stats) - 1,
                )
                for bin_index, bin_stats in enumerate(confidence_stats):
                    selected = confidence_bin == bin_index
                    if not selected.any():
                        continue
                    bin_stats["rows"] += int(selected.sum())
                    bin_stats["route_correct"] += int(moe_route_is_correct[selected].sum())
                    bin_stats["dense_correct"] += int(correct["moe_dense"][selected].sum())
                    bin_stats["top1_correct"] += int(correct["moe_learned_top1"][selected].sum())
                    bin_stats["oracle_correct"] += int(correct["moe_oracle_dataset"][selected].sum())
                for (left, right), counts in pair_counts.items():
                    left_correct = correct[left]
                    right_correct = correct[right]
                    counts["both_correct"] += int((left_correct & right_correct).sum())
                    counts["left_only"] += int((left_correct & ~right_correct).sum())
                    counts["right_only"] += int((~left_correct & right_correct).sum())
                    counts["both_wrong"] += int((~left_correct & ~right_correct).sum())

    overall_rows = [
        {"variant": variant, **_metrics(overall_confusions[variant])}
        for variant in VARIANTS
    ]
    per_dataset_rows = [
        {
            "variant": variant,
            "dataset": origin,
            **_metrics(per_dataset_confusions[(variant, origin)]),
        }
        for variant in VARIANTS for origin in dataset_names
    ]
    per_class_rows = [
        {"variant": variant, **row}
        for variant in VARIANTS
        for row in _per_class_rows(overall_confusions[variant], class_names)
    ]
    per_dataset_per_class_rows = [
        {"variant": variant, "dataset": origin, **row}
        for variant in VARIANTS
        for origin in dataset_names
        for row in _per_class_rows(per_dataset_confusions[(variant, origin)], class_names)
    ]
    routing_rows = []
    for origin in dataset_names:
        stats = route_stats[origin]
        rows = stats["rows"]
        routing_rows.append({
            "dataset": origin,
            "rows": rows,
            "moe_route_accuracy": stats["moe_correct"] / rows if rows else math.nan,
            "hard_route_accuracy": stats["hard_correct"] / rows if rows else math.nan,
            "moe_mean_max_gate": stats["moe_confidence_sum"] / rows if rows else math.nan,
            "moe_mean_gate_entropy": stats["moe_entropy_sum"] / rows if rows else math.nan,
        })
    total_rows = sum(stats["rows"] for stats in route_stats.values())
    routing_rows.append({
        "dataset": "ALL",
        "rows": total_rows,
        "moe_route_accuracy": sum(s["moe_correct"] for s in route_stats.values()) / total_rows,
        "hard_route_accuracy": sum(s["hard_correct"] for s in route_stats.values()) / total_rows,
        "moe_mean_max_gate": sum(s["moe_confidence_sum"] for s in route_stats.values()) / total_rows,
        "moe_mean_gate_entropy": sum(s["moe_entropy_sum"] for s in route_stats.values()) / total_rows,
    })
    expert_rows = [
        {
            "origin_dataset": origin,
            "expert": expert,
            "is_owner": origin == expert,
            **_metrics(expert_cross_confusions[(origin, expert)]),
        }
        for origin in dataset_names for expert in dataset_names
    ]
    pair_rows = [
        {"left": left, "right": right, **counts}
        for (left, right), counts in pair_counts.items()
    ]
    route_conditioned_rows = [
        {
            "variant": variant,
            "routing_condition": condition,
            **_metrics(route_conditioned_confusions[(variant, condition)]),
        }
        for variant in ("moe_dense", "moe_learned_top1", "hard_predicted_route")
        for condition in ("route_correct", "route_wrong")
    ]
    confidence_rows = []
    for index, stats in enumerate(confidence_stats):
        rows = stats["rows"]
        confidence_rows.append({
            "confidence_lower_inclusive": float(confidence_edges[index]),
            "confidence_upper_exclusive": float(confidence_edges[index + 1]),
            "rows": rows,
            "row_fraction": rows / total_rows if total_rows else math.nan,
            "moe_route_accuracy": stats["route_correct"] / rows if rows else math.nan,
            "moe_dense_accuracy": stats["dense_correct"] / rows if rows else math.nan,
            "moe_top1_accuracy": stats["top1_correct"] / rows if rows else math.nan,
            "moe_oracle_accuracy": stats["oracle_correct"] / rows if rows else math.nan,
            "top1_minus_dense_accuracy": (
                (stats["top1_correct"] - stats["dense_correct"]) / rows if rows else math.nan
            ),
        })
    moe_route_frame = pd.DataFrame(
        moe_route_confusion, index=dataset_names, columns=dataset_names
    )
    hard_route_frame = pd.DataFrame(
        hard_route_confusion, index=dataset_names, columns=dataset_names
    )
    moe_route_frame.index.name = "true_dataset"
    hard_route_frame.index.name = "true_dataset"
    class_confusions = {}
    for variant in VARIANTS:
        frame = pd.DataFrame(
            overall_confusions[variant], index=class_names, columns=class_names
        )
        frame.index.name = "true_class"
        class_confusions[variant] = frame

    result = RouterDiagnosticResult(
        overall=pd.DataFrame(overall_rows),
        per_dataset=pd.DataFrame(per_dataset_rows),
        per_class=pd.DataFrame(per_class_rows),
        per_dataset_per_class=pd.DataFrame(per_dataset_per_class_rows),
        routing=pd.DataFrame(routing_rows),
        route_conditioned=pd.DataFrame(route_conditioned_rows),
        confidence_bins=pd.DataFrame(confidence_rows),
        expert_cross_dataset=pd.DataFrame(expert_rows),
        pairwise_correctness=pd.DataFrame(pair_rows),
        moe_route_confusion=moe_route_frame,
        hard_route_confusion=hard_route_frame,
        class_confusions=class_confusions,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outputs = {
            "Diagnostic_Overall.csv": (result.overall, False),
            "Diagnostic_Per_Dataset.csv": (result.per_dataset, False),
            "Diagnostic_Per_Class.csv": (result.per_class, False),
            "Diagnostic_Per_Dataset_Per_Class.csv": (result.per_dataset_per_class, False),
            "Routing_Summary.csv": (result.routing, False),
            "Route_Conditioned_Performance.csv": (result.route_conditioned, False),
            "MoE_Confidence_Bins.csv": (result.confidence_bins, False),
            "MoE_Expert_Cross_Dataset.csv": (result.expert_cross_dataset, False),
            "Pairwise_Correctness.csv": (result.pairwise_correctness, False),
            "MoE_Route_Confusion.csv": (result.moe_route_confusion, True),
            "Hard_Route_Confusion.csv": (result.hard_route_confusion, True),
        }
        for filename, (frame, include_index) in outputs.items():
            path = os.path.join(output_dir, filename)
            temporary = path + ".tmp"
            frame.to_csv(temporary, index=include_index)
            os.replace(temporary, path)
        for variant, frame in result.class_confusions.items():
            path = os.path.join(output_dir, f"Class_Confusion__{variant}.csv")
            temporary = path + ".tmp"
            frame.to_csv(temporary, index=True)
            os.replace(temporary, path)
    return result


def diagnostic_gaps(overall: pd.DataFrame) -> pd.DataFrame:
    """Return signed accuracy/F1 gaps with their diagnostic interpretation."""
    indexed = overall.set_index("variant")
    comparisons = (
        (
            "moe_oracle_dataset", "moe_learned_top1", "MoE routing loss",
            "Positive means the learned MoE router leaves performance on the table.",
        ),
        (
            "moe_learned_top1", "moe_dense", "Dense blending penalty",
            "Positive means dense blending hurts; negative means blending helps.",
        ),
        (
            "hard_oracle_dataset", "hard_predicted_route", "Hard routing loss",
            "Positive means the hard router leaves performance on the table.",
        ),
        (
            "hard_oracle_dataset", "moe_oracle_dataset", "Independent-path advantage",
            "Positive means hard per-dataset encoders/classifiers beat the oracle-routed shared-encoder MoE.",
        ),
    )
    rows = []
    for left, right, name, interpretation in comparisons:
        rows.append({
            "diagnostic": name,
            "left": left,
            "right": right,
            "accuracy_gap_left_minus_right": indexed.loc[left, "accuracy"] - indexed.loc[right, "accuracy"],
            "macro_f1_gap_left_minus_right": indexed.loc[left, "macro_f1"] - indexed.loc[right, "macro_f1"],
            "weighted_f1_gap_left_minus_right": indexed.loc[left, "weighted_f1"] - indexed.loc[right, "weighted_f1"],
            "interpretation_when_positive": interpretation,
        })
    return pd.DataFrame(rows)


def diagnostic_readout(
    overall: pd.DataFrame,
    routing: pd.DataFrame,
    confidence_bins: pd.DataFrame,
    *,
    material_gap: float = 0.005,
) -> str:
    """Create a deterministic, human-readable single-seed diagnostic summary."""
    if material_gap < 0:
        raise ValueError("material_gap must be non-negative")
    indexed = overall.set_index("variant")
    route_all = routing.set_index("dataset").loc["ALL"]

    def value(left: str, right: str, metric: str = "accuracy") -> float:
        return float(indexed.loc[left, metric] - indexed.loc[right, metric])

    deployable_gap = value("hard_predicted_route", "moe_dense")
    representation_gap = value("hard_oracle_dataset", "moe_oracle_dataset")
    moe_routing_gap = value("moe_oracle_dataset", "moe_learned_top1")
    blending_penalty = value("moe_learned_top1", "moe_dense")
    hard_routing_gap = value("hard_oracle_dataset", "hard_predicted_route")

    def pp(number: float) -> str:
        return f"{number * 100:+.3f} percentage points"

    conclusions = []
    if representation_gap > material_gap:
        conclusions.append(
            "The hard model retains a material classification-path advantage even when both routers are bypassed. "
            "This points to independent per-dataset representation/classifier capacity or training, not a router-only problem."
        )
    elif representation_gap < -material_gap:
        conclusions.append(
            "The oracle-routed MoE classification path is materially stronger than the hard model's oracle path. "
            "Any deployable deficit is therefore caused downstream by learned routing or blending."
        )
    else:
        conclusions.append(
            "The two oracle-routed classification paths are practically close at the configured threshold; "
            "raw classification-path capacity is not the dominant observed gap."
        )
    if moe_routing_gap > material_gap:
        conclusions.append("The learned MoE gate incurs a material routing penalty relative to its oracle owner.")
    else:
        conclusions.append("The learned MoE top-1 route is close to its oracle owner on aggregate accuracy.")
    if blending_penalty > material_gap:
        conclusions.append("Dense probability blending materially dilutes the MoE's learned top-1 prediction.")
    elif blending_penalty < -material_gap:
        conclusions.append("Dense probability blending materially improves on learned top-1 routing.")
    else:
        conclusions.append("Dense blending and learned top-1 inference are practically close on aggregate accuracy.")

    populated_bins = confidence_bins[confidence_bins["rows"] > 0]
    best_bin = None
    if not populated_bins.empty:
        best_bin = populated_bins.loc[populated_bins["top1_minus_dense_accuracy"].idxmax()]

    lines = [
        "# Router diagnostic readout",
        "",
        f"Material-gap threshold: {material_gap * 100:.3f} percentage points.",
        "",
        "## Signed accuracy gaps",
        "",
        f"- Hard deployable minus dense MoE: {pp(deployable_gap)}.",
        f"- Hard oracle minus MoE oracle: {pp(representation_gap)}.",
        f"- MoE oracle minus MoE learned top-1: {pp(moe_routing_gap)}.",
        f"- MoE learned top-1 minus dense: {pp(blending_penalty)}.",
        f"- Hard oracle minus hard deployable: {pp(hard_routing_gap)}.",
        "",
        "## Router summary",
        "",
        f"- MoE dataset-route accuracy: {float(route_all['moe_route_accuracy']):.6f}.",
        f"- Hard dataset-route accuracy: {float(route_all['hard_route_accuracy']):.6f}.",
        f"- MoE mean maximum gate weight: {float(route_all['moe_mean_max_gate']):.6f}.",
        f"- MoE mean gate entropy: {float(route_all['moe_mean_gate_entropy']):.6f}.",
        "",
        "## Diagnostic conclusions",
        "",
        *[f"- {conclusion}" for conclusion in conclusions],
    ]
    if best_bin is not None:
        lines.extend([
            "",
            "## Confidence-bin signal",
            "",
            "- The largest observed top-1-minus-dense accuracy difference occurs for gate confidence "
            f"[{best_bin['confidence_lower_inclusive']:.1f}, {best_bin['confidence_upper_exclusive']:.1f}) "
            f"with {int(best_bin['rows'])} rows and gap {pp(float(best_bin['top1_minus_dense_accuracy']))}.",
        ])
    lines.extend([
        "",
        "This is a deterministic single-run diagnosis, not a statistical significance statement. "
        "Repeat the paired experiment across seeds before making a paper-level claim.",
    ])
    return "\n".join(lines) + "\n"
