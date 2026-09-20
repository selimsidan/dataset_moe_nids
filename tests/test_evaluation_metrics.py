from __future__ import annotations

import numpy as np

from evaluation.metrics import evaluate_per_dataset, evaluate_predictions
from evaluation.report import comparison_table, write_tracker_csvs


def _example():
    truth = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array([
        [0.90, 0.05, 0.05],
        [0.70, 0.20, 0.10],
        [0.10, 0.80, 0.10],
        [0.20, 0.70, 0.10],
        [0.10, 0.20, 0.70],
        [0.05, 0.05, 0.90],
    ])
    return truth, scores.argmax(axis=1), scores


def test_multiclass_roc_auc_is_reported_at_all_levels():
    truth, prediction, scores = _example()
    result = evaluate_predictions(truth, prediction, ["a", "b", "c"], scores)

    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.roc_auc_ovr_macro == 1.0
    assert result.roc_auc_ovr_weighted == 1.0
    assert result.roc_auc_ovr_micro == 1.0
    assert [metric.roc_auc_ovr for metric in result.per_class] == [1.0, 1.0, 1.0]
    assert result.pr_auc_ovr_macro == 1.0
    assert result.pr_auc_ovr_weighted == 1.0
    assert result.pr_auc_ovr_micro == 1.0
    assert [metric.pr_auc_ovr for metric in result.per_class] == [1.0, 1.0, 1.0]


def test_absent_class_auc_is_nan_but_valid_classes_are_aggregated():
    truth, prediction, scores = _example()
    keep = truth != 2
    result = evaluate_predictions(truth[keep], prediction[keep], ["a", "b", "c"], scores[keep])

    assert np.isnan(result.per_class[2].roc_auc_ovr)
    assert result.roc_auc_ovr_macro == 1.0


def test_macro_micro_and_weighted_precision_recall_f1_are_complete():
    truth = np.array([0, 0, 0, 1])
    prediction = np.array([0, 0, 1, 1])
    result = evaluate_predictions(truth, prediction, ["majority", "minority"])

    assert np.isclose(result.macro_precision, 0.75)
    assert np.isclose(result.macro_recall, 5 / 6)
    assert np.isclose(result.macro_f1, 11 / 15)
    assert np.isclose(result.micro_precision, 0.75)
    assert np.isclose(result.micro_recall, 0.75)
    assert np.isclose(result.micro_f1, 0.75)
    assert np.isclose(result.weighted_precision, 0.875)
    assert np.isclose(result.weighted_recall, 0.75)
    assert np.isclose(result.weighted_f1, 23 / 30)

    table = comparison_table({"model": result}).set_index("class")
    assert np.isclose(table.loc["MACRO_AVG (headline)", "model__precision"], 0.75)
    assert np.isclose(table.loc["MICRO_AVG", "model__f1"], 0.75)
    assert np.isclose(
        table.loc["WEIGHTED_AVG (reference only, never for model selection)", "model__f1"],
        23 / 30,
    )


def test_tracker_csvs_include_roc_auc_overall_per_dataset_and_per_class(tmp_path):
    truth, prediction, scores = _example()
    result = evaluate_predictions(truth, prediction, ["a", "b", "c"], scores)
    origins = np.array(["left", "left", "left", "right", "right", "right"])
    per_dataset = evaluate_per_dataset(
        truth, prediction, origins, ["a", "b", "c"], scores
    )
    config = {
        "model": {"gate": {"routing": "dense"}},
        "training": {"stage_c": {}},
        "data": {"active_datasets": ["left", "right"]},
        "evaluation": {"low_sample_threshold": 2},
        "run_name": "metric-test",
    }

    write_tracker_csvs(
        str(tmp_path), "trial", config, {"moe_basic": result},
        per_dataset_by_variant={"moe_basic": per_dataset},
    )

    overall_header = (tmp_path / "Overall_Metrics.csv").read_text().splitlines()[0]
    dataset_header = (tmp_path / "Per_Dataset_Metrics.csv").read_text().splitlines()[0]
    class_header = (tmp_path / "Per_Class_Metrics.csv").read_text().splitlines()[0]
    assert "roc_auc_ovr_macro" in overall_header
    assert "pr_auc_ovr_macro" in overall_header
    assert "weighted_precision" in overall_header
    assert "weighted_recall" in overall_header
    assert "weighted_f1" in overall_header
    assert "micro_precision" in overall_header
    assert "micro_recall" in overall_header
    assert "roc_auc_ovr_macro" in dataset_header
    assert "pr_auc_ovr_macro" in dataset_header
    assert "weighted_precision" in dataset_header
    assert "roc_auc_ovr" in class_header
    assert "pr_auc_ovr" in class_header
