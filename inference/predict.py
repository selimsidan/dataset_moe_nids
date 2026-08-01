"""Inference-only entry point: single shared-encoder path, no dataset ID
anywhere in `MoEDatasetNIDS.forward`.

The one place a dataset name is used here is `harmonizer.transform`, which
maps a raw row onto the fixed-width harmonized vector using that source's
FITTED SCALER -- a fixed, offline deployment-schema choice, not a per-sample
runtime decision fed into the model. `_assert_dataset_blind_signature`
checks this structurally: `MoEDatasetNIDS.forward` has exactly one argument
besides `self`, mirroring moe_nids/inference/predict.py.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import torch

from data.harmonization import Harmonizer
from models.encoder import SharedEncoder
from models.moe import MoEDatasetNIDS

from training.checkpoint import load_harmonizer, load_stage_c
from training.model_utils import build_model


def _assert_dataset_blind_signature(model: MoEDatasetNIDS) -> None:
    sig = inspect.signature(model.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["x"], (
        f"MoEDatasetNIDS.forward must take exactly one argument (the harmonized feature tensor); "
        f"found {params}. A dataset-identity parameter here would violate the no-leakage constraint."
    )


def load_inference_model(config: dict) -> tuple[MoEDatasetNIDS, Harmonizer, list[str]]:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    harmonizer = load_harmonizer(checkpoint_dir)
    ckpt = load_stage_c(checkpoint_dir)
    class_names = ckpt["class_names"]
    dataset_names = ckpt["dataset_names"]
    model_cfg = config["model"]

    encoder = SharedEncoder(
        input_dim=harmonizer.output_width,
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    )
    model = build_model(config["architecture"], encoder, dataset_names, class_names, model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _assert_dataset_blind_signature(model)
    return model, harmonizer, class_names


def predict(model: MoEDatasetNIDS, harmonizer: Harmonizer, raw_rows: pd.DataFrame, source_dataset_name: str) -> np.ndarray:
    """`source_dataset_name` selects which fitted per-dataset scaler to use
    when harmonizing -- a schema/deployment-configuration detail, resolved
    before this function is called in a real deployment. It is used only
    here, for harmonization, and never reaches `model.forward`.
    """
    features = harmonizer.transform(raw_rows, source_dataset_name)
    with torch.no_grad():
        preds = model.predict(torch.from_numpy(features))
    return preds.numpy()


def predict_soft(model: MoEDatasetNIDS, harmonizer: Harmonizer, raw_rows: pd.DataFrame, source_dataset_name: str) -> dict[str, np.ndarray]:
    """Same as `predict`, but also returns the soft gate weights and
    combined class probabilities -- useful for inspecting how the gate
    blended experts on ambiguous/OOD traffic instead of just the argmax
    prediction (see evaluation/ood_ambiguity_eval.py, gate_analysis.py).
    """
    features = harmonizer.transform(raw_rows, source_dataset_name)
    with torch.no_grad():
        out = model(torch.from_numpy(features))
    return {
        "prediction": out["combined_probs"].argmax(dim=1).numpy(),
        "gate_weights": out["gate_weights"].numpy(),
        "combined_probs": out["combined_probs"].numpy(),
    }


def main() -> None:
    """CLI entry point:

        python -m inference.predict --config config/default.yaml \\
            --input-csv /path/to/raw_rows.csv --source-dataset NF-UNSW-NB15-v3
    """
    import argparse

    from training.config import load_config
    from training.logging_utils import tee_stdout_to_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--input-csv", required=True, help="Raw CSV of rows in --source-dataset's schema.")
    parser.add_argument(
        "--source-dataset", required=True,
        help="A data/registry.py DATASET_REGISTRY key -- selects which fitted per-dataset scaler/harmonization to apply.",
    )
    parser.add_argument("--output-csv", default=None, help="Defaults to <checkpoint_dir>/predictions.csv")
    parser.add_argument("--n-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    tee_stdout_to_file(os.path.join(checkpoint_dir, "infer.log"))

    model, harmonizer, class_names = load_inference_model(config)
    raw_rows = pd.read_csv(args.input_csv, nrows=args.n_rows)
    preds = predict(model, harmonizer, raw_rows, args.source_dataset)
    pred_labels = [class_names[p] for p in preds]

    output_csv = args.output_csv or os.path.join(checkpoint_dir, "predictions.csv")
    pd.DataFrame({"prediction": pred_labels}).to_csv(output_csv, index=False)
    print(f"[infer] wrote {len(pred_labels)} predictions to {output_csv}")


if __name__ == "__main__":
    main()
