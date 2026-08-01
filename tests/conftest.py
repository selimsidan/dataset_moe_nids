"""Synthetic, self-contained fixtures -- tests never touch real CSVs or
data/registry.py's real DATASET_REGISTRY entries, so they run fast and
don't depend on the user's Google Drive layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.registry import DatasetSpec  # noqa: E402


@pytest.fixture
def dataset_specs() -> dict[str, DatasetSpec]:
    spec_a = DatasetSpec(
        name="FakeA",
        kind="file",
        default_paths=[],
        label_col="Label",
        benign_label="Benign",
        feature_alias={
            "duration": "dur_a",
            "bytes": "bytes_a",
            "unique_to_a": "special_a_feature",
            "src_ip_lookalike": "IPV4_SRC_ADDR",  # deliberately identity-like, for the leakage test
        },
    )
    spec_b = DatasetSpec(
        name="FakeB",
        kind="file",
        default_paths=[],
        label_col="Label",
        benign_label="Benign",
        feature_alias={
            "duration": "duration_b",
            "bytes": "bytes_b",
            "unique_to_b": "special_b_feature",
        },
    )
    return {"FakeA": spec_a, "FakeB": spec_b}


def _synthetic_frame(n: int, seed: int, kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Benign", "AttackX", "AttackY"], size=n, p=[0.6, 0.3, 0.1])
    if kind == "a":
        return pd.DataFrame(
            {
                "dur_a": rng.normal(10, 2, n),
                "bytes_a": rng.normal(1000, 200, n),
                "special_a_feature": rng.normal(5, 1, n),
                "IPV4_SRC_ADDR": ["10.0.0.1"] * n,
                "Label": labels,
            }
        )
    return pd.DataFrame(
        {
            "duration_b": rng.normal(20, 4, n),
            "bytes_b": rng.normal(2000, 300, n),
            "special_b_feature": rng.normal(-3, 1, n),
            "Label": labels,
        }
    )


@pytest.fixture
def synthetic_frames() -> dict[str, pd.DataFrame]:
    return {"FakeA": _synthetic_frame(200, seed=1, kind="a"), "FakeB": _synthetic_frame(150, seed=2, kind="b")}
