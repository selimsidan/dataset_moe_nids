"""Reducing `data.active_datasets` to a subset must work with zero code
changes -- every module (registry/harmonizer, DatasetExpertBank sizing,
checkpoint shapes) must derive its dataset list/count from that one config
field, never a hardcoded number. Uses the synthetic FakeA/FakeB fixtures
from conftest.py (not real registry entries), so this runs without any
dataset files on disk.
"""
import torch

from data.harmonization import Harmonizer, TrainSplit
from models.adapters import AdapterExpertBank
from models.dataset_experts import DatasetExpertBank
from training.checkpoint import load_stage_b, save_stage_b
from training.model_utils import build_expert_bank, build_model


def test_harmonizer_adapts_to_dataset_subset(dataset_specs, synthetic_frames):
    dataset_specs["FakeA"].feature_alias.pop("src_ip_lookalike")  # not the leakage test's concern here

    full = Harmonizer(["FakeA", "FakeB"], specs=dataset_specs)
    subset = Harmonizer(["FakeA"], specs={"FakeA": dataset_specs["FakeA"]})

    # FakeA's own private feature ("unique_to_a") is "common" once FakeB is
    # dropped from the active set (only one dataset left, so every one of
    # its features is trivially present in "every active dataset").
    assert "unique_to_a" in subset.common_features
    assert subset.unique_features == []
    assert full.output_width != subset.output_width

    subset.fit([TrainSplit("FakeA", synthetic_frames["FakeA"])])
    out = subset.transform(synthetic_frames["FakeA"], "FakeA")
    assert out.shape == (len(synthetic_frames["FakeA"]), subset.output_width)


def _model_cfg():
    return {
        "latent_dim": 8,
        "expert": {"hidden_dims": [16], "dropout": 0.0},
        "adapter": {"rank": 4, "dropout": 0.0},
        "gate": {"hidden_dims": []},
    }


def test_dataset_expert_bank_size_derives_from_active_datasets_list():
    for names in (["A", "B"], ["A", "B", "C", "D", "E"]):
        bank = build_expert_bank("full", names, latent_dim=8, num_classes=5, model_cfg=_model_cfg())
        assert isinstance(bank, DatasetExpertBank)
        assert bank.num_experts == len(names)

        z = torch.randn(10, 8)
        out = bank(z)
        assert out.shape == (10, len(names), 5)


def test_adapter_bank_size_derives_from_active_datasets_list():
    for names in (["A", "B"], ["A", "B", "C"]):
        bank = build_expert_bank("adapter", names, latent_dim=8, num_classes=4, model_cfg=_model_cfg())
        assert isinstance(bank, AdapterExpertBank)
        assert bank.num_experts == len(names)

        z = torch.randn(6, 8)
        out = bank(z)
        assert out.shape == (6, len(names), 4)


def test_gate_output_dim_matches_active_dataset_count():
    from models.encoder import SharedEncoder

    for names in (["A", "B"], ["A", "B", "C", "D"]):
        encoder = SharedEncoder(input_dim=12, latent_dim=8)
        model = build_model("moe_dataset_soft", encoder, names, ["Benign", "c1"], _model_cfg())
        z = torch.randn(5, 8)
        weights = model.gate(z)
        assert weights.shape == (5, len(names))


def test_checkpoint_round_trip_preserves_reduced_dataset_subset(tmp_path):
    names = ["A", "B"]
    bank = build_expert_bank("full", names, latent_dim=8, num_classes=4, model_cfg=_model_cfg())
    save_stage_b(str(tmp_path), bank.state_dict(), names, bank_kind="full")

    ckpt = load_stage_b(str(tmp_path))
    assert ckpt["dataset_names"] == names
    assert ckpt["bank_kind"] == "full"

    rebuilt = build_expert_bank("full", ckpt["dataset_names"], latent_dim=8, num_classes=4, model_cfg=_model_cfg())
    rebuilt.load_state_dict(ckpt["expert_bank_state"])
    assert rebuilt.num_experts == len(names)
