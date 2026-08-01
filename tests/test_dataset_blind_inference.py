"""Same pattern as moe_nids' dataset-blind check: MoEDatasetNIDS.forward(x)
must take exactly one argument, checked via inspect.signature -- so any
future code change threading a dataset identifier into forward() fails this
assertion immediately, rather than silently leaking dataset identity into
the model at inference time.
"""
import inspect

from models.adapters import AdapterExpertBank
from models.dataset_experts import DatasetExpertBank
from models.encoder import SharedEncoder
from models.gate import Gate
from models.moe import MoEDatasetNIDS


def _params(model: MoEDatasetNIDS) -> list[str]:
    return [p for p in inspect.signature(model.forward).parameters if p != "self"]


def test_moe_forward_signature_is_dataset_blind_with_full_expert_bank():
    encoder = SharedEncoder(input_dim=10, latent_dim=8)
    bank = DatasetExpertBank(["DatasetA", "DatasetB", "DatasetC"], latent_dim=8, num_classes=5)
    gate = Gate(8, bank.num_experts)
    model = MoEDatasetNIDS(encoder, bank, gate, class_names=["Benign", "c1", "c2", "c3", "c4"])

    assert _params(model) == ["x"], "MoEDatasetNIDS.forward must not accept a dataset-identity argument"


def test_moe_forward_signature_is_dataset_blind_with_adapter_bank():
    encoder = SharedEncoder(input_dim=10, latent_dim=8)
    bank = AdapterExpertBank(["DatasetA", "DatasetB"], latent_dim=8, num_classes=4)
    gate = Gate(8, bank.num_experts)
    model = MoEDatasetNIDS(encoder, bank, gate, class_names=["Benign", "c1", "c2", "c3"])

    assert _params(model) == ["x"]


def test_every_dataset_expert_sees_full_batch():
    import torch

    latent_dim = 8
    bank = DatasetExpertBank(["DatasetA", "DatasetB", "DatasetC"], latent_dim, num_classes=6)
    batch = torch.randn(37, latent_dim)  # deliberately not a "round" batch size
    out = bank(batch)
    assert out.shape == (37, 3, 6)  # (batch, num_datasets, num_classes) -- no sample dropped
