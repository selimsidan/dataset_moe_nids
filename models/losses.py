"""Gate load balancing and optional dataset-identity supervision losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def load_balance_penalty(gate_weights: torch.Tensor) -> torch.Tensor:
    """Coefficient-of-variation penalty on mean per-expert utilization
    within a batch: 0 when every dataset-expert gets equal average gate
    weight, growing as utilization concentrates onto a subset (gate
    collapse). Ported unchanged from moe_nids/models/losses.py -- the
    formulation doesn't depend on what the experts specialize in.
    """
    mean_utilization = gate_weights.mean(dim=0)
    cv = mean_utilization.std() / (mean_utilization.mean() + 1e-12)
    return cv**2


def dataset_aux_loss(gate_weights: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
    """CE(gate_weights, dataset_id), the direct DAMEX routing target.

    In task-primary modes this is absent or auxiliary. In ``damex`` mode it
    is the gate's semantic training objective. `gate_weights` are
    already softmax'd (Gate.forward applies softmax), so this uses NLLLoss
    on their log rather than F.cross_entropy (which expects raw logits).
    """
    log_probs = torch.log(gate_weights + 1e-12)
    return F.nll_loss(log_probs, dataset_id)


class DatasetIDClassifierHead(nn.Module):
    """Standalone dataset-ID classifier head, used ONLY by the `hard_two_stage`
    baseline (models/baselines.py) -- a real, explicitly-supervised dataset
    classifier, structurally separate from the primary architecture's Gate.
    """

    def __init__(self, latent_dim: int, num_datasets: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_datasets)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
