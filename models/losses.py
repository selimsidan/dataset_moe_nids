"""Gate load-balancing penalty (ported unchanged from moe_nids/models/losses.py)
plus the low-weight, OPTIONAL dataset-ID auxiliary regularizer for the gate.

IMPORTANT -- non-negotiable design constraint: `dataset_aux_loss` must never
be the PRIMARY training signal for the gate. It exists only as a low-weight
regularizer (`training.stage_c.lambda_dataset_aux`, default 0.05-0.1,
analogous in role/magnitude to moe_nids' `lambda_align=0.1` in Stage C), or,
for the explicit `hard` ablation, at a deliberately large weight so the
gate approximates a real dataset classifier for comparison against
`hard_two_stage` -- never as the recommended default. See
training/stage_c_jointfinetune.py for how `gate_supervision` selects the
weight, and tests/test_no_dataset_id_supervision.py for the structural
check that `lambda_dataset_aux=0.0` truly removes dataset_id from the
gate's loss computation graph.
"""
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
    """CE(gate_weights, dataset_id) -- the ONLY place ground-truth dataset
    identity is allowed to influence the gate, and only ever behind a
    caller-supplied low weight (see module docstring). `gate_weights` are
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
