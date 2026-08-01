"""One expert per active DATASET (not per attack class, unlike moe_nids).

Each expert proposes a full classification over the entire canonical class
vocabulary (Benign + every active_classes entry) -- there is no relabeling
step here, since dataset-experts aren't solving a per-class sub-problem the
way moe_nids' class-experts solve a target/benign/other sub-problem; each
one is independently proposing a full answer to "what is this flow".

`DatasetExpertBank.forward` runs every expert on every sample
unconditionally -- same hard structural constraint as moe_nids'
ExpertBank ("every expert sees every sample"): no conditional skipping, no
routing/filtering step, ever.
"""
from __future__ import annotations

import torch
from torch import nn


class Expert(nn.Module):
    """One expert for one dataset: z -> num_classes logits (full task vocabulary)."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dims: list[int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        dims = [latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DatasetExpertBank(nn.Module):
    """Holds one Expert per active dataset. Every expert sees every sample
    in the batch on every forward pass -- there is no routing or sample
    filtering here, structurally: forward() runs all experts on the full
    input `z` unconditionally. Number of experts is always
    `len(dataset_names)`, derived at construction time -- never hardcoded.
    """

    def __init__(
        self,
        dataset_names: list[str],
        latent_dim: int,
        num_classes: int,
        hidden_dims: list[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes
        self.experts = nn.ModuleList(
            [Expert(latent_dim, num_classes, hidden_dims, dropout) for _ in dataset_names]
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, num_datasets, num_classes)."""
        batch = z.shape[0]
        assert batch > 0
        outputs = [expert(z) for expert in self.experts]  # every expert runs on the full batch, unconditionally
        stacked = torch.stack(outputs, dim=1)
        assert stacked.shape == (batch, self.num_experts, self.num_classes)
        return stacked
