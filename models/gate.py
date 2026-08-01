"""Dataset-gate: z -> softmax weights over dataset-experts. Small linear or
shallow MLP, config-selectable -- same pattern as moe_nids' Gate, just
softmax'd over `num_datasets` instead of `num_classes` experts.

Non-negotiable design constraint (see models/moe.py / training/losses.py):
this module never sees or is supervised directly by ground-truth dataset
ID as its PRIMARY training signal. It is optimized primarily so the
downstream combined task prediction is correct; any dataset-ID-aware loss
term is applied outside this module, at low weight, in training/losses.py.
"""
from __future__ import annotations

import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, latent_dim: int, num_datasets: int, hidden_dims: list[int] = ()) -> None:
        super().__init__()
        if not hidden_dims:
            self.net = nn.Linear(latent_dim, num_datasets)
        else:
            dims = [latent_dim, *hidden_dims]
            layers: list[nn.Module] = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            layers.append(nn.Linear(dims[-1], num_datasets))
            self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        return torch.softmax(logits, dim=-1)
