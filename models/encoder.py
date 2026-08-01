"""The one shared encoder. Exactly one instance is ever constructed per run,
shared across all datasets and (after Stage A) all training stages -- there
is no dataset-selection branch anywhere in this module, by construction: it
takes only a harmonized feature vector, never a dataset identifier.

Ported unchanged from moe_nids/models/encoder.py -- the encoder's role
(dataset-agnostic latent representation) is identical in this project; only
what sits on top of `z` (dataset-experts instead of class-experts) differs.
"""
from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class SharedEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128),
        latent_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), act_cls(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProbeHead(nn.Module):
    """Stage-A-only multiclass probe on top of z. Discarded after Stage A --
    never persisted into or loaded by Stage B/C checkpoints."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
