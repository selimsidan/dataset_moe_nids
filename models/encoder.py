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
from copy import deepcopy

from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def resolve_encoder_config(model_cfg: dict, role: str = "encoder") -> dict:
    """Resolve a role-specific encoder config with ``model.encoder`` fallback.

    ``gate_encoder`` and ``private_encoder`` are v3 opt-ins.  Keeping the
    fallback here (rather than at every call site) makes every historical
    configuration construct exactly the same MLP and state-dict layout.
    """
    if role not in {"encoder", "gate_encoder", "private_encoder"}:
        raise ValueError(f"Unknown encoder role {role!r}")
    base = deepcopy(model_cfg["encoder"])
    if role != "encoder":
        base.update(deepcopy(model_cfg.get(role, {}) or {}))
    return base


def _normalization(name: str, width: int) -> nn.Module:
    normalized = str(name).lower()
    if normalized in {"batchnorm", "batchnorm1d"}:
        return nn.BatchNorm1d(width)
    if normalized in {"layernorm", "layer_norm"}:
        return nn.LayerNorm(width)
    raise ValueError("residual encoder normalization must be batchnorm or layernorm")


class ResidualMLPBlock(nn.Module):
    """Pre-normalized two-layer residual block for tabular representations."""

    def __init__(
        self,
        width: int,
        expansion: float = 2.0,
        normalization: str = "layernorm",
        activation: str = "relu",
        dropout: float = 0.1,
        dropout_second: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = max(1, int(round(width * float(expansion))))
        self.norm = _normalization(normalization, width)
        self.linear_in = nn.Linear(width, hidden)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden, width)
        self.dropout_second = nn.Dropout(dropout_second)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        x = self.dropout_second(x)
        return residual + x


class SharedEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128),
        latent_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        *,
        kind: str = "mlp",
        width: int = 128,
        blocks: int = 2,
        expansion: float = 2.0,
        normalization: str = "layernorm",
        dropout_second: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.kind = str(kind)
        act_cls = _ACTIVATIONS[activation]
        if self.kind == "mlp":
            dims = [input_dim, *hidden_dims]
            layers: list[nn.Module] = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(in_d, out_d), act_cls(), nn.Dropout(dropout)]
            layers.append(nn.Linear(dims[-1], latent_dim))
        elif self.kind == "residual_mlp":
            if int(width) <= 0 or int(blocks) <= 0 or float(expansion) <= 0:
                raise ValueError("residual encoder width, blocks, and expansion must be positive")
            layers = [nn.Linear(input_dim, int(width))]
            layers.extend(
                ResidualMLPBlock(
                    int(width), expansion, normalization, activation, dropout, dropout_second
                )
                for _ in range(int(blocks))
            )
            layers.extend([
                _normalization(normalization, int(width)),
                act_cls(),
                nn.Linear(int(width), latent_dim),
            ])
        else:
            raise ValueError("encoder kind must be 'mlp' or 'residual_mlp'")
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def last_layer_parameters(self):
        """Parameters updated by ``stage_c_unfreeze=last_layer``."""
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                return layer.parameters()
        raise RuntimeError("encoder contains no linear projection")


def build_encoder(input_dim: int, latent_dim: int, encoder_cfg: dict) -> SharedEncoder:
    """Construct an encoder from either legacy MLP or v3 residual settings."""
    kind = encoder_cfg.get("kind", "mlp")
    return SharedEncoder(
        input_dim=input_dim,
        hidden_dims=list(encoder_cfg.get("hidden_dims", [])),
        latent_dim=latent_dim,
        activation=encoder_cfg.get("activation", "relu"),
        dropout=float(encoder_cfg.get("dropout", 0.1)),
        kind=kind,
        width=int(encoder_cfg.get("width", 128)),
        blocks=int(encoder_cfg.get("blocks", 2)),
        expansion=float(encoder_cfg.get("expansion", 2.0)),
        normalization=encoder_cfg.get("normalization", "layernorm"),
        dropout_second=float(encoder_cfg.get("dropout_second", 0.0)),
    )


class ProbeHead(nn.Module):
    """Stage-A-only multiclass probe on top of z. Discarded after Stage A --
    never persisted into or loaded by Stage B/C checkpoints."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
