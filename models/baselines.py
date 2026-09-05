"""Baselines, sharing the harmonization layer, data pipeline, and evaluation
code with the primary architecture -- only the model layer differs, selected
by the same `architecture` config flag (see training/run.py).

- `PlainPooledSoftmax` / `NoFusionModel`: ported unchanged in spirit from
  moe_nids/models/baselines.py.
- `HardTwoStageModel`: NEW, specific to this project -- the literal "obvious
  two-step approach" this project needs to demonstrably beat. A standalone
  dataset-ID classifier (stage a) hard-routes (argmax, no blending) each
  sample to an independent per-dataset classifier (stage b, structurally
  identical to NoFusionModel). Unlike NoFusionModel, `forward(x)` takes only
  the harmonized feature tensor -- routing uses the model's OWN predicted
  dataset ID, not a caller-supplied ground-truth one, so this baseline can
  be evaluated dataset-blind at inference exactly like the primary
  architecture and compared apples-to-apples (especially on the OOD/
  ambiguity eval, evaluation/ood_ambiguity_eval.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .encoder import SharedEncoder
from .losses import DatasetIDClassifierHead


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class PlainPooledSoftmax(nn.Module):
    """Same shared encoder + harmonization pipeline as the primary
    architecture, but a single joint C-way softmax head instead of
    per-dataset experts and a gate. Isolates whether the expert/gate layer
    earns its complexity over simply pooling data into one classifier.
    """

    def __init__(self, encoder: SharedEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(encoder.latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        logits = self.head(z)
        return {"z": z, "logits": logits}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"].argmax(dim=1)


class MatchedDenseHead(nn.Module):
    """Two-hidden-layer dense comparator resolved against an MoE budget.

    ``match_info`` is intentionally stored on the module (but not in the
    state_dict) so runners and reports can record the exact integer solution
    and residual instead of describing an approximate match as exact.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, int],
        dropout: float,
        match_info: dict,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(latent_dim, first),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second, num_classes),
        )
        self.hidden_dims = (first, second)
        self.match_info = dict(match_info)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MatchedDenseClassifier(nn.Module):
    """Shared encoder plus a deterministic capacity/compute-matched head."""

    def __init__(self, encoder: SharedEncoder, head: MatchedDenseHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def match_info(self) -> dict:
        return self.head.match_info

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z, "logits": self.head(z)}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"].argmax(dim=1)


class NoFusionModel(nn.Module):
    """Fully separate encoder+head per dataset -- no cross-dataset code path
    at all. `forward` requires the caller to name which dataset's sub-model
    to run, since routing by dataset identity is exactly what this baseline
    is allowed to do (and the primary architecture is not).
    """

    def __init__(
        self,
        dataset_names: list[str],
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dims: list[int] = (256, 128),
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not dataset_names:
            raise ValueError("NoFusionModel requires at least one dataset")
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes
        self.encoders = nn.ModuleDict(
            {
                name: SharedEncoder(input_dim, hidden_dims, latent_dim, activation, dropout)
                for name in dataset_names
            }
        )
        self.heads = nn.ModuleDict(
            {name: ClassificationHead(latent_dim, num_classes) for name in dataset_names}
        )

    def forward(self, x: torch.Tensor, dataset_name: str) -> dict[str, torch.Tensor]:
        if dataset_name not in self.encoders:
            raise KeyError(f"NoFusionModel has no sub-model for dataset '{dataset_name}'")
        z = self.encoders[dataset_name](x)
        logits = self.heads[dataset_name](z)
        return {"z": z, "logits": logits}

    def predict(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        return self.forward(x, dataset_name)["logits"].argmax(dim=1)


class HardTwoStageModel(nn.Module):
    """Stage (a): a standalone dataset classifier (SharedEncoder architecture
    + DatasetIDClassifierHead, trained with plain CE against ground-truth
    dataset ID -- for a fair comparison, the same encoder architecture as
    the primary model, but its own instance/weights, never shared).
    Stage (b): independent per-dataset classifiers, structurally identical
    to NoFusionModel.

    At inference, `forward(x)` predicts dataset ID via stage (a)
    (`argmax`, no blending) and hard-routes each sample to its predicted
    dataset's stage-(b) sub-model -- no ground-truth dataset id is used or
    required, so this can be evaluated dataset-blind exactly like the
    primary architecture.
    """

    def __init__(
        self,
        dataset_names: list[str],
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        id_hidden_dims: list[int] = (256, 128),
        stage_b_hidden_dims: list[int] = (256, 128),
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not dataset_names:
            raise ValueError("HardTwoStageModel requires at least one dataset")
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes

        self.id_encoder = SharedEncoder(input_dim, id_hidden_dims, latent_dim, activation, dropout)
        self.id_head = DatasetIDClassifierHead(latent_dim, len(dataset_names))

        self.stage_b = NoFusionModel(
            dataset_names,
            input_dim,
            latent_dim,
            num_classes,
            stage_b_hidden_dims,
            activation,
            dropout,
        )

    def dataset_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Stage-(a) logits, exposed for routing diagnostics only."""
        return self.id_head(self.id_encoder(x))

    def predict_dataset_id(self, x: torch.Tensor) -> torch.Tensor:
        return self.dataset_logits(x).argmax(dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        dataset_logits = self.dataset_logits(x)
        dataset_pred = dataset_logits.argmax(dim=1)  # (B,) index into self.dataset_names
        logits = x.new_zeros(x.shape[0], self.num_classes)
        for idx, name in enumerate(self.dataset_names):
            mask = dataset_pred == idx
            if mask.any():
                logits[mask] = self.stage_b(x[mask], name)["logits"]
        return {
            "logits": logits,
            "dataset_logits": dataset_logits,
            "dataset_pred": dataset_pred,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"].argmax(dim=1)
