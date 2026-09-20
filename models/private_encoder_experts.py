"""Dataset experts with fully private encoders.

Each branch consumes the harmonized input directly and owns both its encoder
and classification head.  This is the full-private counterpart to
``DatasetExpertBank``, whose experts share the single encoder in
``MoEDatasetNIDS`` and only own downstream heads.

The bank deliberately exposes the same dense/top-1 contract as the existing
expert banks.  ``expects_raw_input`` tells ``MoEDatasetNIDS`` and the training
helpers that the bank must receive ``x`` rather than the gate encoder's latent
representation.
"""
from __future__ import annotations

import torch
from torch import nn

from .dataset_experts import Expert
from .encoder import SharedEncoder


class PrivateEncoderExpert(nn.Module):
    """One complete dataset branch: harmonized input -> private encoder -> head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        encoder_hidden_dims: list[int],
        expert_hidden_dims: list[int],
        activation: str = "relu",
        encoder_dropout: float = 0.1,
        expert_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout=encoder_dropout,
        )
        self.head = Expert(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dims=expert_hidden_dims,
            dropout=expert_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class PrivateEncoderExpertBank(nn.Module):
    """One complete, independently parameterized encoder+head per dataset."""

    expects_raw_input = True

    def __init__(
        self,
        dataset_names: list[str],
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        encoder_hidden_dims: list[int],
        expert_hidden_dims: list[int],
        activation: str = "relu",
        encoder_dropout: float = 0.1,
        expert_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes
        self.experts = nn.ModuleList(
            [
                PrivateEncoderExpert(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                    encoder_hidden_dims=encoder_hidden_dims,
                    expert_hidden_dims=expert_hidden_dims,
                    activation=activation,
                    encoder_dropout=encoder_dropout,
                    expert_dropout=expert_dropout,
                )
                for _ in dataset_names
            ]
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def expert_names(self) -> list[str]:
        return self.dataset_names

    @property
    def encoders(self) -> list[SharedEncoder]:
        return [expert.encoder for expert in self.experts]

    def initialize_encoders(self, source: SharedEncoder) -> None:
        """Clone one Stage-A encoder into every private expert encoder."""
        source_state = source.state_dict()
        for expert in self.experts:
            expert.encoder.load_state_dict(source_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if batch == 0:
            raise ValueError("private-encoder dense routing requires a non-empty batch")
        outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(outputs, dim=1)
        assert stacked.shape == (batch, self.num_experts, self.num_classes)
        return stacked

    def forward_selected(self, x: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Run exactly one complete private branch for each row."""
        batch = x.shape[0]
        if selected_experts.shape != (batch,):
            raise ValueError("selected_experts must have shape (batch,)")
        if batch == 0:
            raise ValueError("top-1 dispatch requires a non-empty batch")
        if int(selected_experts.min()) < 0 or int(selected_experts.max()) >= self.num_experts:
            raise ValueError("selected_experts contains an index outside the expert bank")
        output = x.new_zeros((batch, self.num_classes))
        for expert_id, expert in enumerate(self.experts):
            rows = torch.nonzero(selected_experts == expert_id, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            logits = expert(x.index_select(0, rows))
            output = output.index_copy(0, rows, logits)
        return output
