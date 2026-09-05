"""Lightweight ablation alternative to DatasetExpertBank: a single SHARED
classification head on top of `z`, plus a small per-dataset FiLM-style
adapter (scale+shift low-rank residual) applied to `z` before the shared
head. This forces far more cross-dataset parameter sharing than independent
full expert heads, and is specifically meant to be compared against
DatasetExpertBank on the smaller datasets (e.g. BoT-IoT, individual NF-v3
members), where a fully independent expert head risks overfitting on
limited samples.

Combination rule (models/moe.py) treats this bank exactly like
DatasetExpertBank: it must expose the same `(batch, num_datasets,
num_classes)` forward() contract and `num_experts`/`dataset_names`
attributes, so MoEDatasetNIDS doesn't need to know which bank it's wired to.
"""
from __future__ import annotations

import torch
from torch import nn


class FiLMAdapter(nn.Module):
    """Per-dataset scale+shift correction to z via a low-rank bottleneck:
    z -> down-project (rank) -> up-project back to (scale, shift) of size
    latent_dim, applied as `z * (1 + scale) + shift`. Far fewer parameters
    per dataset than a full independent head.
    """

    def __init__(self, latent_dim: int, rank: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(latent_dim, rank)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, 2 * latent_dim)
        self.latent_dim = latent_dim
        # Zero-init the final projection so every adapter starts as a no-op
        # (scale=0 -> multiplier 1, shift=0), matching the shared head's
        # own behavior at initialization -- adapters only diverge from "no
        # correction" once trained to.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.down(z)))
        scale, shift = self.up(h).chunk(2, dim=-1)
        return z * (1.0 + scale) + shift


class SharedHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class AdapterExpertBank(nn.Module):
    """Drop-in replacement for DatasetExpertBank: one FiLMAdapter per
    dataset, all feeding into ONE shared classification head. ``forward`` is
    dense while ``forward_selected`` conditionally runs one adapter per row.
    """

    def __init__(
        self,
        dataset_names: list[str],
        latent_dim: int,
        num_classes: int,
        rank: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes
        self.adapters = nn.ModuleList(
            [FiLMAdapter(latent_dim, rank, dropout) for _ in dataset_names]
        )
        self.shared_head = SharedHead(latent_dim, num_classes)

    @property
    def num_experts(self) -> int:
        return len(self.adapters)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, num_datasets, num_classes)."""
        batch = z.shape[0]
        assert batch > 0
        outputs = [self.shared_head(adapter(z)) for adapter in self.adapters]
        stacked = torch.stack(outputs, dim=1)
        assert stacked.shape == (batch, self.num_experts, self.num_classes)
        return stacked

    def forward_selected(self, z: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Run one selected adapter and the shared head for each row."""
        batch = z.shape[0]
        if selected_experts.shape != (batch,):
            raise ValueError("selected_experts must have shape (batch,)")
        if batch == 0:
            raise ValueError("top-1 dispatch requires a non-empty batch")
        if int(selected_experts.min()) < 0 or int(selected_experts.max()) >= self.num_experts:
            raise ValueError("selected_experts contains an index outside the expert bank")
        output = z.new_zeros((batch, self.num_classes))
        for expert_id, adapter in enumerate(self.adapters):
            rows = torch.nonzero(selected_experts == expert_id, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            logits = self.shared_head(adapter(z.index_select(0, rows)))
            output = output.index_copy(0, rows, logits)
        return output
