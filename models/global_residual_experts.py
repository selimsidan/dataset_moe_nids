"""Shared global classifier plus zero-initialized dataset residual experts.

Each expert predicts ``global_head(z) + residual_e(z)``.  The global head is
initialized from the exact pooled Stage-A CE probe, while Stage B optimizes
only the owned residual.  This preserves a shared classification boundary and
makes dataset specialization an additive correction rather than an entirely
independent classifier.
"""
from __future__ import annotations

import torch
from torch import nn


class GlobalResidualExpertBank(nn.Module):
    """One shared linear head and one zero-initialized linear residual per dataset."""

    def __init__(
        self,
        dataset_names: list[str],
        latent_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.num_classes = int(num_classes)
        self.global_head = nn.Linear(latent_dim, num_classes)
        self.residuals = nn.ModuleList(
            [nn.Linear(latent_dim, num_classes) for _ in self.dataset_names]
        )
        for residual in self.residuals:
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)

    @property
    def experts(self) -> nn.ModuleList:
        """Compatibility alias used for per-branch accounting and optimization."""
        return self.residuals

    @property
    def expert_names(self) -> list[str]:
        return self.dataset_names

    @property
    def num_experts(self) -> int:
        return len(self.residuals)

    def initialize_global_head(self, representation_state: dict[str, torch.Tensor]) -> None:
        """Load the ordinary CE probe saved in the Stage-A checkpoint."""
        weight = representation_state.get("classifier.linear.weight")
        bias = representation_state.get("classifier.linear.bias")
        if weight is None or bias is None:
            raise ValueError(
                "global-residual experts require a Stage-A CE probe with "
                "classifier.linear.weight and classifier.linear.bias"
            )
        if weight.shape != self.global_head.weight.shape or bias.shape != self.global_head.bias.shape:
            raise ValueError(
                "Stage-A probe shape does not match the configured global residual head: "
                f"weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
            )
        with torch.no_grad():
            self.global_head.weight.copy_(weight)
            self.global_head.bias.copy_(bias)

    def forward_one(self, dataset_idx: int, z: torch.Tensor) -> torch.Tensor:
        return self.global_head(z) + self.residuals[dataset_idx](z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs = [self.forward_one(index, z) for index in range(self.num_experts)]
        return torch.stack(outputs, dim=1)

    def forward_selected(
        self, z: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch = z.shape[0]
        if selected_experts.shape != (batch,):
            raise ValueError("selected_experts must have shape (batch,)")
        if batch == 0:
            raise ValueError("top-1 dispatch requires a non-empty batch")
        if int(selected_experts.min()) < 0 or int(selected_experts.max()) >= self.num_experts:
            raise ValueError("selected_experts contains an index outside the expert bank")
        output = z.new_zeros((batch, self.num_classes))
        for expert_id in range(self.num_experts):
            rows = torch.nonzero(selected_experts == expert_id, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            logits = self.forward_one(expert_id, z.index_select(0, rows))
            output = output.index_copy(0, rows, logits)
        return output
