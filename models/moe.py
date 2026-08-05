"""Wires SharedEncoder + a dataset-expert bank (DatasetExpertBank or
AdapterExpertBank -- both share the same forward() contract) + Gate into
`MoEDatasetNIDS`, and implements the combination rule: a straightforward
soft mixture-of-experts over full per-dataset class distributions.

    expert_probs    = softmax(expert_logits, dim=-1)             # (B, D, C)
    combined_probs  = einsum('bd,bdc->bc', gate_weights, expert_probs)  # (B, C)
    prediction      = argmax(combined_probs, dim=-1)

Simpler than moe_nids' MoENIDS.forward: there's no per-expert relabeling to
reconcile (every expert already predicts the full class vocabulary), so
`combined_probs` is a proper distribution (rows sum to 1) with no
"discarded OTHER mass" renormalization step.

Dataset-blind by construction: `forward(x)` takes exactly one argument.
`_assert_dataset_blind_signature` (inference/predict.py) checks this via
`inspect.signature`, mirroring moe_nids' MoENIDS -- see
tests/test_dataset_blind_inference.py.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .encoder import SharedEncoder
from .gate import Gate


class MoEDatasetNIDS(nn.Module):
    def __init__(self, encoder: SharedEncoder, expert_bank: nn.Module, gate: Gate, class_names: list[str]) -> None:
        super().__init__()
        self.encoder = encoder
        self.expert_bank = expert_bank
        self.gate = gate
        self._class_names = list(class_names)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def dataset_names(self) -> list[str]:
        return self.expert_bank.dataset_names

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        expert_logits = self.expert_bank(z)  # (B, D, C) -- every expert sees every sample
        expert_probs = F.softmax(expert_logits, dim=-1)
        gate_weights = self.gate(z)  # (B, D), softmax already applied, dense/soft (no top-k)

        combined_probs = torch.einsum("bd,bdc->bc", gate_weights, expert_probs)  # (B, C)

        return {
            "z": z,
            "expert_logits": expert_logits,
            "expert_probs": expert_probs,
            "gate_weights": gate_weights,
            "combined_probs": combined_probs,
        }

    @staticmethod
    def combine_probs_for_training(
        gate_weights: torch.Tensor,
        expert_probs: torch.Tensor,
        dataset_ids: torch.Tensor,
        expert_update_policy: str = "all",
    ) -> torch.Tensor:
        """Combine experts while optionally enforcing dataset ownership.

        ``assigned_only`` leaves the forward probabilities exactly equal to
        the ordinary soft mixture, but detaches each non-assigned expert on
        each row. The gate therefore still learns from every expert's output
        through the final task loss, while expert ``d`` receives data-driven
        gradients only from rows whose ``dataset_id == d``. Dataset IDs are
        used only to control the Stage-C backward path; inference remains the
        dataset-blind one-argument :meth:`forward` path.
        """
        if expert_update_policy == "all":
            routed_probs = expert_probs
        elif expert_update_policy == "assigned_only":
            if dataset_ids.ndim != 1 or dataset_ids.shape[0] != expert_probs.shape[0]:
                raise ValueError("dataset_ids must have shape (batch,)")
            if gate_weights.shape != expert_probs.shape[:2]:
                raise ValueError("gate_weights and expert_probs dimensions are incompatible")
            if dataset_ids.numel() and (
                int(dataset_ids.min()) < 0 or int(dataset_ids.max()) >= expert_probs.shape[1]
            ):
                raise ValueError("dataset_ids contains an expert index outside the active expert bank")
            assigned = F.one_hot(
                dataset_ids.to(torch.long), num_classes=expert_probs.shape[1]
            ).to(torch.bool).unsqueeze(-1)
            routed_probs = torch.where(assigned, expert_probs, expert_probs.detach())
        else:
            raise ValueError(
                f"Unknown expert_update_policy {expert_update_policy!r}; expected 'all' or 'assigned_only'"
            )
        return torch.einsum("bd,bdc->bc", gate_weights, routed_probs)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["combined_probs"].argmax(dim=1)

    @staticmethod
    def combined_probs_to_log_probs(combined_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """`combined_probs` is already a proper categorical distribution
        (unlike moe_nids' combined_scores, no OTHER-mass is discarded), so
        this is just a numerically-safe log -- kept as a named staticmethod
        for symmetry with moe_nids' MoENIDS.combined_scores_to_log_probs and
        so callers don't need to know that detail.
        """
        return torch.log(combined_probs + eps)
