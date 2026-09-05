"""Wires SharedEncoder + a dataset-expert bank (DatasetExpertBank or
AdapterExpertBank -- both share the same forward() contract) + Gate into
`MoEDatasetNIDS`, and implements selectable dense or true top-1 routing.

    expert_probs    = softmax(expert_logits, dim=-1)             # (B, D, C)
    combined_probs  = einsum('bd,bdc->bc', gate_weights, expert_probs)  # (B, C)
    prediction      = argmax(combined_probs, dim=-1)

The dense path uses the following mixture:

Simpler than moe_nids' MoENIDS.forward: there's no per-expert relabeling to
reconcile (every expert already predicts the full class vocabulary), so
`combined_probs` is a proper distribution (rows sum to 1) with no
"discarded OTHER mass" renormalization step.

The top-1 path groups rows by gate argmax and invokes only their selected
expert. Dataset-blind by construction: `forward(x)` takes exactly one argument.
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
    def __init__(
        self,
        encoder: SharedEncoder,
        expert_bank: nn.Module,
        gate: Gate,
        class_names: list[str],
        routing_mode: str = "dense",
    ) -> None:
        super().__init__()
        if routing_mode not in {"dense", "top1"}:
            raise ValueError("routing_mode must be 'dense' or 'top1'")
        self.encoder = encoder
        self.expert_bank = expert_bank
        self.gate = gate
        self._class_names = list(class_names)
        self.routing_mode = routing_mode

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def dataset_names(self) -> list[str]:
        return self.expert_bank.dataset_names

    @property
    def expert_names(self) -> list[str]:
        return list(getattr(self.expert_bank, "expert_names", self.expert_bank.dataset_names))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        gate_weights = self.gate(z)
        if self.routing_mode == "top1":
            selected_experts = gate_weights.argmax(dim=1)
            selected_logits = self.expert_bank.forward_selected(z, selected_experts)
            selected_probs = F.softmax(selected_logits, dim=-1)
            return {
                "z": z,
                "gate_weights": gate_weights,
                "selected_experts": selected_experts,
                "selected_logits": selected_logits,
                "selected_probs": selected_probs,
                "combined_probs": selected_probs,
            }

        expert_logits = self.expert_bank(z)
        expert_probs = F.softmax(expert_logits, dim=-1)
        combined_probs = torch.einsum("bd,bdc->bc", gate_weights, expert_probs)
        return {
            "z": z,
            "expert_logits": expert_logits,
            "expert_probs": expert_probs,
            "gate_weights": gate_weights,
            "combined_probs": combined_probs,
        }

    @staticmethod
    def combine_top1_for_training(
        gate_weights: torch.Tensor,
        selected_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        dataset_ids: torch.Tensor | None,
        expert_update_policy: str = "all",
    ) -> torch.Tensor:
        """Top-1 task probabilities with optional strict dataset ownership.

        The ratio has forward value one but carries a task gradient into the
        selected gate probability. Passing detached gate weights (DAMEX) removes
        that gradient. Under ``assigned_only``, misrouted rows update no expert.
        """
        batch = selected_probs.shape[0]
        if gate_weights.shape[0] != batch or selected_experts.shape != (batch,):
            raise ValueError("top-1 routing tensors have incompatible batch dimensions")
        if expert_update_policy == "all":
            routed_probs = selected_probs
        elif expert_update_policy == "assigned_only":
            if dataset_ids is None or dataset_ids.shape != (batch,):
                raise ValueError("dataset_ids must have shape (batch,) for assigned_only updates")
            owned = (selected_experts == dataset_ids).unsqueeze(-1)
            routed_probs = torch.where(owned, selected_probs, selected_probs.detach())
        else:
            raise ValueError(
                f"Unknown expert_update_policy {expert_update_policy!r}; expected 'all' or 'assigned_only'"
            )
        selected_weight = gate_weights.gather(1, selected_experts.unsqueeze(1)).squeeze(1)
        straight_through_scale = selected_weight / selected_weight.detach().clamp_min(1e-12)
        return routed_probs * straight_through_scale.unsqueeze(1)

    @staticmethod
    def combine_probs_for_training(
        gate_weights: torch.Tensor,
        expert_probs: torch.Tensor,
        dataset_ids: torch.Tensor | None,
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
            if dataset_ids is None or dataset_ids.ndim != 1 or dataset_ids.shape[0] != expert_probs.shape[0]:
                raise ValueError("dataset_ids must have shape (batch,) for assigned_only updates")
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
