"""Supervised representation objectives used during Stage-A pretraining.

The losses operate directly on the encoder output ``z`` because that is the
representation consumed by the MoE.  No disposable projection head is used.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import ProbeHead


def _normalized_similarity(z: torch.Tensor, temperature: float) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("embeddings must have shape (batch, latent_dim)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(z, dim=1)
    return normalized @ normalized.T / temperature


class SupervisedContrastiveLoss(nn.Module):
    """All-positive supervised contrastive loss with safe singleton handling."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarity = _normalized_similarity(z, self.temperature)
        batch = len(labels)
        if labels.shape != (batch,):
            raise ValueError("labels must have shape (batch,)")
        identity = torch.eye(batch, dtype=torch.bool, device=z.device)
        positives = labels[:, None].eq(labels[None, :]) & ~identity
        valid = positives.any(dim=1)
        if not valid.any():
            return z.sum() * 0.0
        logits = similarity.masked_fill(identity, -torch.inf)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positives.sum(dim=1).clamp_min(1)
        per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
        return per_anchor[valid].mean()


class BalancedSupervisedContrastiveLoss(nn.Module):
    """Class-averaged SupCon denominator for long-tailed batches.

    Every represented negative class contributes the mean exponentiated
    similarity of its members rather than a sum proportional to its batch
    frequency.  Used with the class/domain-balanced sampler, this gives every
    present class a comparable influence without pair or triplet mining.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarity = _normalized_similarity(z, self.temperature)
        batch = len(labels)
        identity = torch.eye(batch, dtype=torch.bool, device=z.device)
        positives = labels[:, None].eq(labels[None, :]) & ~identity
        valid = positives.any(dim=1)
        if not valid.any():
            return z.sum() * 0.0

        # Subtract a row constant before exponentiation for numerical stability.
        stable = similarity - similarity.masked_fill(identity, -torch.inf).max(dim=1, keepdim=True).values
        exp_similarity = stable.exp().masked_fill(identity, 0.0)
        class_ids = torch.unique(labels)
        class_terms = []
        for class_id in class_ids:
            members = labels.eq(class_id)[None, :].expand(batch, -1) & ~identity
            counts = members.sum(dim=1)
            class_mean = (exp_similarity * members).sum(dim=1) / counts.clamp_min(1)
            class_terms.append(torch.where(counts > 0, class_mean, torch.zeros_like(class_mean)))
        class_terms = torch.stack(class_terms, dim=1)
        denominator = class_terms.sum(dim=1).clamp_min(torch.finfo(z.dtype).tiny)
        own_columns = labels[:, None].eq(class_ids[None, :]).to(class_terms.dtype)
        own_class_mean = (class_terms * own_columns).sum(dim=1).clamp_min(torch.finfo(z.dtype).tiny)
        per_anchor = -(own_class_mean.log() - denominator.log())
        return per_anchor[valid].mean()


class CenterLoss(nn.Module):
    """Learnable class centers with mean squared within-class distance."""

    def __init__(self, num_classes: int, latent_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.empty(num_classes, latent_dim))
        nn.init.normal_(self.centers, std=0.02)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized_z = F.normalize(z, dim=1)
        normalized_centers = F.normalize(self.centers, dim=1)
        return (normalized_z - normalized_centers.index_select(0, labels)).square().sum(dim=1).mean()


class ArcMarginHead(nn.Module):
    """Additive angular-margin classifier used only while labels are known."""

    def __init__(self, latent_dim: int, num_classes: int, margin: float = 0.3, scale: float = 30.0) -> None:
        super().__init__()
        if not 0 <= margin < math.pi / 2:
            raise ValueError("ArcFace margin must be in [0, pi/2)")
        if scale <= 0:
            raise ValueError("ArcFace scale must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = float(margin)
        self.scale = float(scale)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(z, dim=1), F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
        target_cosine = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(torch.bool)
        return torch.where(one_hot, target_cosine, cosine) * self.scale


def representation_config(config: dict) -> dict:
    values = config.get("training", {}).get("representation", {})
    return {
        "objective": values.get("objective", "ce"),
        "sampling": values.get("sampling", "legacy"),
        "class_weighting": values.get("class_weighting", "legacy"),
        "weight": float(values.get("weight", 0.1)),
        "temperature": float(values.get("temperature", 0.1)),
        "center_weight": float(values.get("center_weight", values.get("weight", 0.01))),
        "arc_margin": float(values.get("arc_margin", 0.3)),
        "arc_scale": float(values.get("arc_scale", 30.0)),
    }


class StageARepresentationObjective(nn.Module):
    """Classification plus the configured representation objective."""

    VALID_OBJECTIVES = {"ce", "supcon", "balanced_supcon", "center", "arcface"}

    def __init__(self, latent_dim: int, num_classes: int, config: dict) -> None:
        super().__init__()
        self.config = representation_config(config)
        self.objective = self.config["objective"]
        if self.objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"Unknown representation objective {self.objective!r}; "
                f"expected one of {sorted(self.VALID_OBJECTIVES)}"
            )
        if self.objective == "arcface":
            self.classifier = ArcMarginHead(
                latent_dim, num_classes,
                margin=self.config["arc_margin"], scale=self.config["arc_scale"],
            )
        else:
            self.classifier = ProbeHead(latent_dim, num_classes)
        self.metric: nn.Module | None
        if self.objective == "supcon":
            self.metric = SupervisedContrastiveLoss(self.config["temperature"])
        elif self.objective == "balanced_supcon":
            self.metric = BalancedSupervisedContrastiveLoss(self.config["temperature"])
        elif self.objective == "center":
            self.metric = CenterLoss(num_classes, latent_dim)
        else:
            self.metric = None

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        logits = self.classifier(z, labels) if self.objective == "arcface" else self.classifier(z)
        ce = F.cross_entropy(logits, labels, weight=class_weights)
        metric = ce.new_zeros(())
        if self.metric is not None:
            metric = self.metric(z, labels)
        weight = self.config["center_weight"] if self.objective == "center" else self.config["weight"]
        total = ce + weight * metric
        return total, {"ce": ce, "metric": metric, "total": total}, logits
