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


class ConfusionAdaptiveProxyMarginLoss(nn.Module):
    """Pair-specific rival margins over normalized class proxies.

    The temporary Stage-A classifier weights are reused as proxies.  Positive
    margins are added only to selected rival logits, making those rivals
    harder without changing the true-class logit.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        proxy_weights: torch.Tensor,
        margins: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if margins.shape != (proxy_weights.shape[0], proxy_weights.shape[0]):
            raise ValueError("margins must have shape (num_classes, num_classes)")
        cosine = F.linear(F.normalize(z, dim=1), F.normalize(proxy_weights, dim=1))
        sample_margins = margins.index_select(0, labels).detach()
        logits = (cosine + sample_margins) / self.temperature
        return F.cross_entropy(logits, labels, weight=class_weights)


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


def stage_a_encoder_role(config: dict) -> str:
    """Return the encoder role trained by the current Stage-A invocation.

    Historical runs omit this key and continue to train ``model.encoder``.
    Fixed asymmetric studies can instead create distinct, contract-checked
    gate/private initialization checkpoints without changing the Stage-A API.
    """
    role = str(config.get("training", {}).get("stage_a_encoder_role", "encoder"))
    if role not in {"encoder", "gate_encoder", "private_encoder"}:
        raise ValueError(
            "training.stage_a_encoder_role must be encoder, gate_encoder, or private_encoder"
        )
    return role


def representation_config(config: dict, encoder_role: str | None = None) -> dict:
    training = config.get("training", {})
    values = dict(training.get("representation", {}) or {})
    role = encoder_role or stage_a_encoder_role(config)
    role_overrides = (training.get("representation_by_role", {}) or {}).get(role, {}) or {}
    values.update(role_overrides)
    resolved = {
        "objective": values.get("objective", "ce"),
        "sampling": values.get("sampling", "legacy"),
        "class_weighting": values.get("class_weighting", "legacy"),
        "weight": float(values.get("weight", 0.1)),
        "temperature": float(values.get("temperature", 0.1)),
        "center_weight": float(values.get("center_weight", values.get("weight", 0.01))),
        "arc_margin": float(values.get("arc_margin", 0.3)),
        "arc_scale": float(values.get("arc_scale", 30.0)),
    }
    if resolved["objective"] == "confusion_adaptive_margin":
        resolved.update({
            "confusion_warmup_epochs": int(values.get("confusion_warmup_epochs", 5)),
            "confusion_ema": float(values.get("confusion_ema", 0.5)),
            "confusion_shrinkage": float(values.get("confusion_shrinkage", 100.0)),
            "confusion_top_k": int(values.get("confusion_top_k", 2)),
            "confusion_max_margin": float(values.get("confusion_max_margin", 0.15)),
        })
    return resolved


class StageARepresentationObjective(nn.Module):
    """Classification plus the configured representation objective."""

    VALID_OBJECTIVES = {
        "ce", "supcon", "balanced_supcon", "center", "arcface",
        "confusion_adaptive_margin",
    }

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        config: dict,
        encoder_role: str | None = None,
    ) -> None:
        super().__init__()
        self.config = representation_config(config, encoder_role)
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
        elif self.objective == "confusion_adaptive_margin":
            self.metric = ConfusionAdaptiveProxyMarginLoss(self.config["temperature"])
        else:
            self.metric = None
        self.num_classes = int(num_classes)
        if self.objective == "confusion_adaptive_margin":
            if self.config["confusion_warmup_epochs"] < 1:
                raise ValueError("confusion_warmup_epochs must be at least 1")
            if not 0 <= self.config["confusion_ema"] < 1:
                raise ValueError("confusion_ema must be in [0, 1)")
            if self.config["confusion_shrinkage"] < 0:
                raise ValueError("confusion_shrinkage must be non-negative")
            if not 1 <= self.config["confusion_top_k"] < num_classes:
                raise ValueError("confusion_top_k must be in [1, num_classes)")
            if self.config["confusion_max_margin"] < 0:
                raise ValueError("confusion_max_margin must be non-negative")
            self.register_buffer("confusion_raw", torch.zeros(num_classes, num_classes))
            self.register_buffer("confusion_shrunk", torch.zeros(num_classes, num_classes))
            self.register_buffer("confusion_ema", torch.zeros(num_classes, num_classes))
            self.register_buffer("confusion_margins", torch.zeros(num_classes, num_classes))
            self.register_buffer("confusion_initialized", torch.tensor(False, dtype=torch.bool))
        self._confusion_probability_sum: torch.Tensor | None = None
        self._confusion_support: torch.Tensor | None = None
        self._confusion_epoch = 0
        self._confusion_proxy_active = False

    @property
    def confusion_proxy_active(self) -> bool:
        return bool(self._confusion_proxy_active)

    def begin_confusion_epoch(self, epoch: int) -> None:
        """Reset detached accumulators for a one-based Stage-A epoch."""
        self._confusion_epoch = int(epoch)
        self._confusion_proxy_active = bool(
            self.objective == "confusion_adaptive_margin"
            and epoch > self.config.get("confusion_warmup_epochs", 5)
            and self.confusion_initialized.item()
        )
        if self.objective != "confusion_adaptive_margin":
            self._confusion_probability_sum = None
            self._confusion_support = None
            return
        device = self.confusion_ema.device
        self._confusion_probability_sum = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.float64, device=device
        )
        self._confusion_support = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )

    @torch.no_grad()
    def accumulate_confusion(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate ordinary-head soft predictions without autograd state."""
        if self.objective != "confusion_adaptive_margin":
            return
        if self._confusion_probability_sum is None or self._confusion_support is None:
            raise RuntimeError("begin_confusion_epoch must be called before accumulation")
        probabilities = logits.detach().softmax(dim=1).to(torch.float64)
        labels = labels.detach()
        indicator = F.one_hot(labels, num_classes=self.num_classes).to(torch.float64)
        # Matrix multiplication avoids repeated-index CUDA atomics, preserving
        # compatibility with deterministic-algorithm experiment runs.
        self._confusion_probability_sum.add_(indicator.T @ probabilities)
        self._confusion_support.add_(indicator.sum(dim=0))

    @torch.no_grad()
    def finalize_confusion_epoch(self) -> dict[str, torch.Tensor] | None:
        """Create the matrix used by the next epoch and return report tensors."""
        if self.objective != "confusion_adaptive_margin":
            return None
        if self._confusion_probability_sum is None or self._confusion_support is None:
            raise RuntimeError("begin_confusion_epoch must be called before finalization")
        support = self._confusion_support
        raw = self._confusion_probability_sum / support.clamp_min(1.0).unsqueeze(1)
        raw.fill_diagonal_(0.0)

        total_support = support.sum().clamp_min(1.0)
        global_rival = (raw * support.unsqueeze(1)).sum(dim=0) / total_support
        prior = global_rival.unsqueeze(0).expand_as(raw).clone()
        prior.fill_diagonal_(0.0)
        shrinkage = float(self.config["confusion_shrinkage"])
        denominator = support + shrinkage
        reliability = torch.where(
            denominator > 0,
            support / denominator.clamp_min(torch.finfo(support.dtype).tiny),
            torch.zeros_like(support),
        )
        shrunk = reliability.unsqueeze(1) * raw + (1.0 - reliability).unsqueeze(1) * prior
        shrunk.fill_diagonal_(0.0)

        warmup = int(self.config["confusion_warmup_epochs"])
        if self._confusion_epoch >= warmup:
            if self.confusion_initialized.item():
                alpha = float(self.config["confusion_ema"])
                ema = alpha * self.confusion_ema.to(torch.float64) + (1.0 - alpha) * shrunk
            else:
                ema = shrunk
            ema.fill_diagonal_(0.0)
            margins_cpu = torch.zeros_like(ema, device="cpu")
            top_k = min(int(self.config["confusion_top_k"]), max(0, self.num_classes - 1))
            if top_k:
                ranking = ema.detach().cpu().clone()
                ranking.fill_diagonal_(-torch.inf)
                indices = torch.argsort(
                    ranking, dim=1, descending=True, stable=True
                )[:, :top_k]
                values = ranking.gather(1, indices)
                selected = values.clamp(max=float(self.config["confusion_max_margin"]))
                margins_cpu.scatter_(1, indices, selected)
            margins = margins_cpu.to(ema.device)
            margins.fill_diagonal_(0.0)
            self.confusion_ema.copy_(ema.to(self.confusion_ema.dtype))
            self.confusion_margins.copy_(margins.to(self.confusion_margins.dtype))
            self.confusion_initialized.fill_(True)

        self.confusion_raw.copy_(raw.to(self.confusion_raw.dtype))
        self.confusion_shrunk.copy_(shrunk.to(self.confusion_shrunk.dtype))
        snapshot = {
            "epoch": torch.tensor(self._confusion_epoch),
            "raw": raw.cpu(),
            "shrunk": shrunk.cpu(),
            "ema": self.confusion_ema.detach().to(torch.float64).cpu(),
            "margins": self.confusion_margins.detach().to(torch.float64).cpu(),
            "support": support.cpu(),
            "reliability": reliability.cpu(),
            "global_rival": global_rival.cpu(),
            "initialized": self.confusion_initialized.detach().cpu(),
        }
        self._confusion_probability_sum = None
        self._confusion_support = None
        return snapshot

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        logits = self.classifier(z, labels) if self.objective == "arcface" else self.classifier(z)
        ce = F.cross_entropy(logits, labels, weight=class_weights)
        metric = ce.new_zeros(())
        if self.objective == "confusion_adaptive_margin" and self.confusion_proxy_active:
            if not isinstance(self.classifier, ProbeHead):
                raise TypeError("confusion-adaptive margins require the ordinary Stage-A probe")
            metric = self.metric(
                z, labels, self.classifier.linear.weight,
                self.confusion_margins, class_weights,
            )
        elif self.metric is not None and self.objective != "confusion_adaptive_margin":
            metric = self.metric(z, labels)
        weight = self.config["center_weight"] if self.objective == "center" else self.config["weight"]
        total = ce + weight * metric
        return total, {"ce": ce, "metric": metric, "total": total}, logits
