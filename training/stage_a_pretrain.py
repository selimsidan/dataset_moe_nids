"""Stage A -- Encoder pretraining.

Shared encoder + a temporary multiclass probe head, trained on ALL pooled
training data (every class, every dataset) with plain classification CE. No
dataset-related loss term at this stage -- this project doesn't use
moe_nids' cross-dataset alignment loss (out of scope here; the
representation-sharing this project cares about is delegated to the shared
encoder + gate/expert combination learned in Stage C, not an explicit
latent-space alignment penalty). The probe head is discarded after this
stage; only the encoder is persisted.
"""
from __future__ import annotations

import time
import torch
from torch.utils.data import DataLoader

from models.encoder import SharedEncoder
from models.representation_losses import StageARepresentationObjective, representation_config

from .checkpoint import clear_progress, load_progress, save_progress, save_stage_a, stage_a_metadata
from .dataset import HarmonizedTensorDataset, PreparedData
from .sampler import ClassDomainBalancedBatchSampler


def run_stage_a(config: dict, data: PreparedData) -> SharedEncoder:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))

    input_dim = data.train.features.shape[1]
    num_classes = len(data.class_names)
    model_cfg = config["model"]

    encoder = SharedEncoder(
        input_dim=input_dim,
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    objective = StageARepresentationObjective(
        model_cfg["latent_dim"], num_classes, config
    ).to(device)

    params = list(encoder.parameters()) + list(objective.parameters())
    optimizer = torch.optim.Adam(params, lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0))

    sampling = representation_config(config)["sampling"]
    if sampling == "legacy":
        loader = DataLoader(
            HarmonizedTensorDataset(data.train),
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )
    elif sampling == "class_domain_balanced":
        sampler = ClassDomainBalancedBatchSampler(
            data.train.class_idx,
            data.train.dataset_idx,
            batch_size=config["training"]["batch_size"],
            min_per_class=config["training"]["min_per_class_per_batch"],
            seed=config.get("seed", 0),
        )
        loader = DataLoader(HarmonizedTensorDataset(data.train), batch_sampler=sampler)
    else:
        raise ValueError("training.representation.sampling must be legacy or class_domain_balanced")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_every = config["training"].get("checkpoint_every_n_epochs", 1)

    start_epoch = 0
    optimizer_steps = 0
    examples_seen = 0
    started = time.monotonic()
    progress = load_progress(checkpoint_dir, "A")
    if progress is not None:
        encoder.load_state_dict(progress["encoder_state"])
        if "objective_state" in progress:
            objective.load_state_dict(progress["objective_state"])
        else:
            objective.classifier.load_state_dict(progress["probe_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = progress["epoch"]
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
        print(f"[Stage A] resuming from epoch {start_epoch} (found existing progress checkpoint)")

    for epoch in range(start_epoch, config["training"]["epochs_a"]):
        totals = {"ce": 0.0, "metric": 0.0, "total": 0.0}
        n_batches = 0
        for features, class_idx, _dataset_idx in loader:
            features, class_idx = features.to(device), class_idx.to(device)
            z = encoder(features)
            loss, parts, _logits = objective(z, class_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_steps += 1
            examples_seen += len(class_idx)

            for name in totals:
                totals[name] += float(parts[name].item())
            n_batches += 1
        print(
            f"[Stage A] epoch {epoch}: objective={objective.objective} "
            f"CE={totals['ce'] / n_batches:.4f} metric={totals['metric'] / n_batches:.4f} "
            f"total={totals['total'] / n_batches:.4f}"
        )

        if (epoch + 1) % checkpoint_every == 0:
            save_progress(checkpoint_dir, "A", {
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "objective_state": objective.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "examples_seen": examples_seen,
            })

    metadata = stage_a_metadata(config, data)
    metadata["training_summary"] = {
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "epochs_completed": config["training"]["epochs_a"],
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": config["training"]["epochs_a"],
    }
    save_stage_a(
        checkpoint_dir,
        encoder.state_dict(),
        data.class_names,
        metadata=metadata,
        representation_state=objective.state_dict(),
        representation_config=representation_config(config),
    )
    clear_progress(checkpoint_dir, "A")
    return encoder
