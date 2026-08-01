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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder import ProbeHead, SharedEncoder

from .checkpoint import clear_progress, load_progress, save_progress, save_stage_a
from .dataset import HarmonizedTensorDataset, PreparedData


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
    probe = ProbeHead(model_cfg["latent_dim"], num_classes).to(device)

    params = list(encoder.parameters()) + list(probe.parameters())
    optimizer = torch.optim.Adam(params, lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0))

    loader = DataLoader(
        HarmonizedTensorDataset(data.train),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_every = config["training"].get("checkpoint_every_n_epochs", 1)

    start_epoch = 0
    progress = load_progress(checkpoint_dir, "A")
    if progress is not None:
        encoder.load_state_dict(progress["encoder_state"])
        probe.load_state_dict(progress["probe_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = progress["epoch"]
        print(f"[Stage A] resuming from epoch {start_epoch} (found existing progress checkpoint)")

    for epoch in range(start_epoch, config["training"]["epochs_a"]):
        total_ce, n_batches = 0.0, 0
        for features, class_idx, _dataset_idx in loader:
            features, class_idx = features.to(device), class_idx.to(device)
            z = encoder(features)
            logits = probe(z)
            ce = F.cross_entropy(logits, class_idx)

            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

            total_ce += ce.item()
            n_batches += 1
        print(f"[Stage A] epoch {epoch}: CE={total_ce / n_batches:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            save_progress(checkpoint_dir, "A", {
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "probe_state": probe.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

    save_stage_a(checkpoint_dir, encoder.state_dict(), data.class_names)
    clear_progress(checkpoint_dir, "A")
    return encoder
