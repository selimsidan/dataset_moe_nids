"""Stage B -- selectable expert-bank initialization/warm-start.

``training.stage_b.warmstart_mode=random_init`` saves a randomly initialized
generic expert bank without reading dataset membership. This is the native
``moe_basic`` path: experts and router begin joint task training in Stage C.

With ``warmstart_mode=dataset`` (the unchanged dataset-MoE path), the encoder
is frozen and each dataset-expert
is trained independently, on a class-balanced view of ONLY its own
dataset's rows, against the full task label (no relabeling -- unlike
moe_nids' per-class target/benign/other relabeling, each dataset-expert
here is directly answering "what is this flow" over the whole canonical
class vocabulary).

This stage is deliberately the "N separate models" baseline in disguise --
same cold-start rationale as moe_nids' Stage B (see its ARCHITECTURE.md
Sec. 7): don't make the gate choose between one competent expert and N
randomly-initialized ones. Number of experts/datasets trained here is
`len(active_datasets)`, derived at runtime -- never hardcoded.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder import SharedEncoder

from .checkpoint import (
    clear_progress, load_progress, load_validated_stage_a, save_progress,
    save_stage_b, stage_a_metadata,
)
from .dataset import HarmonizedTensorDataset, PreparedData
from .model_utils import (
    bank_kind_for_architecture,
    build_expert_bank,
    expert_forward_one,
    expert_names_for_architecture,
    expert_train_params,
    initialize_private_expert_encoders,
)
from .sampler import ClassBalancedBatchSampler


class _RemappedBatchSampler:
    """Wraps a ClassBalancedBatchSampler built over a dataset-local label
    array so its yielded indices are remapped into the pooled dataset's
    global row indices. A plain class (not a generator function) so it can
    be iterated once per epoch across Stage B's multi-epoch loop -- a bare
    generator would be exhausted after the first epoch.
    """

    def __init__(self, sampler: ClassBalancedBatchSampler, row_indices) -> None:
        self.sampler = sampler
        self.row_indices = row_indices

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        for batch in self.sampler:
            yield self.row_indices[batch].tolist()


def _build_frozen_encoder(config: dict, data: PreparedData, device: torch.device) -> SharedEncoder:
    ckpt = load_validated_stage_a(config, stage_a_metadata(config, data))
    model_cfg = config["model"]
    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1],
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


def run_stage_b(config: dict, data: PreparedData):
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))
    active_datasets = data.active_datasets
    model_cfg = config["model"]
    bank_kind = bank_kind_for_architecture(config["architecture"])
    input_dim = data.train.features.shape[1] if bank_kind == "private_encoder" else None

    expert_names = expert_names_for_architecture(config["architecture"], active_datasets)
    expert_bank = build_expert_bank(
        bank_kind,
        expert_names,
        model_cfg["latent_dim"],
        len(data.class_names),
        model_cfg,
        input_dim=input_dim,
    ).to(device)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    warmstart_mode = config["training"].get("stage_b", {}).get("warmstart_mode", "dataset")
    if warmstart_mode == "random_init":
        print(
            "[Stage B] basic MoE: saving randomly initialized generic experts; "
            "no dataset is assigned to or used to warm-start an expert"
        )
        save_stage_b(
            checkpoint_dir,
            expert_bank.state_dict(),
            expert_names,
            bank_kind,
            training_summary={
                "optimizer_steps": 0,
                "examples_seen": 0,
                "epochs_completed": 0,
                "wall_seconds": 0.0,
                "selected_epoch": 0,
            },
        )
        clear_progress(checkpoint_dir, "B")
        return expert_bank
    if warmstart_mode != "dataset":
        raise ValueError("training.stage_b.warmstart_mode must be 'dataset' or 'random_init'")

    encoder = _build_frozen_encoder(config, data, device)
    initialize_private_expert_encoders(expert_bank, encoder)

    dataset = HarmonizedTensorDataset(data.train)
    all_class_idx = data.train.class_idx
    all_dataset_idx = data.train.dataset_idx

    checkpoint_every = config["training"].get("checkpoint_every_n_epochs", 1)

    resume_dataset_i, resume_epoch, resume_optimizer_state = 0, 0, None
    optimizer_steps = examples_seen = 0
    started = time.monotonic()
    progress = load_progress(checkpoint_dir, "B")
    if progress is not None:
        expert_bank.load_state_dict(progress["expert_bank_state"])
        resume_dataset_i = progress["dataset_i"]
        resume_epoch = progress["epoch"]
        resume_optimizer_state = progress["optimizer_state"]
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
        print(f"[Stage B] resuming from dataset[{resume_dataset_i}] epoch {resume_epoch} (found existing progress checkpoint)")

    for i, dataset_name in enumerate(active_datasets):
        if i < resume_dataset_i:
            continue

        row_mask = all_dataset_idx == i
        if row_mask.sum() == 0:
            print(f"[Stage B] WARNING: dataset '{dataset_name}' has no training rows -- skipping its expert warm-start.")
            continue
        row_indices = row_mask.nonzero()[0]
        sub_labels = all_class_idx[row_indices]

        sampler = ClassBalancedBatchSampler(
            sub_labels,
            batch_size=config["training"]["batch_size"],
            min_per_class=config["training"]["min_per_class_per_batch"],
            seed=config.get("seed", 0) + i,
        )
        # `sampler` indexes into `sub_labels` (this dataset's own rows only);
        # remap to the pooled dataset's global row indices before fetching,
        # since `dataset` holds every active dataset's rows pooled together.
        loader = DataLoader(dataset, batch_sampler=_RemappedBatchSampler(sampler, row_indices))

        params = expert_train_params(expert_bank, i)
        optimizer = torch.optim.Adam(params, lr=config["training"]["lr"])
        expert_start_epoch = 0
        if i == resume_dataset_i and resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
            expert_start_epoch = resume_epoch

        for epoch in range(expert_start_epoch, config["training"]["epochs_b"]):
            total_loss, n_batches = 0.0, 0
            for features, class_idx, _dataset_idx in loader:
                features, class_idx = features.to(device), class_idx.to(device)
                if getattr(expert_bank, "expects_raw_input", False):
                    expert_input = features
                else:
                    with torch.no_grad():
                        expert_input = encoder(features)
                logits = expert_forward_one(expert_bank, i, expert_input)
                loss = F.cross_entropy(logits, class_idx)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                examples_seen += len(class_idx)
                total_loss += loss.item()
                n_batches += 1
            print(f"[Stage B] expert[{dataset_name}] epoch {epoch}: CE={total_loss / max(1, n_batches):.4f}")

            if (epoch + 1) % checkpoint_every == 0:
                save_progress(checkpoint_dir, "B", {
                    "dataset_i": i,
                    "epoch": epoch + 1,
                    "expert_bank_state": expert_bank.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "optimizer_steps": optimizer_steps,
                    "examples_seen": examples_seen,
                })

    save_stage_b(
        checkpoint_dir,
        expert_bank.state_dict(),
        expert_names,
        bank_kind,
        training_summary={
            "optimizer_steps": optimizer_steps,
            "examples_seen": examples_seen,
            "epochs_completed": config["training"]["epochs_b"],
            "wall_seconds": time.monotonic() - started,
            "selected_epoch": config["training"]["epochs_b"],
        },
    )
    clear_progress(checkpoint_dir, "B")
    return expert_bank
