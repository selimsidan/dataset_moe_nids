"""Stage C -- Joint fine-tune.

Instantiates the Gate (fresh init -- it has no prior stage), unfreezes the
encoder per `training.stage_c_unfreeze`, and trains end-to-end with:

    loss = CE(combined_probs, y_class)
         + lambda_dataset_aux * CE(gate_weights, dataset_id)   [see below]
         + lambda_balance     * load_balance_penalty(gate_weights)

`lambda_dataset_aux` and whether the dataset-aux term is even active are
controlled by `training.stage_c.gate_supervision`:

    "none"      -> lambda = 0.0 (pure task-loss-driven gate)
    "light_aux" -> lambda = training.stage_c.lambda_dataset_aux (default 0.05-0.1;
                   analogous role/magnitude to moe_nids' Stage C lambda_align=0.1)
    "hard"      -> lambda = training.stage_c.lambda_dataset_aux_hard (large;
                   ablation only, makes the gate loss dominant enough to
                   approximate a real dataset classifier -- never the
                   recommended default, see models/losses.py docstring)

This is the ONLY place ground-truth dataset_id may influence gate training,
and only ever through this single, explicitly-weighted term -- never as
`gate_weights`' primary supervision. See
tests/test_no_dataset_id_supervision.py for the structural check that
`gate_supervision: none` truly removes dataset_id from the gate's
computation graph.

Batches are drawn via ClassBalancedBatchSampler over the TASK class label
(not dataset ID) -- guaranteeing a configurable minimum count of every
canonical class, including rare ones, in every batch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder import SharedEncoder
from models.losses import dataset_aux_loss, load_balance_penalty
from models.moe import MoEDatasetNIDS

from .checkpoint import clear_progress, load_progress, load_stage_a, load_stage_b, save_progress, save_stage_c
from .dataset import HarmonizedTensorDataset, PreparedData
from .model_utils import bank_kind_for_architecture, build_model
from .sampler import ClassBalancedBatchSampler


def _set_encoder_trainable(encoder: SharedEncoder, mode: str) -> None:
    if mode == "none":
        for p in encoder.parameters():
            p.requires_grad_(False)
    elif mode == "all":
        for p in encoder.parameters():
            p.requires_grad_(True)
    elif mode == "last_layer":
        for p in encoder.parameters():
            p.requires_grad_(False)
        last_linear = encoder.net[-1]
        for p in last_linear.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown stage_c_unfreeze mode: {mode}")


def _lambda_dataset_aux(config: dict) -> float:
    stage_c_cfg = config["training"]["stage_c"]
    mode = stage_c_cfg.get("gate_supervision", "light_aux")
    if mode == "none":
        return 0.0
    if mode == "light_aux":
        return float(stage_c_cfg.get("lambda_dataset_aux", 0.1))
    if mode == "hard":
        return float(stage_c_cfg.get("lambda_dataset_aux_hard", 5.0))
    raise ValueError(f"Unknown training.stage_c.gate_supervision '{mode}'. Expected none | light_aux | hard.")


def build_model_from_checkpoints(config: dict, data: PreparedData, device: torch.device) -> MoEDatasetNIDS:
    model_cfg = config["model"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    architecture = config["architecture"]

    encoder = SharedEncoder(
        input_dim=data.train.features.shape[1],
        hidden_dims=model_cfg["encoder"]["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        activation=model_cfg["encoder"]["activation"],
        dropout=model_cfg["encoder"]["dropout"],
    ).to(device)
    encoder.load_state_dict(load_stage_a(checkpoint_dir)["encoder_state"])

    model = build_model(architecture, encoder, data.active_datasets, data.class_names, model_cfg).to(device)
    model.expert_bank.load_state_dict(load_stage_b(checkpoint_dir)["expert_bank_state"])
    return model


def run_stage_c(config: dict, data: PreparedData) -> MoEDatasetNIDS:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))

    model = build_model_from_checkpoints(config, data, device)
    _set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])

    lambda_balance = config["load_balance"]["lambda_balance"]
    lambda_dataset_aux = _lambda_dataset_aux(config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["training"]["lr"], weight_decay=config["training"].get("weight_decay", 0.0))

    dataset = HarmonizedTensorDataset(data.train)
    sampler = ClassBalancedBatchSampler(
        data.train.class_idx,
        batch_size=config["training"]["batch_size"],
        min_per_class=config["training"]["min_per_class_per_batch"],
        seed=config.get("seed", 0),
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_every = config["training"].get("checkpoint_every_n_epochs", 1)
    bank_kind = bank_kind_for_architecture(config["architecture"])

    start_epoch = 0
    progress = load_progress(checkpoint_dir, "C")
    if progress is not None:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = progress["epoch"]
        print(f"[Stage C] resuming from epoch {start_epoch} (found existing progress checkpoint)")

    for epoch in range(start_epoch, config["training"]["epochs_c"]):
        totals = {"ce": 0.0, "balance": 0.0, "dataset_aux": 0.0}
        n_batches = 0
        for features, class_idx, dataset_idx in loader:
            features, class_idx, dataset_idx = features.to(device), class_idx.to(device), dataset_idx.to(device)

            out = model(features)  # every expert sees the full batch -- no filtering, ever
            log_probs = MoEDatasetNIDS.combined_probs_to_log_probs(out["combined_probs"])
            ce = F.nll_loss(log_probs, class_idx)
            balance = load_balance_penalty(out["gate_weights"])

            loss = ce + lambda_balance * balance
            aux_value = 0.0
            if lambda_dataset_aux > 0.0:
                aux = dataset_aux_loss(out["gate_weights"], dataset_idx)
                loss = loss + lambda_dataset_aux * aux
                aux_value = aux.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["ce"] += ce.item()
            totals["balance"] += balance.item()
            totals["dataset_aux"] += aux_value
            n_batches += 1
        print(
            f"[Stage C] epoch {epoch}: "
            f"CE={totals['ce'] / n_batches:.4f} balance={totals['balance'] / n_batches:.4f} "
            f"dataset_aux={totals['dataset_aux'] / n_batches:.4f} (lambda={lambda_dataset_aux})"
        )

        if (epoch + 1) % checkpoint_every == 0:
            save_progress(checkpoint_dir, "C", {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })

    save_stage_c(checkpoint_dir, model.state_dict(), data.class_names, data.active_datasets, bank_kind)
    clear_progress(checkpoint_dir, "C")
    return model
