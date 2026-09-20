"""Stage C -- Joint fine-tune.

Instantiates the Gate (fresh init -- it has no prior stage), unfreezes the
encoder per `training.stage_c_unfreeze`, and trains end-to-end with:

    loss = CE(combined_probs, y_class)
         + lambda_dataset_aux * CE(gate_weights, dataset_id)   [see below]
         + lambda_balance     * load_balance_penalty(gate_weights)
         + lambda_expert_anchor * MSE(experts, stage_b_experts)

`lambda_dataset_aux` and whether the dataset-aux term is even active are
controlled by `training.stage_c.gate_supervision`:

    "none"      -> lambda = 0.0 (pure task-loss-driven gate)
    "light_aux" -> lambda = training.stage_c.lambda_dataset_aux (default 0.05-0.1;
                   analogous role/magnitude to moe_nids' Stage C lambda_align=0.1)
    "hard"      -> lambda = training.stage_c.lambda_dataset_aux_hard (large;
                   ablation only, makes the gate loss dominant enough to
                   approximate a real dataset classifier -- never the
                   recommended default, see models/losses.py docstring)
    "damex"     -> lambda = training.stage_c.lambda_dataset_aux_damex; the
                   task mixture uses detached gate weights, so dataset CE and
                   load balancing are the only losses that update the gate.

This is the ONLY place ground-truth dataset_id may directly supervise the
gate. It is auxiliary in the existing modes and primary only in the explicit
``damex`` mode. With ``expert_update_policy`` set to ``assigned_only``, dataset
IDs additionally mask the backward path into the expert bank without changing
mixture values. In DAMEX mode, detaching task gate weights separately removes
the gate's task-loss gradient.
See
tests/test_no_dataset_id_supervision.py for the structural check that
`gate_supervision: none` truly removes dataset_id from the gate's
computation graph.

Batches are drawn via ClassBalancedBatchSampler over the TASK class label
(not dataset ID) -- guaranteeing a configurable minimum count of every
canonical class, including rare ones, in every batch.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder import SharedEncoder
from models.losses import dataset_aux_loss, load_balance_penalty
from models.moe import MoEDatasetNIDS

from .checkpoint import (
    clear_progress, load_progress, load_validated_stage_a, load_stage_b,
    save_progress, save_stage_c, stage_a_metadata,
)
from .dataset import HarmonizedTensorDataset, PreparedData
from .model_utils import bank_kind_for_architecture, build_model, model_encoder_modules
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
    if mode == "damex":
        return float(stage_c_cfg.get("lambda_dataset_aux_damex", 1.0))
    raise ValueError(
        f"Unknown training.stage_c.gate_supervision '{mode}'. "
        "Expected none | light_aux | hard | damex."
    )


def _gate_weights_for_task(gate_weights: torch.Tensor, gate_supervision: str) -> torch.Tensor:
    """Keep the downstream task objective out of the router graph in DAMEX mode."""
    return gate_weights.detach() if gate_supervision == "damex" else gate_weights


def _expert_anchor(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.expert_bank.named_parameters()
    }


def _expert_anchor_penalty(model, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    values = []
    parameter_count = 0
    for name, parameter in model.expert_bank.named_parameters():
        values.append((parameter - anchor[name]).square().sum())
        parameter_count += parameter.numel()
    if not values or parameter_count == 0:
        raise ValueError("expert bank has no parameters to anchor")
    return torch.stack(values).sum() / parameter_count


def _validation_macro_f1(model, data: PreparedData, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model.predict(torch.from_numpy(data.val.features).to(device)).cpu().numpy()
    truth = data.val.class_idx
    classes = len(data.class_names)
    confusion = np.bincount(truth * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=(precision + recall) != 0)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


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
    encoder.load_state_dict(load_validated_stage_a(config, stage_a_metadata(config, data))["encoder_state"])

    model = build_model(architecture, encoder, data.active_datasets, data.class_names, model_cfg).to(device)
    model.expert_bank.load_state_dict(load_stage_b(checkpoint_dir)["expert_bank_state"])
    return model


def run_stage_c(config: dict, data: PreparedData) -> MoEDatasetNIDS:
    device = torch.device(config["training"].get("device", "cpu"))
    torch.manual_seed(config.get("seed", 0))

    model = build_model_from_checkpoints(config, data, device)
    stage_b_anchor = _expert_anchor(model)
    for encoder in model_encoder_modules(model):
        _set_encoder_trainable(encoder, config["training"]["stage_c_unfreeze"])

    lambda_balance = config["load_balance"]["lambda_balance"]
    lambda_dataset_aux = _lambda_dataset_aux(config)
    stage_c_cfg = config["training"]["stage_c"]
    gate_supervision = stage_c_cfg.get("gate_supervision", "light_aux")
    expert_update_policy = stage_c_cfg.get("expert_update_policy", "all")
    if expert_update_policy not in {"all", "assigned_only"}:
        raise ValueError("training.stage_c.expert_update_policy must be 'all' or 'assigned_only'")
    bank_kind = bank_kind_for_architecture(config["architecture"])
    if expert_update_policy == "assigned_only" and bank_kind not in {"full", "private_encoder"}:
        raise ValueError("assigned_only expert updates require independent full experts, not a shared adapter head")
    lambda_expert_anchor = float(stage_c_cfg.get("lambda_expert_anchor", 0.0))
    if lambda_expert_anchor < 0:
        raise ValueError("training.stage_c.lambda_expert_anchor must be non-negative")

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
    start_epoch = 0
    optimizer_steps = examples_seen = 0
    started = time.monotonic()
    selection_mode = config["training"].get("selection_mode", "fixed_epochs")
    if selection_mode not in {"fixed_epochs", "best_val"}:
        raise ValueError("training.selection_mode must be fixed_epochs or best_val")
    best_score = -float("inf")
    best_state = None
    best_epoch = None
    patience_left = int(config["training"].get("early_stopping_patience", 5))
    progress = load_progress(checkpoint_dir, "C")
    if progress is not None:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = progress["epoch"]
        optimizer_steps = int(progress.get("optimizer_steps", 0))
        examples_seen = int(progress.get("examples_seen", 0))
        best_score = float(progress.get("best_score", best_score))
        best_state = progress.get("best_state")
        best_epoch = progress.get("best_epoch")
        patience_left = int(progress.get("patience_left", patience_left))
        print(f"[Stage C] resuming from epoch {start_epoch} (found existing progress checkpoint)")

    completed_epoch = start_epoch
    for epoch in range(start_epoch, config["training"]["epochs_c"]):
        completed_epoch = epoch + 1
        totals = {"ce": 0.0, "balance": 0.0, "dataset_aux": 0.0, "anchor": 0.0}
        n_batches = 0
        for features, class_idx, dataset_idx in loader:
            features, class_idx, dataset_idx = features.to(device), class_idx.to(device), dataset_idx.to(device)

            out = model(features)  # dense runs all experts; top1 dispatches one per row
            task_gate_weights = _gate_weights_for_task(out["gate_weights"], gate_supervision)
            ownership_ids = dataset_idx if expert_update_policy == "assigned_only" else None
            if model.routing_mode == "top1":
                training_probs = MoEDatasetNIDS.combine_top1_for_training(
                    task_gate_weights,
                    out["selected_probs"],
                    out["selected_experts"],
                    ownership_ids,
                    expert_update_policy,
                )
            else:
                training_probs = MoEDatasetNIDS.combine_probs_for_training(
                    task_gate_weights, out["expert_probs"], ownership_ids, expert_update_policy
                )
            log_probs = MoEDatasetNIDS.combined_probs_to_log_probs(training_probs)
            ce = F.nll_loss(log_probs, class_idx)
            balance = load_balance_penalty(out["gate_weights"])

            loss = ce + lambda_balance * balance
            anchor = _expert_anchor_penalty(model, stage_b_anchor) if lambda_expert_anchor > 0 else ce.new_zeros(())
            loss = loss + lambda_expert_anchor * anchor
            aux_value = 0.0
            if lambda_dataset_aux > 0.0:
                aux = dataset_aux_loss(out["gate_weights"], dataset_idx)
                loss = loss + lambda_dataset_aux * aux
                aux_value = aux.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(class_idx)

            totals["ce"] += ce.item()
            totals["balance"] += balance.item()
            totals["dataset_aux"] += aux_value
            totals["anchor"] += anchor.item()
            n_batches += 1
        print(
            f"[Stage C] epoch {epoch}: "
            f"CE={totals['ce'] / n_batches:.4f} balance={totals['balance'] / n_batches:.4f} "
            f"dataset_aux={totals['dataset_aux'] / n_batches:.4f} anchor={totals['anchor'] / n_batches:.6f} "
            f"policy={expert_update_policy} gate_supervision={gate_supervision} "
            f"(dataset_aux_lambda={lambda_dataset_aux}, anchor_lambda={lambda_expert_anchor})"
        )

        if selection_mode == "best_val":
            score = _validation_macro_f1(model, data, device)
            if score > best_score + 1e-12:
                best_score, best_epoch = score, epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience_left = int(config["training"].get("early_stopping_patience", 5))
            else:
                patience_left -= 1
            print(f"[Stage C] val_macro_f1={score:.6f} best={best_score:.6f} patience_left={patience_left}")

        if (epoch + 1) % checkpoint_every == 0:
            save_progress(checkpoint_dir, "C", {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "examples_seen": examples_seen,
                "best_score": best_score,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "patience_left": patience_left,
            })
        if selection_mode == "best_val" and patience_left <= 0:
            break

    if selection_mode == "best_val" and best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "epochs_completed": completed_epoch,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": best_epoch if selection_mode == "best_val" else completed_epoch,
    }
    save_stage_c(
        checkpoint_dir,
        model.state_dict(),
        data.class_names,
        data.active_datasets,
        bank_kind,
        training_summary=summary,
    )
    model.training_summary = {"C": summary}
    clear_progress(checkpoint_dir, "C")
    return model
