# dataset_moe_nids — Technical Architecture

This document explains the actual model architecture, gating mechanism,
expert design, training procedure, and data pipeline implemented in this
repository, as of the current code in `models/`, `training/`, and `data/`.
It is a companion to the top-level `README.md`, written at the same
implementation-level depth as `moe_nids/docs/ARCHITECTURE.md`, which this
project reuses the data pipeline from unchanged.

## 1. Problem framing

Same underlying research program as `moe_nids`: a single NIDS classifier
trained jointly across heterogeneous public NIDS datasets that never
conditions on which dataset a sample came from at inference time. Different
objective, though: `moe_nids` targets rare-class pooling (lifting recall on
underperforming attack classes by pooling cross-dataset evidence per
class). This project targets **negative transfer** -- the loss in
per-dataset performance a single pooled model suffers relative to a model
that could specialize per dataset, while still sharing whatever
representation genuinely transfers across datasets. The architecture
mirrors that difference exactly: **one expert per DATASET** (not per
attack class), combined via a soft gate that is optimized primarily so the
downstream task prediction is correct -- never primarily so the gate
recovers dataset identity.

## 2. High-level architecture

```
raw row (dataset-specific schema)
        │
        ▼
 ┌───────────────┐
 │  Harmonizer    │  ported from moe_nids, unchanged: per-dataset alias mapping +
 │ (data/         │  per-dataset-fit StandardScaler → fixed-width vector:
 │  harmonization)│  [common | unique | presence_mask]
 └──────┬────────┘
        │ x  (harmonized feature vector, same width regardless of source dataset)
        ▼
 ┌───────────────┐
 │ SharedEncoder  │  MLP: x → z  (single instance, shared across ALL datasets)
 │ (models/       │
 │  encoder.py)   │
 └──────┬────────┘
        │ z  (latent_dim = 64)
        ├─────────────────────────────┬───────────────────────────┐
        ▼                             ▼                           │
 ┌────────────────┐           ┌───────────────┐                   │
 │ DatasetExpertBank│           │     Gate       │                  │
 │ (or Adapter-    │           │  z → softmax   │                  │
 │  ExpertBank)     │           │  over D        │                  │
 │  D experts, one  │           │  datasets      │                  │
 │  PER DATASET,    │           │  (models/      │                  │
 │  each predicts   │           │  gate.py)      │                  │
 │  the FULL task   │           └──────┬────────┘                  │
 │  vocabulary       │                  │                           │
 └──────┬────────┘                     │                           │
        │ (B, D, C) logits              │ (B, D) weights α (dense)  │
        └──────────────┬───────────────┘                           │
                        ▼                                          │
              combination rule (models/moe.py)                     │
       expert_probs   = softmax(expert_logits, dim=-1)              │
       combined_probs = einsum('bd,bdc->bc', α, expert_probs)       │
                        │                                          │
                        ▼                                          │
              combined_probs  (B, C)  →  argmax → predicted class ─┘
              (C = 1 + len(active_classes), class order [Benign, *active_classes])
```

The model is `MoEDatasetNIDS` (`models/moe.py`), wiring exactly three
submodules: `SharedEncoder`, an expert bank (`DatasetExpertBank` or
`AdapterExpertBank`), and `Gate`.

## 3. The shared encoder

`models/encoder.py::SharedEncoder` — **ported unchanged** from
`moe_nids/models/encoder.py`. Same role: a plain feed-forward MLP
(`input_dim → hidden_dims → latent_dim`, default `hidden_dims=[256,128]`,
`latent_dim=64`), exactly one instance per run, shared across every dataset
and every downstream expert/gate computation. `ProbeHead` is a Stage-A-only
scratch classification head, discarded after Stage A.

## 4. Dataset-experts — architecture and semantics

`models/dataset_experts.py`

This is where the project diverges structurally from `moe_nids`. Instead of
one expert per canonical attack class solving a fixed 3-way
target/benign/other sub-problem, this project gives **one expert per active
DATASET**, each proposing a **full classification** over the entire
canonical class vocabulary (`Benign` + every `active_classes` entry):

- `Expert(latent_dim, num_classes, hidden_dims=(128,64), dropout=0.1)`: small
  MLP, `latent_dim → hidden_dims → num_classes` logits over the FULL task
  vocabulary. There is no relabeling step anywhere in this module — unlike
  `moe_nids`'s `relabel_for_expert`, which deterministically maps ground
  truth onto a 3-way per-expert sub-problem, a dataset-expert's target is
  already the ground-truth class label as-is. Dataset-experts aren't
  solving a per-class sub-problem; each is independently proposing a
  complete answer to "what is this flow."
- `DatasetExpertBank` holds all `D = len(active_datasets)` experts in an
  `nn.ModuleList`, sized at construction time from
  `len(dataset_names)` — never hardcoded. Its `forward` runs **every
  expert on every sample in the batch, unconditionally** (same hard
  structural constraint as `moe_nids`'s `ExpertBank`: "every expert sees
  every sample," enforced with no conditional skip in the forward pass) —
  output shape `(batch, D, num_classes)`.

### `AdapterExpertBank` — the lightweight ablation alternative

`models/adapters.py`

A single **shared** classification head sits on top of `z`; each dataset
gets a small `FiLMAdapter` (a low-rank bottleneck producing a per-dataset
`scale`/`shift` correction: `z' = z * (1 + scale) + shift`, zero-initialized
so every adapter starts as a no-op) applied to `z` before the shared head.
This forces far more cross-dataset parameter sharing than
`DatasetExpertBank`'s fully independent per-dataset heads, and is
specifically meant to be compared against it on the smaller datasets (e.g.
`NF-BoT-IoT-v3`, individual `NF-v3` members) where a fully independent
expert head risks overfitting on limited samples. `AdapterExpertBank`
exposes the exact same `(batch, D, num_classes)` forward contract and
`dataset_names`/`num_experts` attributes as `DatasetExpertBank`, so
`MoEDatasetNIDS` and every training stage are agnostic to which bank
they're wired to — selected via `training/model_utils.py::build_expert_bank`,
the one place `architecture` maps onto a concrete bank class.

## 5. The gate

`models/gate.py::Gate`

`z → softmax over D dataset-experts`. Same config-selectable
linear-or-shallow-MLP pattern as `moe_nids`'s `Gate`
(`model.gate.hidden_dims`), just softmax'd over `D = len(active_datasets)`
instead of over attack-class experts. The gate always produces probabilities
over all experts, but `model.gate.routing` chooses how they are consumed:
`dense` blends all experts (default), while `top1` executes only the argmax
expert for each row. The gate is not trained until Stage C.

Gate training is config-selectable. The primary method is task-loss-driven;
the explicit `moe_dataset_damex` comparison instead makes dataset-ID CE the
semantic router target and prevents downstream task gradients from updating
the gate. Neither method supplies dataset identity as an inference input, and
inference routing is independently selectable as dense or top-1. See §7 and §8.

## 6. Combination rule — how expert + gate outputs become a prediction

`models/moe.py::MoEDatasetNIDS.forward`

```python
expert_probs   = softmax(expert_logits, dim=-1)              # (B, D, C)
combined_probs = einsum('bd,bdc->bc', gate_weights, expert_probs)  # (B, C)
prediction     = argmax(combined_probs, dim=-1)
```

A straightforward soft mixture-of-experts over full per-dataset class
distributions — **simpler than `moe_nids`'s combination rule**, since every
expert already predicts the full class vocabulary directly; there is no
per-expert relabeling to reconcile (`moe_nids` has to separately pool
"benign votes" across experts and discard undefined "OTHER" mass because
each of its experts only answers a 3-way sub-question). `combined_probs`
here is already a proper categorical distribution (rows sum to 1), so
`combined_probs_to_log_probs` is just a numerically-safe `log`, kept as a
named method purely for API symmetry with `moe_nids`'s
`combined_scores_to_log_probs`.

With `routing: dense`, dataset-ambiguous traffic receives the original blended
prediction. With `routing: top1`, rows are grouped by gate argmax and only the
selected expert runs. Offline evaluation may explicitly execute all experts
to build diagnostic cross-dataset matrices, but those extra calls never
participate in the deployed prediction.

Under `top1 + assigned_only`, a predicted route is allowed to update an expert
only when it matches the row's training-time dataset owner. A misroute updates
no expert. This preserves strict dataset ownership while direct DAMEX CE trains
the gate; at inference, dataset ID remains unavailable.

`MoEDatasetNIDS.forward(x)` takes exactly one argument — `_assert_dataset_
blind_signature` (`inference/predict.py`) checks this via
`inspect.signature` at model-load time, mirroring `moe_nids`; see
`tests/test_dataset_blind_inference.py`.

## 7. Three-stage training procedure

Same cold-start rationale as `moe_nids` (see its `docs/ARCHITECTURE.md`
§7): a curriculum specifically designed so the gate isn't forced to choose
between one competent expert and `D-1` randomly-initialized ones the moment
it's introduced.

### Stage A — Encoder pretraining (`training/stage_a_pretrain.py`)

`SharedEncoder` + a temporary `ProbeHead`, trained on all pooled training
data (every class, every dataset) with plain `CE(probe(z), y_class)`. **No
dataset-related loss term at this stage** — unlike `moe_nids`, this project
does not use a cross-dataset latent-alignment loss at all (see §11, "what's
deliberately different"); Stage A here is pure classification pretraining.
Only `encoder.state_dict()` is persisted; the probe is discarded.

### Stage B — Independent dataset-expert warm-start (`training/stage_b_warmstart.py`)

Loads the Stage A encoder and freezes it. Trains each dataset-expert (or,
for `moe_dataset_adapters`, each adapter + the shared head — see
`training/model_utils.py::expert_train_params`/`expert_forward_one`)
**independently and sequentially**, one dataset at a time, on a
class-balanced view of ONLY that dataset's own rows, with plain
`CE(expert(z), y_class)` — no relabeling, since each dataset-expert already
predicts the full class vocabulary. This stage is deliberately "D separate
models trained independently" in disguise, same as `moe_nids`'s Stage B —
it exists purely to give every dataset-expert (including experts for
smaller datasets) a reasonable independent starting point before the gate
is introduced. For `moe_dataset_adapters`, note that the shared head is
touched by every dataset's warm-start in sequence, so later datasets can
nudge it away from what earlier ones learned — expected, documented
behavior for that ablation, not a bug (see `model_utils.py` docstring).
Checkpointing is resumable per-dataset, per-epoch.

### Stage C — Joint fine-tune (`training/stage_c_jointfinetune.py`)

Loads Stage A encoder weights + Stage B expert-bank weights, **instantiates
the Gate for the first time** (fresh init), and assembles the full
`MoEDatasetNIDS`. Encoder trainability is configurable via
`training.stage_c_unfreeze` (`"none"` / `"all"` / `"last_layer"`, default
`"last_layer"`, same semantics as `moe_nids`). Loss:

```
loss = CE(combined_probs, y_class)
     + lambda_balance     * load_balance_penalty(gate_weights)
     + lambda_dataset_aux * CE(gate_weights, dataset_id)      [optional, see §8]
```

`y_class` is the ground-truth class index into `[Benign, *active_classes]`
— unlike `moe_nids`, there's no separate "combined space" vs. "full
vocabulary" distinction to track, since every dataset-expert already
predicts the fixed, full canonical task vocabulary directly (this project's
`data.active_classes` IS the fixed output space, not a data-derived
vocabulary — see §9.3). Batches are drawn via `ClassBalancedBatchSampler`
over the **task class label**, not dataset ID — guaranteeing
`min_per_class_per_batch` samples of every canonical class, including rare
ones, in every batch. `load_balance_penalty` is ported unchanged from
`moe_nids/models/losses.py` (coefficient-of-variation-squared over
per-expert mean gate utilization).

### Resumability

All three stages checkpoint per-epoch (`training.checkpoint_every_n_epochs`)
to `training.checkpoint_dir`. Re-running `training.run` picks up mid-stage
automatically; `--set training.force_restart=true` discards existing
checkpoints. `training/checkpoint.py` additionally tags Stage B/C
checkpoints with `bank_kind` (`"full"` or `"adapter"`), so a checkpoint
always self-describes which expert-bank class to reconstruct it into.

## 8. Selectable gate-supervision methods

`models/losses.py::dataset_aux_loss` + `training/stage_c_jointfinetune.py::_lambda_dataset_aux`

This is the ONLY function in the entire codebase allowed to compute a loss
between `gate_weights` and ground-truth `dataset_id`. Its weight is
selected by `training.stage_c.gate_supervision`:

| `gate_supervision` | weight | role |
|---|---|---|
| `"none"` | `0.0` | Pure task-loss-driven gate. `dataset_aux_loss` is **never called** — not called-with-zero-weight, structurally skipped (`if lambda_dataset_aux > 0.0:` gate in `run_stage_c`). Verified by `tests/test_no_dataset_id_supervision.py` via a monkeypatched `dataset_aux_loss` that raises if invoked. |
| `"light_aux"` (**default**) | `training.stage_c.lambda_dataset_aux`, default `0.1` | Low-weight regularizer — same role/magnitude as `moe_nids`'s Stage C `lambda_align=0.1`: a soft nudge for training stability, never the dominant signal. |
| `"hard"` | `training.stage_c.lambda_dataset_aux_hard`, default `5.0` | **Ablation/baseline only, never the recommended default.** Makes the gate loss dominant enough that the gate approximates a real dataset classifier — selecting `architecture=moe_dataset_hard_gate` applies this preset automatically. This run is expected to converge empirically close to the `hard_two_stage` baseline (§10); that convergence is itself a sanity check on the whole setup — if `moe_dataset_hard_gate` DIDN'T end up close to `hard_two_stage` under a dominant dataset-ID loss, something else in the pipeline would be suspect. |
| `"damex"` | `training.stage_c.lambda_dataset_aux_damex`, default `1.0` | Direct DAMEX-style router supervision. The gate weights are detached in the downstream task mixture, so only dataset-ID CE and load balancing update the gate. Selecting `architecture=moe_dataset_damex` also defaults `expert_update_policy` to `assigned_only`, ensuring each expert receives parameter gradients only from its owned dataset. |

The `hard` mode differs deliberately from `damex`: `hard` retains the task
gradient into the gate and merely gives dataset CE a large coefficient.
`damex` structurally removes that task gradient. Architecture presets are
applied by `training.config.apply_architecture_defaults`; explicit `--set`
values still win, making focused ablations possible without code edits.

## 9. Data pipeline

### 9.1 Dataset registry, harmonization, loaders — reused unchanged

`data/registry.py`, `data/harmonization.py`, `data/loaders.py` are ported
near-identically from `moe_nids` (the alias maps, per-dataset scaler
fitting, presence masking, `DatasetIdentityLeakageError` leakage guard, and
`TrainSplit`-only-fit type guard are all unchanged) — see
`moe_nids/docs/ARCHITECTURE.md` §9.1–9.2 for the full design rationale.
Feature harmonization is explicitly **not** re-derived or redesigned here,
per the project brief.

### 9.2 Label harmonization

`config/default.yaml::data.label_mapping` — same per-dataset
raw-label-to-canonical-class taxonomy as `moe_nids` (18 canonical classes),
same `strict_label_mapping` fail-loud philosophy.

### 9.3 Class vocabulary — a key structural difference from `moe_nids`

`training/dataset.py::prepare_datasets`

`moe_nids`'s `class_vocab` is *data-derived* (the sorted set of canonical
labels actually observed in the pooled training data), separate from its
*combined_class_names* (`[Benign, *active_classes]`, the space experts
individually cover). This project collapses that distinction: `class_names
= [Benign, *active_classes]` is the **fixed, config-derived** output space
every dataset-expert predicts over directly — there's no relabeling step
downstream that would need a separately-tracked "full vocabulary" space.
`prepare_datasets` raises loudly (fail-loud, matching the
`strict_label_mapping` philosophy) if any row's `canonical_label` maps to
something outside `[Benign, *active_classes]` — a config-mapping bug should
surface immediately, not silently produce out-of-vocabulary training
targets.

`dataset_idx` (index into `active_datasets`) is new relative to `moe_nids`:
tracked in `PreparedSplit` purely as a training-time TARGET for Stage B's
per-dataset routing and the dataset-aux regularizer (§8) — never
concatenated into the feature tensor, never passed to
`MoEDatasetNIDS.forward` (see `training/dataset.py::HarmonizedTensorDataset`
docstring).

### 9.4 Config-driven dataset selection

`training/dataset.py::assert_active_datasets_consistent` is a fail-loud
startup guard: every dataset referenced in `data.label_mapping` must be a
subset of `data.active_datasets`, raising a clear error listing any stale
entry left over after narrowing the active set. Combined with
`DatasetExpertBank`/`AdapterExpertBank` sizing themselves from
`len(dataset_names)` at construction time, and `Gate`'s output dimension
likewise, reducing `active_datasets` to 2–3 datasets for fast iteration
requires editing only `config/default.yaml` — verified end-to-end by
`tests/test_active_dataset_toggle.py`.

## 10. Baselines

`models/baselines.py` — selected via the same `architecture` config flag,
sharing the harmonization pipeline, data loading, and evaluation code.

- **`PlainPooledSoftmax`** — ported from `moe_nids`: same shared encoder,
  single joint `C`-way softmax head, no dataset structure at all.
- **`MatchedDenseClassifier`** — the capacity/compute control. It keeps the
  shared encoder and replaces the expert bank plus gate with one pyramidal,
  two-hidden-layer ReLU/dropout head. A deterministic integer solver matches
  either all stored expert+gate parameters, top-1 active expert+gate
  parameters, or top-1 Linear-layer MACs within 0.5%. The exact dimensions,
  target, achieved value, and residual are checkpointed and reported.
- **`NoFusionModel`** — ported from `moe_nids`: fully separate encoder+head
  per dataset, `forward(x, dataset_name)` requires ground-truth dataset ID
  at both train and inference.
- **`HardTwoStageModel`** — **new to this project**, the literal "obvious
  two-step approach" it needs to demonstrably beat. Stage (a): a standalone
  dataset-ID classifier (`SharedEncoder` architecture + a plain
  `Linear` head, `models/losses.py::DatasetIDClassifierHead`), trained with
  ordinary `CE` against ground-truth `dataset_id` — the one place in this
  entire codebase a model is *deliberately and primarily* supervised on
  dataset identity, because that's exactly what this baseline is for.
  Stage (b): independent per-dataset classifiers, structurally identical to
  `NoFusionModel`. At inference, `forward(x)` predicts dataset ID via stage
  (a) (`argmax`, no blending) and hard-routes each sample to its predicted
  dataset's stage-(b) sub-model — critically, this uses the model's OWN
  predicted dataset ID, not a caller-supplied ground-truth one, so
  `HardTwoStageModel.forward` is still dataset-blind in the same sense as
  `MoEDatasetNIDS.forward` and can be evaluated apples-to-apples on
  `evaluation/ood_ambiguity_eval.py`.

### Fairness curriculum and accounting

`training.baseline.encoder_init` explicitly selects cold initialization or
the exact Stage-A encoder artifact. A Stage-A artifact is accepted only when
its seed, encoder configuration, classes, active datasets, feature schema, and
split signature match. `training.baseline.stage_b_warmstart=matched_exposure`
freezes that encoder and feeds a dense head the same Stage-B batch schedule,
examples, and optimizer-step count as the reference experts before the shared
Stage-C schedule.

Headline runs use `training.selection_mode=fixed_epochs`; `best_val` remains
available for sensitivity analysis. `evaluation/resource_accounting.py`
reports total/trainable/active parameters, component breakdowns, per-sample
Linear MACs, `2 * MACs` FLOPs, stage steps/examples/epochs/wall time, and the
Stage-A hash. For soft dense routing all experts are active; for top-1 routing
the active path is encoder + gate + one expert; hard two-stage includes its
router encoder and the selected classifier encoder sequentially. No-fusion is
marked as an oracle because evaluation supplies the true dataset identity.

`training/fairness_matrix.py` executes the complete paired seed set (0, 1, 2)
with a shared immutable Stage-A checkpoint per seed. Aggregation rejects
missing seeds or mixed split signatures and writes descriptive paired
Student-t intervals to `Seed_Summary.csv`.

`moe_dataset_hard_gate` is expected to converge empirically close to
`hard_two_stage` (§8) — demonstrating that convergence is a documented
sanity check on the primary architecture's implementation, not a separate
claim to verify by other means.

## 11. Evaluation

`evaluation/`

- **Primary metric: per-dataset macro-F1 AND per-dataset per-class recall**
  (`evaluation/metrics.py::evaluate_per_dataset`), not just pooled
  aggregate macro-F1 — the entire point of this project is measuring which
  specific datasets get rescued or hurt relative to `plain_pooled`/
  `no_fusion`, so per-dataset breakdown is the headline result, not a
  secondary diagnostic. Weighted-F1 remains reference-logging-only, never
  for model selection, same convention as `moe_nids`.
- **`evaluation/ood_ambiguity_eval.py`** — the direct test of the
  soft-mixing hypothesis. Two legs: (1) a cross-dataset near-neighbor split
  within the pooled test set (same canonical class, nearest neighbor in
  harmonized feature space lives in a *different* dataset — via
  `sklearn.neighbors.NearestNeighbors`), always available; (2) a genuinely
  held-out dataset (`evaluation.ood_holdout_dataset`, not in
  `active_datasets` at all, never seen by any expert or the gate), which
  fits a scratch harmonizer purely for feature scaling (never touching
  model training) so its rows land in the same harmonized space. Compares
  `hard_two_stage` vs. the primary soft-gated MoE on both legs — expect
  `hard_two_stage` to degrade sharply if the soft-mixing hypothesis holds.
- **`evaluation/gate_analysis.py`** — gate weight distribution per TRUE
  source dataset (does the gate's learned partition line up with, diverge
  from, or refine the ground-truth dataset boundary?), per-expert
  utilization, and a gate-collapse check (`detect_gate_collapse`, flags any
  expert whose mean utilization across the eval set falls below a
  threshold).
- **`evaluation/report.py`** — `Trial_ID`-keyed tracker CSVs
  (`Trials.csv`/`Overall_Metrics.csv`/`Per_Class_Metrics.csv`, plus
  `Per_Dataset_Metrics.csv`, new here) and both `comparison_table`
  (per-class) and `per_dataset_comparison_table` (per-dataset) helpers,
  each pivoting the configured architecture variants into one ablation table — see
  `notebooks/08_full_ablation_report.ipynb`.

## 12. Config-driven design

### Basic MoE comparator

`architecture=moe_basic` keeps the same shared encoder, gate, number of
experts, per-expert MLP, class vocabulary, optimizer, routing implementation,
and evaluation path as the full dataset-MoE. The expert count equals the
number of active datasets solely to match parameter capacity; experts are
named `expert_0`, `expert_1`, and so on and have no dataset ownership.

Its preset changes only the dataset-specific parts of training:

- `training.stage_b.warmstart_mode=random_init` replaces per-dataset expert
  warm-start with a deterministic random expert-bank checkpoint.
- `training.stage_c.gate_supervision=none` removes dataset-ID router loss.
- `training.stage_c.expert_update_policy=all` lets pooled task gradients update
  every softly weighted expert.
- `training.stage_c.lambda_expert_anchor=0.0` removes the Stage-B specialization
  anchor.

The preset is additive. Switch between the original and comparator using
`architecture=moe_dataset_soft` and `architecture=moe_basic`; explicit dotted
overrides remain available for controlled ablations.

Every structural knob — active datasets, active classes, the cross-dataset
label taxonomy, architecture variant (including which expert-bank class and
gate-supervision weight it implies), which training stages to run, all
hyperparameters — is driven by `config/default.yaml`, loaded via
`training/config.py` with dotted `--set key.path=value` CLI overrides,
identical mechanism to `moe_nids`.

## 13. Summary of what's different from `moe_nids`

| | `moe_nids` | `dataset_moe_nids` |
|---|---|---|
| Expert granularity | One expert per canonical **attack class** | One expert per **dataset** |
| Expert output space | 3-way (target/benign/other), via deterministic relabeling | Full task vocabulary (`Benign` + every active class) directly, no relabeling |
| Gate output dimension | `num_classes` (experts) | `num_datasets` (experts) |
| Gate ground-truth supervision | N/A (no per-expert identity to supervise against) | Configurable: task-primary (`none` / `light_aux` / `hard`) or DAMEX-style direct dataset supervision (`damex`) |
| Combination rule | Vote-pooling with discarded "OTHER" mass, per-row renormalization for CE | Straightforward soft mixture over full per-dataset distributions, already a proper distribution |
| Cross-dataset representation sharing mechanism | Explicit latent-space alignment loss (centroid/momentum-centroid/contrastive) | None — whatever the shared encoder + soft gate/expert combination learns on its own in Stage C |
| Target failure mode | Rare-class starvation / pooling | Negative transfer across datasets |
| New required baseline | — | `hard_two_stage` (standalone dataset classifier + hard-routed per-dataset models) |
| Lightweight ablation | — | `AdapterExpertBank` (shared head + per-dataset FiLM correction) |
| Data pipeline | Origin (registry/harmonizer/loaders/label taxonomy) | Ported near-identically, unchanged in spirit |

Explicitly out of scope for this build, per the project brief: rare-class
pooling (that remains `moe_nids`'s problem), any cross-dataset
latent-alignment loss, and TabDDPM-style synthetic augmentation (a
downstream extension point, not part of the current build, exactly as in
`moe_nids`).
