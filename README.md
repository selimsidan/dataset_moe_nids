# dataset_moe_nids

Sibling project to [`moe_nids`](https://github.com/selimsidan/moe_nids): same
research program (a single NIDS classifier trained across heterogeneous
public NIDS datasets, dataset identity unavailable at inference), different
architecture. `moe_nids` gives every canonical **attack class** its own
expert to lift rare-class recall; this project gives every **dataset** its
own expert, to mitigate negative transfer -- recovering per-dataset
performance a single pooled model loses relative to a model that could
specialize per dataset, while still sharing whatever representation
genuinely transfers across datasets.

## Hard constraints (enforced structurally, not by convention)

- **Gate supervision is an explicit experimental choice.** The primary
  `moe_dataset_soft` method remains task-loss-driven, with dataset identity
  absent or used only as a light auxiliary regularizer. The opt-in
  `moe_dataset_damex` method reverses that contract: dataset-ID CE and load
  balancing are the only router gradients, and dataset ownership is enforced
  for expert-parameter updates. The older task-plus-dominant-auxiliary
  `moe_dataset_hard_gate` ablation remains available. See
  `tests/test_no_dataset_id_supervision.py`, which verifies this
  structurally (the loop never even calls `dataset_aux_loss` when
  `gate_supervision: none`), not just that its weight happens to be zero.
- **Routing is an explicit experimental choice.** `model.gate.routing: dense`
  remains the default and blends every expert. The opt-in `top1` path performs
  real grouped dispatch: only the argmax expert executes for each row. The gate
  still receives only features, never ground-truth dataset identity, at
  inference.
- **No dataset identity at inference.** `MoEDatasetNIDS.forward(x)` takes
  exactly one argument; `inference/predict.py` asserts this via
  `inspect.signature` before serving predictions. See
  `tests/test_dataset_blind_inference.py`.
- **Sparse dispatch is structural when selected.** The dense bank path still
  runs every expert. `forward_selected` groups rows by argmax expert and never
  invokes an unselected expert. With DAMEX plus `assigned_only`, misrouted rows
  update no expert, so predicted routing cannot contaminate another dataset's
  expert.
- **Config-driven dataset selection, zero hardcoded dataset counts.** Every
  module (registry, harmonizer, expert-bank sizing, training, evaluation)
  derives its dataset list from `data.active_datasets` alone. See
  `tests/test_active_dataset_toggle.py`.
- **Data pipeline reused, not redesigned.** `data/registry.py`,
  `data/harmonization.py`, `data/loaders.py` are ported near-identically
  from `moe_nids` -- feature harmonization, per-dataset scaler fitting,
  presence masking, and the identity-column leakage guard
  (`DatasetIdentityLeakageError`) are unchanged.

## Repository layout

```
config/default.yaml         single YAML config driving a full run
data/
  registry.py                DATASET_REGISTRY (ported from moe_nids)
  paths.py                   Drive/local path resolution
  harmonization.py           Harmonizer (ported from moe_nids)
  loaders.py                 chunked CSV/folder loaders, label harmonization, splitting (ported)
models/
  encoder.py                 SharedEncoder (+ Stage-A-only ProbeHead) -- ported
  dataset_experts.py         Expert / DatasetExpertBank -- one expert PER DATASET, full task vocabulary
  private_encoder_experts.py full private encoder+head per dataset ablation
  adapters.py                AdapterExpertBank -- FiLM-style lightweight ablation alternative
  gate.py                    Gate (softmax over dataset-experts)
  moe.py                     MoEDatasetNIDS -- soft mixture-of-experts combination rule
  losses.py                  load_balance_penalty + direct/auxiliary dataset-ID CE
  baselines.py                plain, matched-dense, no-fusion, and hard-two-stage models
training/
  config.py                   YAML loader + --set dotted overrides (ported)
  dataset.py                  prepare_datasets(): raw -> split -> harmonized tensors + dataset_idx
  sampler.py                  ClassBalancedBatchSampler (ported)
  checkpoint.py                per-stage + per-epoch resumable checkpoint I/O
  model_utils.py               architecture -> concrete model wiring (one place)
  stage_a_pretrain.py          Stage A
  stage_b_warmstart.py         Stage B (independent per-dataset expert warm-start)
  stage_c_jointfinetune.py     Stage C (gate + joint fine-tune)
  baseline_train.py            fairness-controlled in-memory baseline trainers
  dense_ooc.py                 resumable plain/matched-dense full-data training
  no_fusion_ooc.py             resumable oracle no-fusion full-data training
  hard_two_stage_ooc.py        resumable disk-backed phases A/B for the full-data baseline
  fairness_matrix.py           paired seeds 0/1/2 experiment launcher + aggregation
  run.py                       CLI entry point (architecture + stage selection via config)
inference/predict.py         single shared-encoder inference path, dataset-blind by construction
evaluation/
  metrics.py                  macro-F1 + per-class AND per-dataset breakdowns
  bootstrap_ci.py              bootstrap CIs for low-sample-count classes (ported)
  ood_ambiguity_eval.py        hard_two_stage vs soft-gated MoE on dataset-ambiguous/OOD traffic
  gate_analysis.py             gate weight distribution, expert utilization, collapse check
  report.py                    comparison tables + Trial_ID-keyed tracker CSVs incl. Per_Dataset_Metrics.csv
  resource_accounting.py       total/active parameters, MACs/FLOPs, and training budgets
  seed_summary.py              paired three-seed mean/SD/difference summaries
notebooks/                   Colab-runnable workflows, one config cell each
tests/                       dataset-blind / no-dataset-id-supervision / active-dataset-toggle tests
```

## One-notebook Google Colab run

[`notebooks/00_colab_end_to_end.ipynb`](notebooks/00_colab_end_to_end.ipynb)
is the canonical start-to-finish Colab workflow. It is currently finalized
for the four-way dense `moe_dataset_soft` active-budget comparison against
hard two-stage routing. Its `64 -> 45 -> classes` experts match the hard
model's active parameters and Linear MACs within 0.5% for the finalized
22-class combination; its configuration cell exposes execution mode, seed,
run name, and resume behavior while documenting the locked model choices.
It securely clones this private repository with a token read from
Colab Secrets, mounts the shared Drive datasets, runs the tests and full
training pipeline, evaluates the test split, and displays the tracker CSVs
persisted to Drive.

[`notebooks/10_colab_end_to_end_damex.ipynb`](notebooks/10_colab_end_to_end_damex.ipynb)
is the dedicated start-to-finish DAMEX workflow. It locks Stage C to direct
dataset-ID gate supervision and `assigned_only` expert updates, verifies the
resolved contract before training, and supports both a small in-memory smoke
run and the full disk-backed NF-v3 run.

[`notebooks/11_colab_end_to_end_damex_top1.ipynb`](notebooks/11_colab_end_to_end_damex_top1.ipynb)
adds true top-1 sparse expert execution to the DAMEX contract. Dense routing
remains the default and can be restored with `model.gate.routing=dense`.

[`notebooks/12_colab_end_to_end_basic_moe.ipynb`](notebooks/12_colab_end_to_end_basic_moe.ipynb)
is the capacity-matched native MoE comparison: generic experts, no
dataset-specific warm-start or ownership, and no dataset-ID router loss.

[`notebooks/13_colab_end_to_end_hard_two_stage.ipynb`](notebooks/13_colab_end_to_end_hard_two_stage.ipynb)
is the matching standalone four-dataset hard two-stage run. It locks the
same 47-feature preprocessing and encoder MLP dimensions, trains only the
two independent phases, reports stage-A dataset-routing accuracy, and joins
its per-origin macro-F1 results with the primary run when those results are
available.

[`notebooks/15_router_bottleneck_diagnostics.ipynb`](notebooks/15_router_bottleneck_diagnostics.ipynb)
loads notebook 00's completed capacity-matched MoE, trains or reuses a hard
baseline initialized from the exact same Stage-A checkpoint, and produces the
deployable, learned-top-1, oracle-route, confidence-bin, route-conditioned,
per-dataset/per-class, expert-cross-dataset, and resource-accounting diagnostics
in one top-to-bottom run.

[`notebooks/16_colab_end_to_end_private_encoder_moe.ipynb`](notebooks/16_colab_end_to_end_private_encoder_moe.ipynb)
is the paired follow-up to that diagnostic. It reuses notebook 00's exact
Stage-A checkpoint, clones it into a dedicated gate encoder plus one full
private encoder per expert, trains/evaluates the dense task-driven soft MoE,
and reports private-minus-shared metric deltas alongside the extra parameter
and MAC cost.

[`notebooks/17_colab_discriminative_latent_private_encoder_moe.ipynb`](notebooks/17_colab_discriminative_latent_private_encoder_moe.ipynb)
keeps notebook 16's private-encoder topology but trains a new Stage-A encoder
with a configurable discriminative objective (`supcon`, `balanced_supcon`,
`center`, or `arcface`). It reports original-space geometry, frozen probes,
and reproducible PCA/UMAP views for Stage A, every private Stage-B encoder,
and the Stage-C gate/expert encoders. The default `ce`/`legacy` training
configuration remains backwards-compatible with earlier runs.

In Colab, add a secret named `GITHUB_TOKEN` (fine-grained token with read-only
Contents access to this repository) and grant the notebook access. Never put
the token directly in a notebook cell or clone URL. The numbered notebooks
remain useful for focused diagnostics and individual-stage experimentation.

[`notebooks/09_colab_reduced_warmstart_soft_moe.ipynb`](notebooks/09_colab_reduced_warmstart_soft_moe.ipynb)
is the faster full-data four-way alternative. It retains mandatory
dataset-specific Stage B expert warm-starting, but reduces the schedule to
Stage A = 3, Stage B = 2, and Stage C = at most 30 epochs. Its subprocess is
unbuffered and reports row progress and elapsed time during long epochs.
Stage C preserves expert ownership: all four outputs remain in the soft
mixture, but each row can update only its dataset-assigned expert. The shared
encoder and dataset-blind gate still learn from the pooled task objective, and
a small optional Stage-B parameter anchor discourages expert drift.

### Full-data 2-way / 3-way / 4-way NF-v3 MoE runs

The notebook's production `out_of_core_full` mode supports any selection of
two to four schema-compatible NF-v3 datasets:

- `NF-UNSW-NB15-v3`
- `NF-ToN-IoT-v3`
- `NF-BoT-IoT-v3`
- `NF-CICIDS2018-v3`

It uses the confirmed 47 behavior features shared by these releases while
excluding IP addresses, ports, and absolute capture timestamps. Every mapped
row is assigned exactly once to a deterministic, class-stratified signed split
artifact. Split arrays are globally shuffled on disk, pooled through logical
views rather than copied, standardized with pooled training statistics only,
and read in bounded batches during Stage A/B/C and evaluation.

Changing `ACTIVE_DATASETS` in the notebook is sufficient to choose a 2-way,
3-way, or 4-way combination. Change `RUN_NAME` whenever the combination,
architecture, capacity, or training settings change. A signed run contract
prevents incompatible checkpoints from being reused even if the name is
accidentally left unchanged.

The default configuration's full experts are MLPs (`64 -> 128 -> 64 -> classes`), so the
plain linear pooled head is intentionally a minimal ablation rather than a
capacity-matched baseline. `matched_dense` now constructs the corresponding
total-parameter, top-1-active-parameter, and top-1-MAC controls dynamically for
the actual dataset/class combination. The primary gate's supervision remains
configurable and is recorded with every trial. Notebook 00 deliberately
overrides the expert width to `64 -> 45 -> classes` for its hard-active-budget
control; it does not change the repository-wide default architecture.

Detailed full-data reports include combined and per-origin overall metrics,
per-class confusion metrics and one-vs-rest ROC-AUC, the full confusion matrix, mean gate
weights by true dataset origin, expert utilization, and reusable chunked
prediction files. In-memory runs report exact per-class plus macro, weighted,
and micro ROC-AUC. Full-data runs preserve bounded memory with a configurable
4096-bin streaming approximation (`evaluation.roc_auc_bins`) and label the
method in each CSV. Prepared splits are reusable across different MoE
architectures when their data contracts match.

The heterogeneous CICFlowMeter/UNSW/CICIoT datasets remain available to the
legacy in-memory/smoke pipeline. They are deliberately rejected by the current
full-data path until official/group-aware split policies and streaming schema
harmonization are implemented for those multi-file formats.

## Running a full experiment

```bash
pip install -r requirements.txt

# Primary architecture (Stage A -> B -> C, then evaluation + tracker CSVs):
python -m training.run --config config/default.yaml --set architecture=moe_dataset_soft

# Ablations, through the same entry point:
python -m training.run --config config/default.yaml --set architecture=moe_basic
python -m training.run --config config/default.yaml --set architecture=moe_dataset_hard_gate
python -m training.run --config config/default.yaml --set architecture=moe_dataset_damex
python -m training.run --config config/default.yaml --set architecture=moe_dataset_adapters
python -m training.run --config config/default.yaml --set architecture=moe_dataset_private_encoders

# Baselines:
python -m training.run --config config/default.yaml --set architecture=plain_pooled
python -m training.run --config config/default.yaml --set architecture=matched_dense \
    --set model.dense_match.axis=total_params
python -m training.run --config config/default.yaml --set architecture=no_fusion
python -m training.run --config config/default.yaml --set architecture=hard_two_stage

# Print the complete three-seed fairness matrix; add --execute to run it:
python -m training.fairness_matrix --config config/default.yaml \
    --prefix fair_comparison_v1 --execution-mode out_of_core

# Full four-dataset disk-backed hard two-stage run (phases A and B only):
python -m training.ooc_run --config config/default.yaml \
    --set architecture=hard_two_stage --set training.stages=[A,B]

# Fast iteration on a 2-dataset subset -- zero code changes:
python -m training.run --config config/default.yaml \
    --set data.active_datasets=[NF-UNSW-NB15-v3,NF-BoT-IoT-v3]

# Ablation: re-run only Stage C from an existing Stage A+B checkpoint
python -m training.run --config config/default.yaml --set training.stages=[C]
```

Results land in `evaluation.output_dir` as `Trials.csv` / `Overall_Metrics.csv`
/ `Per_Class_Metrics.csv` / `Per_Dataset_Metrics.csv` /
`Resource_Accounting.csv`, keyed by `Trial_ID`.
Overall and per-dataset outputs include accuracy, balanced accuracy, complete
macro/micro/weighted precision-recall-F1 groups, and macro/weighted/micro
one-vs-rest ROC-AUC;
the per-class output includes one-vs-rest ROC-AUC for every evaluable class.
Use `evaluation.report.comparison_table` / `per_dataset_comparison_table` to
pivot several `EvaluationResult`s (one per architecture variant) into a
single ablation table -- see `notebooks/08_full_ablation_report.ipynb`.

## Architecture variants (`architecture` config flag)

| Value | What it is |
|---|---|
| `moe_dataset_soft` | **Primary.** One full expert per dataset, soft gate, `gate_supervision: light_aux`. |
| `moe_basic` | Capacity-matched native MoE comparator. Generic unassigned experts learn jointly from pooled task gradients; the gate uses task loss plus load balancing only. No dataset-ID loss, ownership mask, or dataset-specific warm-start. |
| `moe_dataset_hard_gate` | Ablation: same architecture, `gate_supervision: hard` -- shows what's lost by undoing the "don't supervise the gate on dataset ID" decision. Should converge close to `hard_two_stage`. |
| `moe_dataset_damex` | DAMEX-style strict option: the gate learns only from direct dataset-ID CE plus load balancing (no downstream task gradient), and expert updates default to `assigned_only`. Inference remains dataset-blind; routing can be dense or top-1. |
| `moe_dataset_adapters` | Ablation: `AdapterExpertBank` (shared head + per-dataset low-rank FiLM correction) instead of full independent expert heads -- less overfitting risk on small datasets. |
| `plain_pooled` | Baseline: single shared encoder + one joint classification head, no dataset structure. |
| `matched_dense` | Two-hidden-layer dense control. `model.dense_match.axis` selects total parameters, top-1 active parameters, or top-1 forward MACs; resolved widths and residual are reported. |
| `no_fusion` | Oracle reference: fully separate encoder+head per dataset, `forward(x, dataset_name)` requires ground-truth dataset ID at inference and is therefore excluded from the dataset-blind headline ranking. |
| `hard_two_stage` | Baseline (new to this project): standalone dataset classifier (stage a) hard-routes each sample to an independent per-dataset classifier (stage b). The literal "obvious two-step approach" this project needs to beat -- `forward(x)` is still dataset-blind (routes on its OWN predicted dataset ID), so it's comparable apples-to-apples on the OOD/ambiguity eval. |

Routing is independently controlled by `model.gate.routing`: `dense`
(default, backwards-compatible) or `top1` (one executed expert per row).
For the combined experiment use:

```bash
python -m training.run --config config/default.yaml \
  --set architecture=moe_dataset_damex \
  --set model.gate.routing=top1
```

## Fair comparison protocol

The paper-facing protocol separates initialization, capacity, and deployed
compute instead of treating them as one comparison. For every seed, one
immutable Stage-A checkpoint is shared and validated against the encoder
shape, class vocabulary, feature schema, split signature, active datasets,
and seed. Baselines select `training.baseline.encoder_init=stage_a|random` and
`training.baseline.stage_b_warmstart=none|matched_exposure`. Matched exposure
uses the same Stage-B per-dataset batch sequence, example count, and optimizer
step budget as expert warm-starting.

`training.selection_mode=fixed_epochs` is the headline setting;
`best_val` is a secondary sensitivity analysis. Parameter matching is
downstream of the common encoder. The deterministic solver retains a
pyramidal two-hidden-layer MLP, fails above 0.5% residual, and never pads the
model with unused parameters. Active-parameter matching is not described as
iso-FLOPs unless its separately reported forward-FLOP residual also qualifies.

Run seeds 0, 1, and 2 through `training.fairness_matrix`. The resulting
`Seed_Summary.csv` contains mean, sample standard deviation, and paired
method-minus-reference Student-t intervals. With only three seeds, those
intervals are descriptive and are not significance claims. Historical runs
without the Stage-A metadata contract must not be merged into this table.

## Three-stage training curriculum

Same cold-start rationale as `moe_nids` (see its `docs/ARCHITECTURE.md` §7):
training a randomly-initialized gate jointly with randomly-initialized
experts from scratch risks the gate locking onto whichever expert looks
best early and starving the rest of gradient.

1. **Stage A** -- `SharedEncoder` + a temporary multiclass `ProbeHead`,
   trained on all pooled data, plain CE. No dataset-related loss term. Probe
   discarded; only the encoder persists.
2. **Stage B** -- encoder frozen; each dataset-expert (or adapter+shared-head,
   for `moe_dataset_adapters`) trained independently on only its own
   dataset's rows, full-vocabulary CE (no relabeling -- unlike `moe_nids`'
   class-experts, each dataset-expert predicts the whole canonical task
   space directly).
3. **Stage C** -- gate instantiated (fresh init), encoder unfrozen per
   `training.stage_c_unfreeze`, trained end-to-end with
   `CE(combined_probs, y) + lambda_balance * load_balance_penalty(gate_weights)
   + lambda_dataset_aux * CE(gate_weights, dataset_id)`. Batches drawn via
   `ClassBalancedBatchSampler` over the TASK class label, not dataset ID. In
   `gate_supervision: damex`, the gate weights used by the task term are
   detached, so only the latter two terms update the router.

## Data

Ported unchanged from `moe_nids`: `data/paths.py` resolves dataset locations
the same way (`NIDS_DRIVE_BASE`/`NIDS_OUTPUT_DIR` env vars, Colab Drive, or a
local Google Drive mirror), so both projects can share one `NIDS_datasets/`
folder without duplicating data. Output artifacts land under a separate
`dataset_moe_nids_runs` sub-folder so a run of this project never clobbers
`moe_nids` results. See `moe_nids/docs/ARCHITECTURE.md` §9 for the full
harmonization/label-mapping design rationale -- unchanged here.

## Tests

```bash
pip install -r requirements.txt
pytest
```

Tests use synthetic in-memory fixtures -- no real dataset files required.

## Out of scope for this build

Per the project brief: this project does not attempt rare-class pooling
(that's `moe_nids`'s job) -- its target failure mode is negative transfer
across datasets, not class imbalance. No latent-space cross-dataset
alignment loss (unlike `moe_nids`'s Stage A/C alignment loss) -- the
representation sharing this project studies is whatever the shared encoder
+ soft gate/expert combination learns on its own, not an explicit alignment
penalty. TabDDPM-style synthetic augmentation remains a downstream extension
point, not part of this build.
