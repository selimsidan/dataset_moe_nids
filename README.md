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

- **The gate is never PRIMARILY supervised on ground-truth dataset ID.**
  `training/losses.py::dataset_aux_loss` is the only place dataset identity
  can influence the gate, and only as a low-weight (`training.stage_c.
  lambda_dataset_aux`, default 0.05-0.1) optional regularizer --
  `training.stage_c.gate_supervision: hard` exists ONLY as an explicit
  ablation/baseline, never the recommended default. See
  `tests/test_no_dataset_id_supervision.py`, which verifies this
  structurally (the loop never even calls `dataset_aux_loss` when
  `gate_supervision: none`), not just that its weight happens to be zero.
- **Soft gating at inference -- no hard routing, no argmax dataset
  assignment.** `Gate` outputs a dense softmax over dataset-experts;
  `MoEDatasetNIDS`'s combination rule blends every expert's full
  class-probability distribution by that soft weight. Dataset-ambiguous or
  out-of-distribution traffic gets a blended prediction, not a coin-flip
  commitment to one expert -- this is the hypothesis
  `evaluation/ood_ambiguity_eval.py` tests directly against `hard_two_stage`.
- **No dataset identity at inference.** `MoEDatasetNIDS.forward(x)` takes
  exactly one argument; `inference/predict.py` asserts this via
  `inspect.signature` before serving predictions. See
  `tests/test_dataset_blind_inference.py`.
- **Every expert sees every sample.** `DatasetExpertBank.forward` (and
  `AdapterExpertBank.forward`) runs every expert on the full batch
  unconditionally, in Stage C and at inference alike -- no routing/filtering
  step to accidentally introduce. (Stage B is the one deliberate exception:
  it trains one dataset-expert at a time, on purpose, on only that
  dataset's own rows -- see `training/stage_b_warmstart.py`.)
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
  adapters.py                AdapterExpertBank -- FiLM-style lightweight ablation alternative
  gate.py                    Gate (softmax over dataset-experts)
  moe.py                     MoEDatasetNIDS -- soft mixture-of-experts combination rule
  losses.py                  load_balance_penalty (ported) + dataset_aux_loss (low-weight-only)
  baselines.py                PlainPooledSoftmax, NoFusionModel, HardTwoStageModel
training/
  config.py                   YAML loader + --set dotted overrides (ported)
  dataset.py                  prepare_datasets(): raw -> split -> harmonized tensors + dataset_idx
  sampler.py                  ClassBalancedBatchSampler (ported)
  checkpoint.py                per-stage + per-epoch resumable checkpoint I/O
  model_utils.py               architecture -> concrete model wiring (one place)
  stage_a_pretrain.py          Stage A
  stage_b_warmstart.py         Stage B (independent per-dataset expert warm-start)
  stage_c_jointfinetune.py     Stage C (gate + joint fine-tune)
  baseline_train.py            trains plain_pooled / no_fusion / hard_two_stage
  run.py                       CLI entry point (architecture + stage selection via config)
inference/predict.py         single shared-encoder inference path, dataset-blind by construction
evaluation/
  metrics.py                  macro-F1 + per-class AND per-dataset breakdowns
  bootstrap_ci.py              bootstrap CIs for low-sample-count classes (ported)
  ood_ambiguity_eval.py        hard_two_stage vs soft-gated MoE on dataset-ambiguous/OOD traffic
  gate_analysis.py             gate weight distribution, expert utilization, collapse check
  report.py                    comparison tables + Trial_ID-keyed tracker CSVs incl. Per_Dataset_Metrics.csv
notebooks/                   8 Colab-runnable notebooks, one config cell each
tests/                       dataset-blind / no-dataset-id-supervision / active-dataset-toggle tests
```

## One-notebook Google Colab run

[`notebooks/00_colab_end_to_end.ipynb`](notebooks/00_colab_end_to_end.ipynb)
is the canonical start-to-finish Colab workflow. Its single configuration
cell controls smoke/full mode, datasets, architectures, run name, and resume
behavior. It securely clones this private repository with a token read from
Colab Secrets, mounts the shared Drive datasets, runs the tests and full
training pipeline, evaluates the test split, and displays the tracker CSVs
persisted to Drive.

In Colab, add a secret named `GITHUB_TOKEN` (fine-grained token with read-only
Contents access to this repository) and grant the notebook access. Never put
the token directly in a notebook cell or clone URL. The numbered notebooks
remain useful for focused diagnostics and individual-stage experimentation.

## Running a full experiment

```bash
pip install -r requirements.txt

# Primary architecture (Stage A -> B -> C, then evaluation + tracker CSVs):
python -m training.run --config config/default.yaml --set architecture=moe_dataset_soft

# Ablations, through the same entry point:
python -m training.run --config config/default.yaml --set architecture=moe_dataset_hard_gate
python -m training.run --config config/default.yaml --set architecture=moe_dataset_adapters

# Baselines:
python -m training.run --config config/default.yaml --set architecture=plain_pooled
python -m training.run --config config/default.yaml --set architecture=no_fusion
python -m training.run --config config/default.yaml --set architecture=hard_two_stage

# Fast iteration on a 2-dataset subset -- zero code changes:
python -m training.run --config config/default.yaml \
    --set data.active_datasets=[NF-UNSW-NB15-v3,NF-BoT-IoT-v3]

# Ablation: re-run only Stage C from an existing Stage A+B checkpoint
python -m training.run --config config/default.yaml --set training.stages=[C]
```

Results land in `evaluation.output_dir` as `Trials.csv` / `Overall_Metrics.csv`
/ `Per_Class_Metrics.csv` / `Per_Dataset_Metrics.csv`, keyed by `Trial_ID`.
Use `evaluation.report.comparison_table` / `per_dataset_comparison_table` to
pivot several `EvaluationResult`s (one per architecture variant) into a
single ablation table -- see `notebooks/08_full_ablation_report.ipynb`.

## Architecture variants (`architecture` config flag)

| Value | What it is |
|---|---|
| `moe_dataset_soft` | **Primary.** One full expert per dataset, soft gate, `gate_supervision: light_aux`. |
| `moe_dataset_hard_gate` | Ablation: same architecture, `gate_supervision: hard` -- shows what's lost by undoing the "don't supervise the gate on dataset ID" decision. Should converge close to `hard_two_stage`. |
| `moe_dataset_adapters` | Ablation: `AdapterExpertBank` (shared head + per-dataset low-rank FiLM correction) instead of full independent expert heads -- less overfitting risk on small datasets. |
| `plain_pooled` | Baseline: single shared encoder + one joint classification head, no dataset structure. |
| `no_fusion` | Baseline: fully separate encoder+head per dataset, `forward(x, dataset_name)` requires ground-truth dataset ID (train AND inference). |
| `hard_two_stage` | Baseline (new to this project): standalone dataset classifier (stage a) hard-routes each sample to an independent per-dataset classifier (stage b). The literal "obvious two-step approach" this project needs to beat -- `forward(x)` is still dataset-blind (routes on its OWN predicted dataset ID), so it's comparable apples-to-apples on the OOD/ambiguity eval. |

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
   `ClassBalancedBatchSampler` over the TASK class label, not dataset ID.

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
