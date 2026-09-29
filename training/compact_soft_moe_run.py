"""Fixed three-seed reproduction of the historical compact soft dataset-MoE.

The shared aggregation and latent-report machinery is inherited from the
notebook-30 runner.  This runner deliberately has only one training command
per seed because the compact model uses one shared encoder and keeps A/B/C in
one signed, resumable checkpoint directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone

import pandas as pd

from .recommended_private_run import (
    RecommendedPrivateStudy,
    _atomic_csv,
    _atomic_json,
)
from .training_history import (
    HISTORY_COLUMNS,
    load_or_reconstruct_history,
    plot_seed_training_history,
)


class CompactSoftMoEStudy(RecommendedPrivateStudy):
    """Three-seed fixed study for the 20,380-parameter historical topology."""

    def _base_nested(self, seed: int, run_name: str) -> dict:
        nested = copy.deepcopy(self.study["backbone"])
        nested["run_name"] = run_name
        nested["seed"] = int(seed)
        nested.setdefault("data", {})["split_seed"] = int(seed)
        nested.setdefault("evaluation", {}).setdefault("latent", {})[
            "random_seed"
        ] = int(seed)
        nested.setdefault("training", {})["save_epoch_history"] = True
        return nested

    def _final_nested(self, seed: int, *, evaluate: bool = True) -> dict:
        nested = self._base_nested(seed, f"{self.prefix}_seed{seed}")
        nested["training"]["stages"] = ["A", "B", "C"]
        nested["training"]["run_final_evaluation"] = bool(evaluate)
        return nested

    def _final_config(self, seed: int) -> dict:
        return self._config(self._final_nested(seed))

    def _materialize_training_report(self, seed: int) -> None:
        if not self.execute:
            print(
                f"[training-report] seed {seed}: combine A/B/C history, logs, and curves",
                flush=True,
            )
            return
        config = self._final_config(seed)
        checkpoint_dir = config["training"]["checkpoint_dir"]
        training_dir = self._training_dir(seed)
        os.makedirs(training_dir, exist_ok=True)
        roles = {"A": "encoder", "B": "expert", "C": "full_model"}
        histories = []
        for stage, role in roles.items():
            stage_cfg = config["training"].get(f"stage_{stage.lower()}", {})
            optimizer = stage_cfg.get("optimizer", {}) or {}
            history = load_or_reconstruct_history(
                checkpoint_dir,
                seed=seed,
                stage=stage,
                encoder_role=role,
                learning_rate=float(optimizer.get("lr", config["training"]["lr"])),
            )
            if not history.empty:
                histories.append(history)
        history = (
            pd.concat(histories, ignore_index=True)
            if histories
            else pd.DataFrame(columns=HISTORY_COLUMNS)
        )
        summary_path = os.path.join(checkpoint_dir, "stage_c_training_summary.json")
        selected_epoch = None
        if os.path.isfile(summary_path):
            with open(summary_path) as handle:
                selected_epoch = json.load(handle).get("best_epoch")
        _atomic_csv(history, os.path.join(training_dir, "Training_History.csv"))
        log_path = os.path.join(checkpoint_dir, "train.log")
        log_manifest = pd.DataFrame([{
            "seed": seed,
            "stage": "A|B|C",
            "encoder_role": "encoder|expert|full_model",
            "checkpoint_dir": checkpoint_dir,
            "log_path": log_path,
            "log_exists": os.path.isfile(log_path),
            "log_size_bytes": os.path.getsize(log_path) if os.path.isfile(log_path) else 0,
            "structured_history_exists": os.path.isfile(os.path.join(
                checkpoint_dir, "Training_History.csv"
            )),
            "history_rows": int(len(history)),
        }])
        _atomic_csv(log_manifest, os.path.join(training_dir, "Training_Log_Manifest.csv"))
        figures = plot_seed_training_history(
            history,
            os.path.join(training_dir, "training_figures"),
            selected_epoch=selected_epoch,
        )
        present_stages = sorted(set(history["stage"].astype(str))) if not history.empty else []
        _atomic_json({
            "format_version": 1,
            "seed": seed,
            "selected_stage_c_epoch": selected_epoch,
            "present_stages": present_stages,
            "missing_stages": [stage for stage in ("A", "B", "C") if stage not in present_stages],
            "history_sources": sorted(set(history["history_source"].astype(str))) if not history.empty else [],
            "figure_files": figures,
            "log_manifest": os.path.join(training_dir, "Training_Log_Manifest.csv"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }, os.path.join(training_dir, "Training_Report_Config.json"))
        print(
            f"[training-report] seed {seed}: rows={len(history)} figures={len(figures)} "
            f"directory={training_dir}",
            flush=True,
        )

    def run_seed(self, seed: int) -> None:
        if self.execute:
            self._preflight(seed)
        if self.execute and self._test_complete(seed):
            print(f"[train-and-test] seed {seed}: complete; skipping", flush=True)
        else:
            self._run("train-and-test", self._final_nested(seed, evaluate=True))
        for stage in ("A", "B", "C"):
            self._materialize_latent(seed, stage)
        self._materialize_training_report(seed)
        if self.execute and not self._seed_complete(seed):
            raise RuntimeError(
                f"seed {seed} did not produce all required task, latent, and training artifacts"
            )
        if self.execute:
            self.state["seeds"][str(seed)] = {
                "status": "complete",
                "result_dir": self._result_dir(seed),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--study-config", default="config/compact_soft_moe_3seed.yaml")
    parser.add_argument("--prefix", default="nfv3_4way_moe_soft_comparable_repro_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-seeds", type=int)
    args = parser.parse_args()
    runner = CompactSoftMoEStudy(
        base_config_path=args.config,
        study_config_path=args.study_config,
        prefix=args.prefix,
        execute=args.execute,
        max_new_seeds=args.max_new_seeds,
    )
    runner.run()


if __name__ == "__main__":
    main()
