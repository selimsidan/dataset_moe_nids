"""Resolves where raw NIDS dataset files live (Colab Drive, local Google Drive
mirror, or an explicit override), and where run outputs are written.

Ported unchanged in spirit from moe_nids/data/paths.py -- same env-var
overrides (`NIDS_DRIVE_BASE`, `NIDS_OUTPUT_DIR`, `NIDS_SCRATCH_DIR`), same
raw dataset location (`NIDS_datasets/`), so both projects can point at the
same Drive-mounted data without duplicating it. Only the default output
sub-folder name differs (`dataset_moe_nids_runs` instead of
`moe_nids_runs`), so a run of this project never clobbers moe_nids results.
"""
from __future__ import annotations

import glob
import os
import shutil

DEFAULT_COLAB_DRIVE_BASE = "/content/drive/MyDrive/NIDS_datasets"
LOCAL_DRIVE_BASE_CANDIDATES = [
    os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/NIDS_datasets"),
    os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/MyDrive/NIDS_datasets"),
    os.path.expanduser("~/Google Drive/My Drive/NIDS_datasets"),
    os.path.expanduser("~/Google Drive/NIDS_datasets"),
]


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


IN_COLAB = in_colab()


def resolve_scratch_dir() -> str:
    """Fast local (non-Drive-synced) directory for heavy/frequent reads and
    writes during training -- see moe_nids/data/paths.py::resolve_scratch_dir
    for the full rationale (FUSE-mounted Drive is slow for per-epoch writes).
    """
    if "NIDS_SCRATCH_DIR" in os.environ:
        return os.environ["NIDS_SCRATCH_DIR"]
    if IN_COLAB:
        return "/content/dataset_moe_nids_scratch"
    return os.path.abspath(os.path.join(os.getcwd(), ".local_scratch"))


def resolve_drive_base() -> str:
    if "NIDS_DRIVE_BASE" in os.environ:
        return os.environ["NIDS_DRIVE_BASE"]
    if os.path.isdir(DEFAULT_COLAB_DRIVE_BASE):
        return DEFAULT_COLAB_DRIVE_BASE
    local_matches: list[str] = []
    for pattern in LOCAL_DRIVE_BASE_CANDIDATES:
        local_matches.extend(glob.glob(pattern))
    return local_matches[0] if local_matches else os.getcwd()


def resolve_output_dir(drive_base: str) -> str:
    if "NIDS_OUTPUT_DIR" in os.environ:
        return os.environ["NIDS_OUTPUT_DIR"]
    if os.path.isdir("/content/drive/MyDrive"):
        return os.path.join("/content/drive/MyDrive", "NIDS_analysis_outputs", "dataset_moe_nids_runs")
    if drive_base != os.getcwd():
        return os.path.join(
            os.path.dirname(drive_base.rstrip(os.sep)), "NIDS_analysis_outputs", "dataset_moe_nids_runs"
        )
    return os.path.abspath(os.path.join("NIDS_analysis_outputs", "dataset_moe_nids_runs"))


DRIVE_BASE = resolve_drive_base()
OUTPUT_DIR = resolve_output_dir(DRIVE_BASE)
SCRATCH_DIR = resolve_scratch_dir()

# Optional per-dataset path overrides, e.g.:
#   PATH_OVERRIDES = {"CICIoT2023": "/content/drive/MyDrive/NIDS_datasets/CICIoT2023/CSV"}
PATH_OVERRIDES: dict[str, str] = {}


def dataset_paths(name: str, default_paths: list[str]) -> list[str]:
    """Applies PATH_OVERRIDES, falling back to the registry-provided defaults."""
    if name in PATH_OVERRIDES:
        return [PATH_OVERRIDES[name]]
    return default_paths


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def local_mirror_dir(drive_dir: str) -> str:
    """Maps a Drive-backed artifact dir onto a same-named directory under
    SCRATCH_DIR, so writes during training hit local disk and only get
    synced back to Drive at defined checkpoints (see sync_to_drive)."""
    key = os.path.relpath(os.path.abspath(drive_dir), os.path.abspath(OUTPUT_DIR)) if drive_dir.startswith(OUTPUT_DIR) else os.path.basename(drive_dir.rstrip(os.sep))
    return os.path.join(SCRATCH_DIR, "artifacts", key)


def sync_to_drive(local_dir: str, drive_dir: str) -> None:
    """Copies any new/changed files from local_dir up to drive_dir. Cheap
    no-op if local_dir doesn't exist yet or nothing changed since the last
    sync (compares size + mtime, not full content)."""
    if not os.path.isdir(local_dir):
        return
    os.makedirs(drive_dir, exist_ok=True)
    for name in os.listdir(local_dir):
        src = os.path.join(local_dir, name)
        dst = os.path.join(drive_dir, name)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            src_stat, dst_stat = os.stat(src), os.stat(dst)
            if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                continue
        shutil.copy2(src, dst)


def sync_from_drive(drive_dir: str, local_dir: str) -> None:
    """Hydrates local_dir from drive_dir -- used at the start of a (resumed)
    run so a fresh Colab runtime picks up checkpoints written by a previous,
    disconnected session before deciding whether to resume."""
    if not os.path.isdir(drive_dir):
        return
    os.makedirs(local_dir, exist_ok=True)
    for name in os.listdir(drive_dir):
        src = os.path.join(drive_dir, name)
        dst = os.path.join(local_dir, name)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            src_stat, dst_stat = os.stat(src), os.stat(dst)
            if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                continue
        shutil.copy2(src, dst)
