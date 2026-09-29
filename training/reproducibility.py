"""Process-level seeding for explicitly deterministic experiment runs."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_reproducibility(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG used by the training stack.

    The data pipeline also constructs explicit per-stage NumPy generators from
    ``seed``.  This process-level setup covers initialization, dropout, Python
    hashing/randomness, and every CUDA device.  Deterministic algorithms are
    opt-in so historical/default experiments retain their previous behavior.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
