"""Tiny output-to-file tee used by long-running Colab commands.

Both stdout and stderr are persisted so progress messages and uncaught
tracebacks survive a Colab disconnect while remaining visible live.
"""
from __future__ import annotations

import os
import sys


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, msg: str) -> None:
        for s in self._streams:
            s.write(msg)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def tee_stdout_to_file(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
