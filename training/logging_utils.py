"""Tiny stdout-to-file tee, shared by training.run and inference.predict, so
CLI output survives a Colab disconnect -- inspectable from Drive afterward
alongside checkpoints, in addition to the live stdout stream."""
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
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)
