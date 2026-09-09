"""Tiny output-to-file tee used by long-running Colab commands.

Both stdout and stderr are persisted so progress messages and uncaught
tracebacks survive a Colab disconnect while remaining visible live.
"""
from __future__ import annotations

import os
import sys


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = list(streams)

    def write(self, msg: str) -> None:
        for index, stream in enumerate(list(self._streams)):
            try:
                stream.write(msg)
            except (OSError, ValueError) as exc:
                if index == 0:
                    raise
                self._streams.remove(stream)
                self._streams[0].write(
                    f"\n[logging] persistent log became unavailable; continuing live only: {exc}\n"
                )
                self._streams[0].flush()

    def flush(self) -> None:
        for index, stream in enumerate(list(self._streams)):
            try:
                stream.flush()
            except (OSError, ValueError):
                if index == 0:
                    raise
                self._streams.remove(stream)


def tee_stdout_to_file(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
