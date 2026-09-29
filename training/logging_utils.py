"""Tiny output-to-file tee used by long-running Colab commands.

Both stdout and stderr are persisted so progress messages and uncaught
tracebacks survive a Colab disconnect while remaining visible live.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from typing import IO, Sequence


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


def run_streaming_logged(
    command: Sequence[str],
    log_path: str,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    heartbeat_seconds: float = 30.0,
    tail_lines: int = 40,
    label: str = "training",
    output_stream: IO[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with enforced live output and an identical durable log.

    A reader thread prevents the notebook's main thread from blocking on a
    silent child, which lets it emit periodic heartbeat lines.  Both child
    output and monitor messages are written to the supplied stream and log.
    """
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if tail_lines < 1:
        raise ValueError("tail_lines must be positive")
    output_stream = output_stream or sys.stdout
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    lines: queue.Queue[str | None] = queue.Queue()
    tail: deque[str] = deque(maxlen=tail_lines)

    with open(log_path, "a", buffering=1, encoding="utf-8") as log_file:
        def emit(message: str) -> None:
            output_stream.write(message)
            output_stream.flush()
            log_file.write(message)
            log_file.flush()

        emit(
            f"\n[stream] {label} started_utc={datetime.now(timezone.utc).isoformat()} "
            f"log={log_path}\n[stream] command={shlex.join(list(command))}\n"
        )
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        emit(f"[stream] pid={process.pid} heartbeat_seconds={heartbeat_seconds:g}\n")

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                process.stdout.close()
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_output = time.monotonic()
        last_child_line = "child has not emitted a status line yet"
        finished = False
        try:
            while not finished:
                timeout = max(0.01, min(1.0, heartbeat_seconds))
                try:
                    line = lines.get(timeout=timeout)
                except queue.Empty:
                    line = ""
                if line is None:
                    finished = True
                elif line:
                    emit(line)
                    tail.append(line.rstrip("\n"))
                    last_child_line = line.strip()[-240:] or last_child_line
                    last_output = time.monotonic()
                now = time.monotonic()
                if not finished and now - last_output >= heartbeat_seconds:
                    emit(
                        f"[heartbeat] {label} pid={process.pid} still running "
                        f"elapsed={now - started:.0f}s waiting_for_output "
                        f"last_status={last_child_line!r}\n"
                    )
                    last_output = now
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            emit(
                f"[stream] {label} interrupted; child stopped return_code={process.returncode}\n"
            )
            raise

        reader.join(timeout=1.0)
        return_code = process.wait()
        elapsed = time.monotonic() - started
        status = "completed" if return_code == 0 else "failed"
        emit(
            f"[stream] {label} {status} return_code={return_code} "
            f"elapsed={elapsed:.1f}s log={log_path}\n"
        )
        if return_code != 0:
            emit(f"[stream] last {len(tail)} child-output line(s):\n")
            for line in tail:
                emit(f"[tail] {line}\n")
            raise subprocess.CalledProcessError(return_code, list(command))
    return subprocess.CompletedProcess(list(command), return_code)
