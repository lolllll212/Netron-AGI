"""Secret-safe live progress, circuit breaking and atomic checkpoints."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def stderr_progress(message):
    print(message, file=sys.stderr, flush=True)


class LiveMonitor:
    def __init__(self, *, api_key, progress=None, failure_limit=2, heartbeat=15):
        if failure_limit < 1:
            raise ValueError("transport failure limit must be positive")
        self.key = api_key
        self.progress = progress
        self.failure_limit = failure_limit
        self.heartbeat = heartbeat
        self.consecutive_failures = 0
        self.stopped = False
        self.errors = []
        self._lock = threading.RLock()

    def safe(self, text):
        text = str(text)
        if self.key:
            text = text.replace(self.key, "[REDACTED]")
        return re.sub(r"(?i)(bearer\s+\S+|nvapi-[\w-]+)", "[REDACTED]", text)[:800]

    def emit(self, event, **fields):
        if self.progress:
            with self._lock:
                self.progress(
                    self.safe(
                        f"[{time.strftime('%H:%M:%S')}] {event} "
                        + " ".join(f"{k}={v}" for k, v in fields.items())
                    )
                )

    def failure(self, exc, *, transport=False):
        with self._lock:
            detail = {"type": type(exc).__name__, "message": self.safe(exc), "transport": transport}
            self.errors.append(detail)
            if transport:
                self.consecutive_failures += 1
                self.stopped = self.stopped or self.consecutive_failures >= self.failure_limit
            self.emit("request_failed", **detail)
            if self.stopped:
                self.emit("circuit_open", consecutive_failures=self.consecutive_failures)

    def success(self):
        with self._lock:
            self.consecutive_failures = 0

    @contextmanager
    def waiting(self, *, model, timeout):
        done = threading.Event()
        started = time.monotonic()
        self.emit("request_start", model=model, timeout_s=timeout)

        def heartbeat():
            while not done.wait(self.heartbeat):
                self.emit("request_waiting", model=model, elapsed_s=round(time.monotonic() - started))

        thread = threading.Thread(target=heartbeat, daemon=True)
        if self.progress:
            thread.start()
        try:
            yield
        finally:
            done.set()
            if self.progress:
                thread.join()
            self.emit("request_end", model=model, elapsed_s=round(time.monotonic() - started, 1))


def checkpoint(path, report):
    """Replace only fully serialized reports; never expose a truncated JSON file."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
