"""Thread-shared request pacing and provider Retry-After parsing."""

from __future__ import annotations

import threading
import time
from email.utils import parsedate_to_datetime


class RequestPacer:
    """Reserve evenly spaced request starts, including retries, across clients."""

    def __init__(self, rpm_limit: float = 40) -> None:
        if rpm_limit <= 0:
            raise ValueError("rpm_limit must be positive")
        self.interval = 60.0 / rpm_limit
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0
