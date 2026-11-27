"""Task/verdict-stratified ColdSink episode sampling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["stratified_episodes"]


def stratified_episodes(records: Iterable[Any], *, per_stratum: int = 2) -> list[Any]:
    """Fairly sample old and new ColdSink episodes by task and gate verdict."""
    buckets: dict[tuple[str, str], list[Any]] = {}
    for record in records:
        payload = getattr(record, "payload", record)
        if not isinstance(payload, Mapping):
            continue
        state = payload.get("state")
        state = state if isinstance(state, Mapping) else {}
        task = str(state.get("task") or payload.get("task") or "unlabelled")
        history = state.get("gates", payload.get("gates"))
        if isinstance(history, Mapping):
            verdict = "pass" if all(str(value) != "fail" for value in history.values()) else "fail"
        else:
            verdict = str(payload.get("verdict") or "pass")
        buckets.setdefault((task, verdict), []).append(record)
    sample: list[Any] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: float(getattr(item, "created_at", 0.0)))
        if per_stratum <= 0 or not bucket:
            continue
        if len(bucket) <= per_stratum:
            sample.extend(bucket)
        elif per_stratum == 1:
            sample.append(bucket[0])
        else:
            last = len(bucket) - 1
            indices = sorted({round(index * last / (per_stratum - 1)) for index in range(per_stratum)})
            sample.extend(bucket[index] for index in indices)
    return sample
