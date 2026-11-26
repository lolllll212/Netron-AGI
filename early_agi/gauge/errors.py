"""Gauge runtime errors shared by the VM's focused modules."""

from __future__ import annotations

__all__ = ["GaugeRuntimeError"]


class GaugeRuntimeError(RuntimeError):
    """A checked Gauge program failed a runtime invariant."""
