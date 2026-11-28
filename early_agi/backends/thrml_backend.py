"""Compatibility facade for the THRML-informed thermodynamic backend.

The bounded-χ implementation lives in ``bounded_chi.py``. This module gives the
manifest's ``thrml_backend.py`` name a stable import target while keeping JAX
and THRML strictly optional.
"""

from __future__ import annotations

from .bounded_chi import (
    BlockGibbsSchedule,
    BoundedChiBackend,
    FactorizationResult,
    GibbsSample,
    THRMLAdapter,
    ThrmlUnavailableError,
    available_thrml_adapter,
)

__all__ = [
    "BlockGibbsSchedule",
    "BoundedChiBackend",
    "FactorizationResult",
    "GibbsSample",
    "THRMLAdapter",
    "ThrmlBackend",
    "ThrmlUnavailableError",
    "available_thrml_adapter",
]


class ThrmlBackend(BoundedChiBackend):
    """Optional-adapter alias of the pure-Python block-Gibbs backend."""

    name = "thrml_informed"
