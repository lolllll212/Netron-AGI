"""Relaxation backends: the only place where "physics" is implemented.

The kernel talks to a ``RelaxationBackend`` protocol, never to numpy directly,
so a bounded-χ tensor backend, a p-bit simulator, or an accelerator can be
dropped in without touching the agent loop.
"""

from __future__ import annotations

from .base import RelaxationBackend
from .bounded_chi import BoundedChiBackend, available_thrml_adapter
from .classical_svd import ClassicalSVDBackend
from .thrml_backend import ThrmlBackend

__all__ = [
    "BoundedChiBackend",
    "ClassicalSVDBackend",
    "RelaxationBackend",
    "ThrmlBackend",
    "available_backends",
    "available_thrml_adapter",
    "get_backend",
]

_FACTORIES = {
    "classical_svd": ClassicalSVDBackend,
    "bounded_chi": BoundedChiBackend,
    "thrml": ThrmlBackend,
    "thrml_informed": ThrmlBackend,
    "thrml_pbit": ThrmlBackend,
}


def available_backends() -> tuple[str, ...]:
    """Names registered in this build."""
    return tuple(sorted(_FACTORIES))


def get_backend(name: str = "classical_svd", **kwargs: object) -> RelaxationBackend:
    """Instantiate a backend by name; THRML/JAX remains an optional adapter."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; available: {', '.join(available_backends())}") from None
    return factory(**kwargs)  # type: ignore[call-arg]
