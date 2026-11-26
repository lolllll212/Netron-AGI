"""Relaxation backends: the only place where "physics" is implemented.

The kernel talks to a ``RelaxationBackend`` protocol, never to numpy directly,
so a PyTorch tensor-network backend, a p-bit simulator (thrml) or an accelerator
can be dropped in without touching the agent loop.
"""

from __future__ import annotations

from .base import RelaxationBackend
from .classical_svd import ClassicalSVDBackend

__all__ = ["RelaxationBackend", "ClassicalSVDBackend", "available_backends", "get_backend"]

_FACTORIES = {
    "classical_svd": ClassicalSVDBackend,
}


def available_backends() -> tuple[str, ...]:
    """Names registered in this build."""
    return tuple(sorted(_FACTORIES))


def get_backend(name: str = "classical_svd", **kwargs: object) -> RelaxationBackend:
    """Instantiate a backend by name (future: ``torch_mps``, ``thrml_pbit``)."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; available: {', '.join(available_backends())}") from None
    return factory(**kwargs)  # type: ignore[call-arg]
