"""The backend seam."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.entropy_sensor import EntanglementReport

__all__ = ["RelaxationBackend"]


@runtime_checkable
class RelaxationBackend(Protocol):
    """Polynomial-cost relaxation of the Schmidt spectrum."""

    name: str

    def measure_entanglement(
        self,
        coupling_matrix: Any,
        chi_max: int = 4,
        tau: float = 0.15,
    ) -> EntanglementReport:
        """Return entropy, truncation cost and regime classification."""
        ...

    def truncate(self, coupling_matrix: Any, chi_max: int = 4) -> Any:
        """Best rank-chi_max approximation of the coupling matrix."""
        ...

    def left_basis(self, coupling_matrix: Any, k: int = 4) -> Any:
        """First k left singular vectors, for deterministic IR clustering."""
        ...
