"""The backend seam.

Anything that can answer "how entangled is this coupling matrix, and what does
it cost to truncate at chi_max?" satisfies this protocol. Keeping it this small
is deliberate: it is the contract that lets numpy SVD be replaced by tensor
networks or a p-bit sampler without an agent-loop rewrite.
"""

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
