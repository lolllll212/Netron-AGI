"""Classical O(n * chi^3) backend: numpy SVD as Schmidt truncation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.entropy_sensor import EntanglementReport, entanglement_report

__all__ = ["ClassicalSVDBackend"]


class ClassicalSVDBackend:
    """Exact SVD of the bipartite coupling matrix, truncated at ``chi_max``.

    Cost is O(min(n, m)^2 * max(n, m)) for the decomposition itself; the
    truncation is a slice, so the kernel's per-tick overhead stays sub-second
    for matrices up to a few thousand entities.
    """

    name = "classical_svd"

    def measure_entanglement(
        self,
        coupling_matrix: Any,
        chi_max: int = 4,
        tau: float = 0.15,
    ) -> EntanglementReport:
        return entanglement_report(coupling_matrix, chi_max=chi_max, tau=tau)

    def truncate(self, coupling_matrix: Any, chi_max: int = 4) -> np.ndarray:
        """Optimal rank-``chi_max`` approximation (Eckart–Young)."""
        array = np.asarray(coupling_matrix, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {array.shape}")
        if array.size == 0:
            return array.copy()
        u, s, vt = np.linalg.svd(array, full_matrices=False)
        k = min(max(1, chi_max), s.size)
        return (u[:, :k] * s[:k]) @ vt[:k]

    def __repr__(self) -> str:  # pragma: no cover - cosmetics
        return "ClassicalSVDBackend()"
