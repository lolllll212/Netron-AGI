"""Classical numpy SVD backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.entropy_sensor import EntanglementReport, entanglement_report

__all__ = ["ClassicalSVDBackend"]


class ClassicalSVDBackend:
    name = "classical_svd"

    def measure_entanglement(
        self,
        coupling_matrix: Any,
        chi_max: int = 4,
        tau: float = 0.15,
    ) -> EntanglementReport:
        return entanglement_report(coupling_matrix, chi_max=chi_max, tau=tau)

    def truncate(self, coupling_matrix: Any, chi_max: int = 4) -> np.ndarray:
        array = np.asarray(coupling_matrix, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {array.shape}")
        if array.size == 0:
            return array.copy()
        u, s, vt = np.linalg.svd(array, full_matrices=False)
        k = min(max(1, chi_max), s.size)
        return (u[:, :k] * s[:k]) @ vt[:k]

    def left_basis(self, coupling_matrix: Any, k: int = 4) -> np.ndarray:
        array = np.asarray(coupling_matrix, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.size == 0:
            return np.zeros((array.shape[0] if array.ndim == 2 else 1, 0), dtype=np.float64)
        u, _, _ = np.linalg.svd(array, full_matrices=False)
        return u[:, : min(max(1, k), u.shape[1])]

    def __repr__(self) -> str:  # pragma: no cover
        return "ClassicalSVDBackend()"
