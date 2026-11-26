"""SVD spectral entropy and the area-law / volume-law detector.

Given a coupling matrix ``M`` (rows = entities, columns = constraints or
hypotheses), the squared singular values are the Schmidt weights of the
bipartition. Their Shannon entropy is the *verified* coupling complexity:

* area law  — entropy stays small as the system grows; the problem decomposes
  along its boundary and truncating at ``chi_max`` costs almost nothing.
* volume law — entropy grows with system size; weight is spread over many
  modes, truncation error explodes, and naive local patching is hopeless.

``truncation_error`` is the discarded weight beyond bond dimension ``chi_max``,
i.e. the exact price of pretending the problem is modular.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "EntanglementReport",
    "entanglement_report",
    "matrix_fingerprint",
    "schmidt_weights",
    "spectral_entropy_bits",
]

NOISE_FLOOR = 1e-12


@dataclass(frozen=True)
class EntanglementReport:
    spectral_entropy_bits: float  # S_SVD: coupling complexity across active modes
    truncation_error: float  # weight discarded beyond chi_max
    effective_rank: int  # modes above the noise floor
    is_volume_law: bool  # True when coupling overflows the bond dimension
    max_entropy_bits: float = 0.0  # log2(rank) ceiling, for context
    top_weight: float = 0.0  # weight of the dominant mode

    @property
    def saturation(self) -> float:
        """0.0 = rank-1 (fully disentangled), 1.0 = maximally mixed."""
        if self.max_entropy_bits <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self.spectral_entropy_bits / self.max_entropy_bits))

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectral_entropy_bits": round(self.spectral_entropy_bits, 6),
            "truncation_error": round(self.truncation_error, 6),
            "effective_rank": self.effective_rank,
            "is_volume_law": self.is_volume_law,
            "max_entropy_bits": round(self.max_entropy_bits, 6),
            "top_weight": round(self.top_weight, 6),
        }


def _as_matrix(coupling_matrix: Any) -> np.ndarray:
    array = np.asarray(coupling_matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"coupling matrix must be 1-D or 2-D, got shape {array.shape}")
    return array


def schmidt_weights(coupling_matrix: Any) -> np.ndarray:
    """Normalised squared singular values, noise-floor filtered."""
    array = _as_matrix(coupling_matrix)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        singular = np.linalg.svd(array, compute_uv=False)
    except np.linalg.LinAlgError:  # pragma: no cover - pathological input
        return np.zeros(0, dtype=np.float64)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros(0, dtype=np.float64)
    singular = singular[singular > max(singular[0] * 1e-10, NOISE_FLOOR)]
    weights = singular**2
    return weights / weights.sum()


def spectral_entropy_bits(coupling_matrix: Any) -> float:
    """Shannon entropy of the Schmidt spectrum, in bits."""
    weights = schmidt_weights(coupling_matrix)
    if weights.size == 0:
        return 0.0
    # max() also normalises the rank-1 case, where -sum() returns -0.0
    return float(max(0.0, -np.sum(weights * np.log2(np.maximum(weights, np.finfo(float).tiny)))))


def entanglement_report(
    coupling_matrix: Any,
    chi_max: int = 4,
    tau: float = 0.15,
) -> EntanglementReport:
    """Measure coupling complexity and classify the entanglement regime."""
    if chi_max < 1:
        raise ValueError("chi_max must be >= 1")
    weights = schmidt_weights(coupling_matrix)
    if weights.size == 0:
        return EntanglementReport(0.0, 0.0, 0, False)

    entropy = float(max(0.0, -np.sum(weights * np.log2(np.maximum(weights, np.finfo(float).tiny)))))
    truncation = float(np.sum(weights[chi_max:])) if weights.size > chi_max else 0.0
    effective_rank = int(np.count_nonzero(weights > 1e-3))
    max_entropy = float(np.log2(weights.size)) if weights.size > 1 else 0.0

    return EntanglementReport(
        spectral_entropy_bits=entropy,
        truncation_error=truncation,
        effective_rank=effective_rank,
        is_volume_law=bool(truncation > tau),
        max_entropy_bits=max_entropy,
        top_weight=float(weights[0]),
    )


def matrix_fingerprint(coupling_matrix: Any, *, digits: int = 6) -> str:
    """Stable digest of a coupling matrix, for reproducible tick records."""
    array = np.round(_as_matrix(coupling_matrix), digits) + 0.0  # +0.0 kills -0.0
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()
