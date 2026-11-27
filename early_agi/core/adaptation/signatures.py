"""One scale-invariant structural-signature encoder for both core façades.

``StructuralSignature`` is the runtime/plasticity view (``motif_ratios``).
``AdaptationSignature`` is the historical ``core.adaptability`` view (``motifs``).
Both are thin data views over the exact same SVD, motif, cosine, and
analogy-matching logic; there is no duplicate encoder or novelty tracker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "AdaptationSignature",
    "AnalogicalMatcher",
    "AnalogyMatch",
    "StructuralSignature",
    "Topology",
    "encode_topology",
    "recall_strategy",
]


@dataclass(frozen=True)
class StructuralSignature:
    """Coordinate-free topological fingerprint used by MERA and reflexes."""

    spectrum: tuple[float, ...]
    motif_ratios: tuple[float, ...]

    @property
    def motifs(self) -> tuple[float, ...]:
        return self.motif_ratios

    def to_vector(self) -> np.ndarray:
        return np.asarray([*self.spectrum, *self.motif_ratios], dtype=np.float64)

    def vector(self) -> np.ndarray:
        return self.to_vector()

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectrum": list(self.spectrum),
            "motif_ratios": list(self.motif_ratios),
        }

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> StructuralSignature:
        motifs = data.get("motif_ratios", data.get("motifs", ()))
        return cls(
            spectrum=tuple(float(item) for item in data.get("spectrum", ())),
            motif_ratios=tuple(float(item) for item in motifs),
        )


@dataclass(frozen=True)
class AdaptationSignature:
    """Compatibility view with the historical ``motifs`` field and dict key."""

    spectrum: tuple[float, ...]
    motifs: tuple[float, ...]

    @property
    def motif_ratios(self) -> tuple[float, ...]:
        return self.motifs

    def vector(self) -> np.ndarray:
        return np.asarray([*self.spectrum, *self.motifs], dtype=np.float64)

    def to_vector(self) -> np.ndarray:
        return self.vector()

    def as_dict(self) -> dict[str, Any]:
        return {"spectrum": list(self.spectrum), "motifs": list(self.motifs)}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> AdaptationSignature:
        motifs = data.get("motifs", data.get("motif_ratios", ()))
        return cls(
            spectrum=tuple(float(item) for item in data.get("spectrum", ())),
            motifs=tuple(float(item) for item in motifs),
        )


@dataclass(frozen=True)
class AnalogyMatch:
    """Best structural analogue found in MERA memory."""

    pattern_id: str
    similarity: float
    strategy_hint: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "similarity": self.similarity,
            "strategy_hint": self.strategy_hint,
            "metadata": dict(self.metadata),
        }


class AnalogicalMatcher:
    """Canonical scale-invariant encoder and analogy store."""

    def __init__(self, top_k_singular: int = 6) -> None:
        self.top_k = max(1, int(top_k_singular))
        self._patterns: list[tuple[str, StructuralSignature, str, dict[str, Any]]] = []

    @staticmethod
    def _matrix(coupling_matrix: Any) -> np.ndarray:
        values = np.asarray(coupling_matrix, dtype=np.float64)
        if values.ndim == 0:
            return values.reshape(1, 1)
        return np.atleast_2d(values)

    def encode(
        self,
        coupling_matrix: Any,
        *,
        num_modules: int,
        num_imports: int,
        num_functions: int,
        num_gates: int,
    ) -> StructuralSignature:
        matrix = self._matrix(coupling_matrix)
        if matrix.size == 0:
            spec = tuple(0.0 for _ in range(self.top_k))
            density = 0.0
        else:
            singular = np.linalg.svd(matrix, compute_uv=False)
            norm = float(np.linalg.norm(singular))
            normalised = singular / norm if norm > 1e-12 else singular
            padded = np.zeros(self.top_k, dtype=np.float64)
            take = min(self.top_k, normalised.size)
            padded[:take] = normalised[:take]
            spec = tuple(float(item) for item in padded)
            density = float(np.count_nonzero(np.abs(matrix) > 1e-9)) / float(matrix.size)

        scale = float(max(int(num_modules), 1))
        motifs = (
            float(min(int(num_modules) / 50.0, 1.0)),
            float(min(int(num_imports) / (scale * 4.0), 1.0)),
            float(min(int(num_functions) / (scale * 8.0), 1.0)),
            float(min(int(num_gates) / 10.0, 1.0)),
            float(np.clip(density, 0.0, 1.0)),
        )
        return StructuralSignature(spectrum=spec, motif_ratios=motifs)

    def register(
        self,
        pattern_id: str,
        signature: StructuralSignature,
        strategy_hint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(metadata or {})
        for index, (existing_id, *_rest) in enumerate(self._patterns):
            if existing_id == pattern_id:
                self._patterns[index] = (pattern_id, signature, strategy_hint, payload)
                return
        self._patterns.append((pattern_id, signature, strategy_hint, payload))

    def best_match(self, signature: StructuralSignature, min_similarity: float = 0.72) -> AnalogyMatch | None:
        query = signature.to_vector()
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 1e-12 or not self._patterns:
            return None

        best: AnalogyMatch | None = None
        for pattern_id, candidate, hint, meta in self._patterns:
            vector = candidate.to_vector()
            v_norm = float(np.linalg.norm(vector))
            if v_norm <= 1e-12:
                continue
            similarity = float(np.dot(query, vector) / (q_norm * v_norm))
            if similarity >= min_similarity and (best is None or similarity > best.similarity):
                best = AnalogyMatch(
                    pattern_id=pattern_id,
                    similarity=round(similarity, 4),
                    strategy_hint=hint,
                    metadata=dict(meta),
                )
        return best

    @property
    def patterns(self) -> tuple[tuple[str, StructuralSignature, str, dict[str, Any]], ...]:
        return tuple((pid, sig, hint, dict(meta)) for pid, sig, hint, meta in self._patterns)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "pattern_id": pid,
                "signature": sig.as_dict(),
                "strategy_hint": hint,
                "metadata": dict(meta),
            }
            for pid, sig, hint, meta in self._patterns
        ]

    def restore(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._patterns.clear()
        for item in records:
            signature = StructuralSignature.from_payload(item.get("signature", {}))
            self.register(
                str(item.get("pattern_id", "")),
                signature,
                str(item.get("strategy_hint", "")),
                item.get("metadata", {}),
            )


@dataclass(frozen=True)
class Topology:
    """A synthetic repository topology or a parsed coupling matrix."""

    coupling: Any
    modules: int
    imports: int
    functions: int
    gates: int

    def signature(self) -> AdaptationSignature:
        return encode_topology(self)


def encode_topology(topology: Topology | Mapping[str, Any], top_k: int = 6) -> AdaptationSignature:
    """Encode through the canonical matcher, then expose the adaptation view."""
    if isinstance(topology, Mapping):
        topology = Topology(
            topology.get("coupling", np.eye(1)),
            int(topology.get("modules", 1)),
            int(topology.get("imports", 0)),
            int(topology.get("functions", 0)),
            int(topology.get("gates", 0)),
        )
    encoded = AnalogicalMatcher(top_k_singular=top_k).encode(
        topology.coupling,
        num_modules=topology.modules,
        num_imports=topology.imports,
        num_functions=topology.functions,
        num_gates=topology.gates,
    )
    return AdaptationSignature(encoded.spectrum, encoded.motif_ratios)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def recall_strategy(
    signature: StructuralSignature | AdaptationSignature,
    examples: Sequence[tuple[StructuralSignature | AdaptationSignature, Mapping[str, Any]]],
    *,
    min_similarity: float = 0.75,
) -> tuple[dict[str, Any], float] | None:
    """Recall a decomposition strategy across different file/function names."""
    query = signature.vector()
    best: tuple[dict[str, Any], float] | None = None
    for candidate, strategy in examples:
        similarity = _cosine(query, candidate.vector())
        if similarity >= min_similarity and (best is None or similarity > best[1]):
            best = (dict(strategy), round(similarity, 6))
    return best
