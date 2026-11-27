"""Dependency-light adaptive plasticity for the thermodynamic kernel.

Three engines live here:

* :class:`HomeostaticRegulator` turns prediction surprise and analogical
  familiarity into live ``eta_min`` / ``eta_high`` / ``tau`` / ``chi_max``
  thresholds; high surprise grants exploration patience and read-only probing,
  while low surprise tightens thresholds toward ``CRYSTALLIZE``.
* :class:`AnalogicalMatcher` compares normalised Schmidt spectra plus AST motif
  ratios, so a strategy learned in one repository can be recalled in another.
* :class:`UserCoAdapter` infers patch granularity, autonomy, and rationale
  density from implicit feedback.

:class:`SynapticDecay` implements the smooth confidence fade used by compiled
reflexes. The module uses only NumPy, which is already a runtime dependency;
JAX/THRML and any LLM client remain optional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AdaptiveThresholds",
    "AnalogicalMatcher",
    "AnalogyMatch",
    "AutonomyProfile",
    "HomeostaticRegulator",
    "StructuralSignature",
    "SynapticDecay",
    "UserCoAdapter",
]


# ---------------------------------------------------------------------------
# 1. Homeostatic / Allostatic Regulator (Dynamic Governor Plasticity)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdaptiveThresholds:
    """Live governor thresholds produced from surprise and familiarity."""

    eta_min: float
    eta_high: float
    tau: float
    chi_max: int
    should_probe_first: bool
    novelty_score: float
    exploration_patience: int = 0
    crystallization_bias: float = 0.0
    prefer_crystallize: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "eta_min": self.eta_min,
            "eta_high": self.eta_high,
            "tau": self.tau,
            "chi_max": self.chi_max,
            "should_probe_first": self.should_probe_first,
            "novelty_score": self.novelty_score,
            "exploration_patience": self.exploration_patience,
            "crystallization_bias": self.crystallization_bias,
            "prefer_crystallize": self.prefer_crystallize,
        }


@dataclass
class HomeostaticRegulator:
    """Adjust governor patience and entanglement tolerance from live surprise."""

    base_eta_min: float = 0.25
    base_eta_high: float = 1.20
    base_tau: float = 0.15
    base_chi_max: int = 4
    ema_alpha: float = 0.30
    surprise_ema: float = 0.0
    _history: list[float] = field(default_factory=list)
    last_thresholds: AdaptiveThresholds | None = None

    def observe_outcome(self, predicted_drop: float, verified_drop: float) -> float:
        """Update running surprise from the gap between expected and verified drop."""
        surprise = abs(float(predicted_drop) - float(verified_drop))
        alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.surprise_ema = alpha * surprise + (1.0 - alpha) * self.surprise_ema
        self._history.append(surprise)
        if len(self._history) > 128:
            del self._history[:-128]
        return self.surprise_ema

    def modulate(self, analogy_similarity: float, risk_level: float = 0.5) -> AdaptiveThresholds:
        """Compute live thresholds from novelty, surprise, and stakes."""
        familiarity = float(np.clip(analogy_similarity, 0.0, 1.0))
        stakes = float(np.clip(risk_level, 0.0, 1.0))
        novelty = float(np.clip(1.0 - familiarity + 0.35 * self.surprise_ema, 0.0, 1.5))

        eta_min = float(np.clip(self.base_eta_min * (1.15 - 0.45 * novelty + 0.25 * stakes), 0.05, 0.80))
        eta_high = float(max(eta_min * 2.2, self.base_eta_high * (1.0 - 0.15 * novelty)))
        tau = float(np.clip(self.base_tau * (1.0 + 0.30 * novelty - 0.20 * stakes), 0.06, 0.35))
        chi_max = max(2, round(self.base_chi_max + (2 if novelty > 0.75 else 0)))
        should_probe = bool(novelty > 0.65 or (stakes > 0.75 and familiarity < 0.70))

        # A stable, familiar task tightens the crystallization envelope instead
        # of leaving the governor in a permanently exploratory mode.
        if novelty < 0.35:
            closeness = novelty / 0.35
            eta_min = float(np.clip(self.base_eta_min * (1.0 + 0.15 * (1.0 - closeness)), 0.05, 0.80))
            eta_high = float(max(eta_min * 2.2, self.base_eta_high * (0.90 + 0.10 * closeness)))
            tau = float(np.clip(self.base_tau * (0.85 + 0.15 * closeness), 0.06, 0.35))
            chi_max = max(2, round(self.base_chi_max))
        exploration_patience = int(np.clip(round(novelty * 3.0), 0, 3))
        crystallization_bias = round(float(np.clip(1.0 - novelty, 0.0, 1.0)), 4)
        prefer_crystallize = bool(novelty < 0.35 and not should_probe)

        thresholds = AdaptiveThresholds(
            eta_min=round(eta_min, 4),
            eta_high=round(eta_high, 4),
            tau=round(tau, 4),
            chi_max=chi_max,
            should_probe_first=should_probe,
            novelty_score=round(novelty, 4),
            exploration_patience=exploration_patience,
            crystallization_bias=crystallization_bias,
            prefer_crystallize=prefer_crystallize,
        )
        self.last_thresholds = thresholds
        return thresholds

    @property
    def history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def snapshot(self) -> dict[str, Any]:
        return {
            "surprise_ema": self.surprise_ema,
            "history": list(self._history),
            "last_thresholds": self.last_thresholds.as_dict() if self.last_thresholds else None,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.surprise_ema = float(snapshot.get("surprise_ema", 0.0) or 0.0)
        self._history = [float(item) for item in snapshot.get("history", ())]
        raw = snapshot.get("last_thresholds")
        self.last_thresholds = AdaptiveThresholds(**dict(raw)) if isinstance(raw, Mapping) else None

    def reset(self) -> None:
        self.surprise_ema = 0.0
        self._history.clear()
        self.last_thresholds = None


# ---------------------------------------------------------------------------
# 2. Scale-Invariant Analogical Transfer (Cross-Domain Pattern Matching)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructuralSignature:
    """Coordinate-free topological fingerprint of a problem state."""

    spectrum: tuple[float, ...]
    motif_ratios: tuple[float, ...]

    def to_vector(self) -> np.ndarray:
        return np.asarray([*self.spectrum, *self.motif_ratios], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectrum": list(self.spectrum),
            "motif_ratios": list(self.motif_ratios),
        }

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> StructuralSignature:
        return cls(
            spectrum=tuple(float(item) for item in data.get("spectrum", ())),
            motif_ratios=tuple(float(item) for item in data.get("motif_ratios", ())),
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
    """Find structurally similar past solutions across unrelated codebases."""

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


# ---------------------------------------------------------------------------
# 3. User Co-Adaptation Model (Implicit Feedback & Communication Plasticity)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AutonomyProfile:
    """Live communication/autonomy preferences inferred from implicit feedback."""

    patch_granularity: str  # "surgical" | "balanced" | "bold"
    probe_before_edit: bool
    rationale_style: str  # "terse" | "detailed"
    accept_rate: float
    epistemic_probe: bool = False
    multi_file_crystallization: bool = False
    manual_overrides: int = 0
    avg_prompt_words: float = 18.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_granularity": self.patch_granularity,
            "probe_before_edit": self.probe_before_edit,
            "rationale_style": self.rationale_style,
            "accept_rate": self.accept_rate,
            "epistemic_probe": self.epistemic_probe,
            "multi_file_crystallization": self.multi_file_crystallization,
            "manual_overrides": self.manual_overrides,
            "avg_prompt_words": round(self.avg_prompt_words, 3),
        }


class UserCoAdapter:
    """Adapts agent autonomy, patch size, and explanation style to a user."""

    def __init__(self) -> None:
        self.accepted_turns: int = 0
        self.rejected_or_reverted_turns: int = 0
        self.manual_overrides: int = 0
        self._prompt_lengths: list[int] = []

    def record_turn(
        self,
        *,
        prompt_text: str,
        edits_accepted: bool,
        was_reverted: bool = False,
        user_overrode: bool = False,
        rejected_edits: int = 0,
    ) -> None:
        words = len(str(prompt_text).split())
        if words > 0:
            self._prompt_lengths.append(words)
            if len(self._prompt_lengths) > 64:
                del self._prompt_lengths[:-64]

        if user_overrode:
            self.manual_overrides += 1
        if edits_accepted and not was_reverted and rejected_edits <= 0:
            self.accepted_turns += 1
        elif was_reverted or user_overrode or rejected_edits > 0:
            self.rejected_or_reverted_turns += 1

    def profile(self) -> AutonomyProfile:
        total = self.accepted_turns + self.rejected_or_reverted_turns
        accept_rate = (self.accepted_turns / total) if total > 0 else 0.75
        avg_words = float(np.mean(self._prompt_lengths)) if self._prompt_lengths else 18.0

        frequent_reverts = (
            self.rejected_or_reverted_turns >= 2 or self.manual_overrides >= 2 or accept_rate < 0.55
        )
        sustained_acceptance = accept_rate > 0.85 and total >= 3 and self.rejected_or_reverted_turns == 0
        if frequent_reverts:
            granularity = "surgical"
            probe_first = True
            epistemic_probe = True
            multi_file = False
        elif sustained_acceptance:
            granularity = "bold"
            probe_first = False
            epistemic_probe = False
            multi_file = True
        else:
            granularity = "balanced"
            probe_first = accept_rate < 0.70
            epistemic_probe = False
            multi_file = False

        return AutonomyProfile(
            patch_granularity=granularity,
            probe_before_edit=probe_first,
            rationale_style="terse" if avg_words < 14.0 else "detailed",
            accept_rate=round(accept_rate, 3),
            epistemic_probe=epistemic_probe,
            multi_file_crystallization=multi_file,
            manual_overrides=self.manual_overrides,
            avg_prompt_words=avg_words,
        )

    def replay(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild a profile from append-only ``user_feedback`` sink records."""
        for record in records:
            payload: Mapping[str, Any] = (
                record if isinstance(record, Mapping) else getattr(record, "payload", {})
            )
            self.record_turn(
                prompt_text=str(payload.get("prompt_text", "")),
                edits_accepted=bool(payload.get("edits_accepted", False)),
                was_reverted=bool(payload.get("was_reverted", False)),
                user_overrode=bool(payload.get("user_overrode", False)),
                rejected_edits=int(payload.get("rejected_edits", 0) or 0),
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted_turns": self.accepted_turns,
            "rejected_or_reverted_turns": self.rejected_or_reverted_turns,
            "manual_overrides": self.manual_overrides,
            "prompt_lengths": list(self._prompt_lengths),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.accepted_turns = int(snapshot.get("accepted_turns", 0) or 0)
        self.rejected_or_reverted_turns = int(snapshot.get("rejected_or_reverted_turns", 0) or 0)
        self.manual_overrides = int(snapshot.get("manual_overrides", 0) or 0)
        self._prompt_lengths = [int(item) for item in snapshot.get("prompt_lengths", ())]


# ---------------------------------------------------------------------------
# 4. Graceful Unlearning (Synaptic Decay & Fast Weights)
# ---------------------------------------------------------------------------
@dataclass
class SynapticDecay:
    """Time- and surprise-weighted confidence fade for compiled habits."""

    base_half_life: float = 8.0
    ema_alpha: float = 0.30
    surprise_ema: float = 0.0
    fast_weight: float = 1.0
    min_confidence: float = 0.12

    def observe(self, *, ok: bool, surprise: float = 0.0) -> float:
        """A miss or a surprising environment accelerates forgetting."""
        signal = abs(float(surprise)) + (0.0 if ok else 0.5)
        alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.surprise_ema = alpha * signal + (1.0 - alpha) * self.surprise_ema
        target = (1.0 if ok else 0.25) / (1.0 + signal)
        self.fast_weight = float(np.clip((1.0 - alpha) * self.fast_weight + alpha * target, 0.05, 1.0))
        return self.surprise_ema

    def confidence(self, *, support: int, hits: int, misses: int, age_steps: int = 0) -> float:
        """Return a smooth confidence in ``[min_confidence, 1]``."""
        positive = max(0, int(support)) + 2 * max(0, int(hits)) + 1.0
        negative = max(0, int(misses)) + 1.0
        base = min(1.0, positive / (positive + negative))
        half_life = max(1.0, self.base_half_life / (1.0 + self.surprise_ema))
        time_decay = 0.5 ** (max(0, int(age_steps)) / half_life)
        surprise_penalty = 1.0 / (1.0 + self.surprise_ema)
        fast = 0.55 + 0.45 * self.fast_weight
        return round(
            float(np.clip(base * time_decay * surprise_penalty * fast, self.min_confidence, 1.0)),
            4,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_half_life": self.base_half_life,
            "ema_alpha": self.ema_alpha,
            "surprise_ema": self.surprise_ema,
            "fast_weight": self.fast_weight,
            "min_confidence": self.min_confidence,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.base_half_life = float(snapshot.get("base_half_life", self.base_half_life))
        self.ema_alpha = float(snapshot.get("ema_alpha", self.ema_alpha))
        self.surprise_ema = float(snapshot.get("surprise_ema", 0.0) or 0.0)
        self.fast_weight = float(snapshot.get("fast_weight", 1.0) or 1.0)
        self.min_confidence = float(snapshot.get("min_confidence", self.min_confidence))

    def snapshot(self) -> dict[str, Any]:
        return self.as_dict()
