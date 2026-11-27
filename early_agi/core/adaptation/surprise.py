"""One source of truth for surprise EMAs and adaptive thresholds.

Both governor plasticity and compiled-reflex decay use ``_blend`` for their
running surprise signal. Thresholds, crystallization bias, and confidence fade
remain deterministic NumPy-only calculations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AdaptiveThresholds",
    "HomeostaticRegulator",
    "SynapticDecay",
]


def _blend(current: float, target: float, alpha: float) -> float:
    """EMA step shared by all surprise/novelty trackers in this boundary."""
    clipped = float(np.clip(alpha, 0.0, 1.0))
    return clipped * target + (1.0 - clipped) * current


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
        self.surprise_ema = _blend(self.surprise_ema, surprise, self.ema_alpha)
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
        self.surprise_ema = _blend(self.surprise_ema, signal, alpha)
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
