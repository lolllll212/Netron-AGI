"""The eta_cog thermodynamic governor with a real rolling stall window.

Plasticity is opt-in: :class:`ThermodynamicGovernor` keeps its historical fixed
thresholds until :meth:`enable_plasticity` is called (by a plastic kernel, the
Gauge ``plastic govern`` declaration, or an explicit caller). Once enabled, the
homeostatic regulator turns prediction surprise and analogical familiarity into
live thresholds and can request a read-only ``EPISTEMIC_PROBE`` before the first
mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from .adaptation.surprise import AdaptiveThresholds, HomeostaticRegulator

if TYPE_CHECKING:
    from .entropy_sensor import EntanglementReport

__all__ = ["CognitiveStepCost", "Directive", "ThermodynamicGovernor"]


class Directive(str, Enum):
    CRYSTALLIZE = "CRYSTALLIZE"
    SHIFT_COORDINATE = "SHIFT_COORDINATE"
    RECALL_SINK = "RECALL_SINK"
    EPISTEMIC_PROBE = "EPISTEMIC_PROBE"


@dataclass
class CognitiveStepCost:
    discarded_tokens: int = 0
    tool_cost: float = 0.0
    pruned_branches: int = 0
    latency_ms: float = 0.0

    def weighted(
        self,
        *,
        w_tok: float = 0.001,
        w_tool: float = 1.0,
        w_branch: float = 2.0,
        w_latency: float = 0.0001,
    ) -> float:
        return (
            w_tok * max(0, self.discarded_tokens)
            + w_tool * max(0.0, self.tool_cost)
            + w_branch * max(0, self.pruned_branches)
            + w_latency * max(0.0, self.latency_ms)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "discarded_tokens": self.discarded_tokens,
            "tool_cost": self.tool_cost,
            "pruned_branches": self.pruned_branches,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class ThermodynamicGovernor:
    eta_high: float = 1.2
    eta_min: float = 0.15
    w_tok: float = 0.001
    w_tool: float = 1.0
    w_branch: float = 2.0
    w_latency: float = 0.0001
    window: int = 3
    stall_ticks: int = 2
    max_history: int = 256
    history: list[float] = field(default_factory=list)
    drops: list[float] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    shift_streak: int = 0
    recall_streak: int = 0
    regulator: HomeostaticRegulator = field(default_factory=HomeostaticRegulator)
    plasticity_enabled: bool = False
    probe_pending: bool = False
    exploration_budget: int = 0
    last_thresholds: AdaptiveThresholds | None = None

    def enable_plasticity(
        self,
        *,
        base_eta_min: float | None = None,
        base_eta_high: float | None = None,
        base_tau: float = 0.15,
        base_chi_max: int = 4,
        ema_alpha: float = 0.30,
    ) -> None:
        """Activate homeostatic modulation using current thresholds as priors."""
        self.regulator.base_eta_min = float(base_eta_min if base_eta_min is not None else self.eta_min)
        self.regulator.base_eta_high = float(base_eta_high if base_eta_high is not None else self.eta_high)
        self.regulator.base_tau = float(base_tau)
        self.regulator.base_chi_max = int(base_chi_max)
        self.regulator.ema_alpha = float(ema_alpha)
        self.plasticity_enabled = True

    def observe_outcome(self, predicted_drop: float, verified_drop: float) -> float:
        """Feed a prediction/verification gap into the surprise EMA."""
        return self.regulator.observe_outcome(predicted_drop, verified_drop)

    def modulate(
        self,
        analogy_similarity: float = 0.0,
        risk_level: float = 0.5,
        *,
        predicted_drop: float | None = None,
        verified_drop: float | None = None,
    ) -> AdaptiveThresholds:
        """Recompute live thresholds; no-op when plasticity is disabled."""
        if not self.plasticity_enabled:
            self.exploration_budget = 0
            return AdaptiveThresholds(
                eta_min=round(self.eta_min, 4),
                eta_high=round(self.eta_high, 4),
                tau=0.15,
                chi_max=4,
                should_probe_first=False,
                novelty_score=0.0,
            )
        if verified_drop is not None:
            predicted = predicted_drop
            if predicted is None:
                predicted = self.drops[-1] if self.drops else verified_drop
            self.observe_outcome(predicted, verified_drop)
        thresholds = self.regulator.modulate(analogy_similarity, risk_level)
        self.last_thresholds = thresholds
        self.eta_min = thresholds.eta_min
        self.eta_high = thresholds.eta_high
        self.probe_pending = thresholds.should_probe_first
        self.exploration_budget = max(0, int(thresholds.exploration_patience))
        return thresholds

    def evaluate(
        self,
        verified_entropy_drop: float,
        cost: CognitiveStepCost,
        entanglement: EntanglementReport,
    ) -> tuple[float, Directive]:
        w_sink = cost.weighted(
            w_tok=self.w_tok,
            w_tool=self.w_tool,
            w_branch=self.w_branch,
            w_latency=self.w_latency,
        )
        eta_cog = float(verified_entropy_drop) / (w_sink + 1e-6)

        self.history.append(eta_cog)
        self.drops.append(float(verified_entropy_drop))
        if verified_entropy_drop <= 0.0:
            self.recall_streak += 1
        else:
            self.recall_streak = 0

        # A high-novelty first tick asks for read-only measurements before the
        # kernel is allowed to crystallise or mutate the workspace.
        if self.probe_pending and len(self.directives) == 0:
            self.probe_pending = False
            self.shift_streak = 0
            directive = Directive.EPISTEMIC_PROBE
            self.directives.append(directive)
            self._trim_history()
            return eta_cog, directive

        rolling = self.rolling_eta
        dragging = rolling is not None and rolling <= self.eta_min
        regressed_past_floor = verified_entropy_drop < 0.0 and (eta_cog <= self.eta_min or dragging)
        stalled = (
            verified_entropy_drop <= 0.0 and self.recall_streak >= self.stall_ticks
        ) or regressed_past_floor

        prefer_crystallize = bool(
            self.plasticity_enabled
            and self.last_thresholds is not None
            and self.last_thresholds.prefer_crystallize
        )
        if stalled:
            if self.exploration_budget > 0 and self.plasticity_enabled:
                directive = Directive.SHIFT_COORDINATE
                self.exploration_budget -= 1
                self.shift_streak += 1
            else:
                directive = Directive.RECALL_SINK
                self.shift_streak = 0
        elif entanglement.is_volume_law:
            directive = Directive.SHIFT_COORDINATE
            self.shift_streak += 1
            self.exploration_budget = max(0, self.exploration_budget - 1)
        elif prefer_crystallize and not dragging and eta_cog >= self.eta_min:
            directive = Directive.CRYSTALLIZE
            self.shift_streak = 0
            self.exploration_budget = 0
        elif eta_cog < self.eta_high or dragging:
            directive = Directive.SHIFT_COORDINATE
            self.shift_streak += 1
            self.exploration_budget = max(0, self.exploration_budget - 1)
        else:
            directive = Directive.CRYSTALLIZE
            self.shift_streak = 0

        self.directives.append(directive)
        self._trim_history()
        return eta_cog, directive

    def _trim_history(self) -> None:
        if self.max_history <= 0 or len(self.history) <= self.max_history:
            return
        keep = self.max_history
        self.history = self.history[-keep:]
        self.drops = self.drops[-keep:]
        self.directives = self.directives[-keep:]

    @property
    def last_eta(self) -> float | None:
        return self.history[-1] if self.history else None

    @property
    def rolling_eta(self) -> float | None:
        tail = self.history[-max(1, self.window) :]
        return sum(tail) / len(tail) if tail else None

    @property
    def drag(self) -> bool:
        rolling = self.rolling_eta
        return rolling is not None and rolling <= self.eta_min

    @property
    def synthesis_due(self) -> bool:
        return self.shift_streak >= 2

    def consume_synthesis(self) -> None:
        """Phase-4 hook: acknowledge that synthesis has been attempted."""
        self.shift_streak = 0

    def clear_probe(self) -> None:
        """Acknowledge that read-only probing has already been performed."""
        self.probe_pending = False
        if self.last_thresholds is not None:
            self.last_thresholds = replace(self.last_thresholds, should_probe_first=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "eta": self.last_eta,
            "rolling_eta": self.rolling_eta,
            "eta_min": self.eta_min,
            "eta_high": self.eta_high,
            "window": self.window,
            "drag": self.drag,
            "ticks": len(self.history),
            "shift_streak": self.shift_streak,
            "recall_streak": self.recall_streak,
            "synthesis_due": self.synthesis_due,
            "last_directive": self.directives[-1].value if self.directives else None,
            "plasticity_enabled": self.plasticity_enabled,
            "probe_pending": self.probe_pending,
            "exploration_budget": self.exploration_budget,
            "surprise_ema": round(self.regulator.surprise_ema, 6),
            "last_thresholds": self.last_thresholds.as_dict() if self.last_thresholds else None,
        }

    def plasticity_snapshot(self) -> dict[str, Any]:
        return {
            "plasticity_enabled": self.plasticity_enabled,
            "probe_pending": self.probe_pending,
            "exploration_budget": self.exploration_budget,
            "eta_min": self.eta_min,
            "eta_high": self.eta_high,
            "last_thresholds": self.last_thresholds.as_dict() if self.last_thresholds else None,
            "regulator": self.regulator.snapshot(),
        }

    def restore_plasticity(self, snapshot: Mapping[str, Any]) -> None:
        self.plasticity_enabled = bool(snapshot.get("plasticity_enabled", False))
        self.probe_pending = bool(snapshot.get("probe_pending", False))
        self.exploration_budget = int(snapshot.get("exploration_budget", 0) or 0)
        self.eta_min = float(snapshot.get("eta_min", self.eta_min))
        self.eta_high = float(snapshot.get("eta_high", self.eta_high))
        raw = snapshot.get("last_thresholds")
        self.last_thresholds = AdaptiveThresholds(**dict(raw)) if isinstance(raw, Mapping) else None
        regulator = snapshot.get("regulator")
        if isinstance(regulator, Mapping):
            self.regulator.restore(regulator)

    def reset(self) -> None:
        self.history.clear()
        self.drops.clear()
        self.directives.clear()
        self.shift_streak = 0
        self.recall_streak = 0
        self.probe_pending = False
        self.exploration_budget = 0
        self.last_thresholds = None
        self.regulator.reset()
