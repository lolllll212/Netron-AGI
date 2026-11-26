"""The eta_cog thermodynamic governor with a real rolling stall window."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .entropy_sensor import EntanglementReport

__all__ = ["CognitiveStepCost", "Directive", "ThermodynamicGovernor"]


class Directive(str, Enum):
    CRYSTALLIZE = "CRYSTALLIZE"
    SHIFT_COORDINATE = "SHIFT_COORDINATE"
    RECALL_SINK = "RECALL_SINK"


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

        rolling = self.rolling_eta
        dragging = rolling is not None and rolling <= self.eta_min
        regressed_past_floor = verified_entropy_drop < 0.0 and (eta_cog <= self.eta_min or dragging)
        stalled = (
            verified_entropy_drop <= 0.0 and self.recall_streak >= self.stall_ticks
        ) or regressed_past_floor

        if stalled:
            directive = Directive.RECALL_SINK
            self.shift_streak = 0
        elif entanglement.is_volume_law or eta_cog < self.eta_high or dragging:
            directive = Directive.SHIFT_COORDINATE
            self.shift_streak += 1
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
        }

    def reset(self) -> None:
        self.history.clear()
        self.drops.clear()
        self.directives.clear()
        self.shift_streak = 0
        self.recall_streak = 0
