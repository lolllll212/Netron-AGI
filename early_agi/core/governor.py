"""The eta_cog thermodynamic governor.

``eta_cog`` is verified entropy collapse per unit of irreversibly spent
resource::

    eta_cog = verified_entropy_drop_bits / (w_tok * tokens + w_tool * tools + w_branch * branches)

The governor turns that single scalar (plus the entanglement regime) into one of
three binary directives. The ordering is a priority ladder, not a vote:

1. RECALL_SINK — two consecutive non-positive drops: the branch is dead, replay
   the pruned tail from the cold sink instead of burning more budget.
2. SHIFT_COORDINATE — volume-law coupling, or positive-but-dragging eta: local
   effort is buying nothing, so reframe the problem / invent a coordinate.
3. CRYSTALLIZE — cheap, area-law, high-yield descent: commit and move on.

Note the comparison in (2) is ``eta < eta_high``. The alternative written as
``eta <= eta_high`` inverted the ladder in the original sketch: it sent both
high-eta and low-eta ticks to SHIFT_COORDINATE and made CRYSTALLIZE unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .entropy_sensor import EntanglementReport

__all__ = ["CognitiveStepCost", "Directive", "ThermodynamicGovernor"]


class Directive(str, Enum):
    CRYSTALLIZE = "CRYSTALLIZE"  # high eta_cog: fast executor converges to a fixed point
    SHIFT_COORDINATE = "SHIFT_COORDINATE"  # volume law or drag: halt search, reframe / invent
    RECALL_SINK = "RECALL_SINK"  # stalled/negative eta_cog: recover pruned branch and revise


@dataclass
class CognitiveStepCost:
    """Irreversible resources burned by a single tick."""

    discarded_tokens: int = 0
    tool_cost: float = 0.0
    pruned_branches: int = 0

    def weighted(
        self,
        *,
        w_tok: float = 0.001,
        w_tool: float = 1.0,
        w_branch: float = 2.0,
    ) -> float:
        return (
            w_tok * max(0, self.discarded_tokens)
            + w_tool * max(0.0, self.tool_cost)
            + w_branch * max(0, self.pruned_branches)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "discarded_tokens": self.discarded_tokens,
            "tool_cost": self.tool_cost,
            "pruned_branches": self.pruned_branches,
        }


@dataclass
class ThermodynamicGovernor:
    eta_high: float = 1.2  # crystallise at or above this efficiency
    eta_min: float = 0.15  # stall floor for the rolling mean
    w_tok: float = 0.001
    w_tool: float = 1.0
    w_branch: float = 2.0
    window: int = 3  # rolling window for the stall test
    stall_ticks: int = 2  # consecutive non-positive drops that trigger recall
    history: list[float] = field(default_factory=list)
    drops: list[float] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    shift_streak: int = 0  # consecutive SHIFT_COORDINATE; >=2 is the Pillar-4 trigger
    recall_streak: int = 0

    def evaluate(
        self,
        verified_entropy_drop: float,
        cost: CognitiveStepCost,
        entanglement: EntanglementReport,
    ) -> tuple[float, Directive]:
        w_sink = cost.weighted(w_tok=self.w_tok, w_tool=self.w_tool, w_branch=self.w_branch)
        eta_cog = float(verified_entropy_drop) / (w_sink + 1e-6)

        self.history.append(eta_cog)
        self.drops.append(float(verified_entropy_drop))

        if verified_entropy_drop <= 0.0:
            self.recall_streak += 1
        else:
            self.recall_streak = 0

        # Two ways to conclude the branch is dead: a *regression* that already
        # dropped below the efficiency floor (severe, act immediately), or the
        # configured number of consecutive non-positive ticks. Merely having a
        # small rolling eta must not bypass stall_ticks, or the counter is moot.
        regressed_past_floor = verified_entropy_drop < 0.0 and eta_cog <= self.eta_min
        stalled = (
            verified_entropy_drop <= 0.0 and self.recall_streak >= self.stall_ticks
        ) or regressed_past_floor

        if stalled:
            directive = Directive.RECALL_SINK
            self.shift_streak = 0
        elif entanglement.is_volume_law or eta_cog < self.eta_high:
            directive = Directive.SHIFT_COORDINATE
            self.shift_streak += 1
        else:
            directive = Directive.CRYSTALLIZE
            self.shift_streak = 0

        self.directives.append(directive)
        return eta_cog, directive

    # -- introspection ------------------------------------------------------
    @property
    def last_eta(self) -> float | None:
        return self.history[-1] if self.history else None

    @property
    def rolling_eta(self) -> float | None:
        tail = self.history[-max(1, self.window) :]
        return sum(tail) / len(tail) if tail else None

    @property
    def synthesis_due(self) -> bool:
        """Pillar 4 trigger: two consecutive coordinate shifts."""
        return self.shift_streak >= 2

    def snapshot(self) -> dict[str, Any]:
        return {
            "eta": self.last_eta,
            "rolling_eta": self.rolling_eta,
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
