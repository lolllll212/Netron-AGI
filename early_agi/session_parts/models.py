"""Turn result record for the transactional session loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["TurnOutcome"]


@dataclass(frozen=True)
class TurnOutcome:
    turn: int
    status: str
    directive: str
    eta_cog: float
    verified_drop_bits: float
    spectral_entropy_bits: float
    truncation_error: float
    volume_law: bool
    applied: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    rationale: str = ""
    hash: str = ""
    writes_locked: bool = False
    reflex: str = ""
    rejected: tuple[Mapping[str, Any], ...] = ()
    cost: Mapping[str, Any] | None = None
    user_profile: Mapping[str, Any] | None = None
    analogy: Mapping[str, Any] | None = None
    plasticity: Mapping[str, Any] | None = None
    probe: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "turn": self.turn,
            "status": self.status,
            "directive": self.directive,
            "eta_cog": round(self.eta_cog, 3),
            "verified_drop_bits": round(self.verified_drop_bits, 3),
            "spectral_entropy_bits": round(self.spectral_entropy_bits, 3),
            "truncation_error": round(self.truncation_error, 3),
            "volume_law": self.volume_law,
            "applied": list(self.applied),
            "reverted": list(self.reverted),
            "rationale": self.rationale,
            "hash": self.hash[:10],
            "writes_locked": self.writes_locked,
            "reflex": self.reflex,
            "rejected": list(self.rejected),
        }
        if self.cost is not None:
            out["cost"] = dict(self.cost)
        if self.user_profile is not None:
            out["user_profile"] = dict(self.user_profile)
        if self.analogy is not None:
            out["analogy"] = dict(self.analogy)
        if self.plasticity is not None:
            out["plasticity"] = dict(self.plasticity)
        if self.probe is not None:
            out["probe"] = dict(self.probe)
        return out
