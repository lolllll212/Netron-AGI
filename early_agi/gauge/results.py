"""Structured result of one Gauge VM run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GaugeRunResult"]


@dataclass
class GaugeRunResult:
    status: str
    program: str
    root: str
    run_index: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, str] = field(default_factory=dict)
    entanglement: dict[str, Any] = field(default_factory=dict)
    governor: dict[str, Any] = field(default_factory=dict)
    manifolds: dict[str, Any] = field(default_factory=dict)
    collapses: dict[str, str] = field(default_factory=dict)
    synthesis: list[dict[str, Any]] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    recalled: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    ir: dict[str, Any] = field(default_factory=dict)
    tick: dict[str, Any] = field(default_factory=dict)
    mera: dict[str, Any] = field(default_factory=dict)
    reflexes: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "halted" and not self.violations

    @property
    def synthesis_events(self) -> list[dict[str, Any]]:
        return self.synthesis

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "program": self.program,
            "root": self.root,
            "run_index": self.run_index,
            "trace": self.trace,
            "gates": self.gates,
            "entanglement": self.entanglement,
            "governor": self.governor,
            "manifolds": self.manifolds,
            "collapses": self.collapses,
            "synthesis": self.synthesis,
            "traps": self.traps,
            "recalled": self.recalled,
            "violations": self.violations,
            "ir": self.ir,
            "tick": self.tick,
            "mera": self.mera,
            "reflexes": self.reflexes,
        }
