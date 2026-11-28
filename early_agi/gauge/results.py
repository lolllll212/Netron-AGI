"""Structured result of one Gauge VM run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

__all__ = ["GaugeRaceOutcome", "GaugeRunResult"]


class GaugeRaceOutcome(TypedDict):
    """Stable public subset of a completed offline race's audit outcome."""

    ok: bool
    run_id: str
    trigger: str
    executor: str
    winner_branch_id: str
    winner_hash: str
    branch_count: int
    pruned_branches: int
    total_tokens: int
    lineage_edges: int
    offline: bool
    network_calls: int


@dataclass
class GaugeRunResult:
    status: str
    program: str
    root: str
    backend: str = "classical_svd"
    backend_impl: str = "classical_svd"
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
    plasticity: dict[str, Any] = field(default_factory=dict)

    races: list[GaugeRaceOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "halted" and not self.violations

    @property
    def synthesis_events(self) -> list[dict[str, Any]]:
        return self.synthesis

    def as_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "ok": self.ok,
            "program": self.program,
            "root": self.root,
            "backend": self.backend,
            "backend_impl": self.backend_impl,
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
            "plasticity": self.plasticity,
        }
        # Keep legacy no-race JSON byte-compatible with the two-arm benchmark.
        if self.races:
            result["races"] = self.races
        return result
