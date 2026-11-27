"""Report records and small numeric helpers for the A/B benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import FileEdit, PatchResponse
from .fixtures import BugFixture

__all__ = [
    "ArmState",
    "AttemptExecution",
    "AttemptRecord",
    "RetentionPoint",
    "TaskRecord",
    "candidate_name",
    "dedupe_strings",
    "mean",
    "ratio",
]


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt: int
    orchestrator: str
    candidate: str
    tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    eta_cog: float
    verified_drop_bits: float
    gates: Mapping[str, str]
    constraints: tuple[str, ...]
    cost: Mapping[str, Any]
    directive: str
    backend: str
    passed: bool
    rollback: bool
    reflex: bool
    preflight_probe: bool
    violation_count: int
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "orchestrator": self.orchestrator,
            "candidate": self.candidate,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "eta_cog": round(self.eta_cog, 6),
            "verified_drop_bits": round(self.verified_drop_bits, 6),
            "gates": dict(sorted(self.gates.items())),
            "constraints": list(self.constraints),
            "cost": dict(self.cost),
            "directive": self.directive,
            "backend": self.backend,
            "passed": self.passed,
            "rollback": self.rollback,
            "reflex": self.reflex,
            "preflight_probe": self.preflight_probe,
            "violation_count": self.violation_count,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AttemptExecution:
    record: AttemptRecord
    edits: tuple[FileEdit, ...]
    before: Mapping[str, bytes | None]


@dataclass(frozen=True, slots=True)
class RetentionPoint:
    task_index: int
    checked_workspaces: int
    passed_workspaces: int
    historical_gate_pass_rate: float
    candidate_gate_pass_rate: float
    regressions: int
    zero_regression: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "checked_workspaces": self.checked_workspaces,
            "passed_workspaces": self.passed_workspaces,
            "historical_gate_pass_rate": round(self.historical_gate_pass_rate, 6),
            "candidate_gate_pass_rate": round(self.candidate_gate_pass_rate, 6),
            "regressions": self.regressions,
            "zero_regression": self.zero_regression,
        }


@dataclass(slots=True)
class TaskRecord:
    task_index: int
    fixture_id: str
    first_seen: bool
    attempts: list[AttemptRecord]
    reflex_hit: bool
    preflight: dict[str, Any] | None
    retention: RetentionPoint | None = None

    @property
    def discovery_attempt(self) -> int | None:
        return next((item.attempt for item in self.attempts if item.passed), None)

    @property
    def success(self) -> bool:
        return self.discovery_attempt is not None

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.attempts)

    @property
    def rollbacks(self) -> int:
        return sum(1 for item in self.attempts if item.rollback)

    @property
    def eta_cog_curve(self) -> list[float]:
        return [round(item.eta_cog, 6) for item in self.attempts]

    @property
    def eta_cog(self) -> float:
        return self.eta_cog_curve[-1] if self.eta_cog_curve else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "fixture_id": self.fixture_id,
            "first_seen": self.first_seen,
            "success": self.success,
            "discovery_attempt": self.discovery_attempt,
            "attempts": [item.as_dict() for item in self.attempts],
            "tokens": self.tokens,
            "rollbacks": self.rollbacks,
            "eta_cog": self.eta_cog,
            "eta_cog_curve": self.eta_cog_curve,
            "reflex_hit": self.reflex_hit,
            "preflight": dict(self.preflight) if self.preflight is not None else None,
            "retention": self.retention.as_dict() if self.retention is not None else None,
        }


@dataclass(slots=True)
class ArmState:
    name: str
    tasks: list[TaskRecord] = field(default_factory=list)
    solved_workspaces: list[tuple[Path, BugFixture]] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    retention: list[RetentionPoint] = field(default_factory=list)
    reflex_compiles: int = 0
    preflight_probes: int = 0

    @property
    def tokens_spent(self) -> int:
        return sum(task.tokens for task in self.tasks)

    @property
    def rollbacks(self) -> int:
        return sum(task.rollbacks for task in self.tasks)

    @property
    def reflex_hits(self) -> int:
        return sum(1 for task in self.tasks if task.reflex_hit)

    @property
    def solved_count(self) -> int:
        return sum(1 for task in self.tasks if task.success)

    @property
    def eta_cog_mean(self) -> float:
        return mean([task.eta_cog for task in self.tasks if task.success])

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.as_dict() for task in self.tasks],
            "tokens_spent": self.tokens_spent,
            "rollbacks": self.rollbacks,
            "reflex_hits": self.reflex_hits,
            "solved_workspaces": len(self.solved_workspaces),
            "solved_tasks": self.solved_count,
            "eta_cog_mean": round(self.eta_cog_mean, 6),
            "eta_cog_curve": [round(task.eta_cog, 6) for task in self.tasks if task.success],
            "reflex_compiles": self.reflex_compiles,
            "preflight_probes": self.preflight_probes,
        }


def candidate_name(response: PatchResponse) -> str:
    raw = response.raw
    candidate = raw.get("candidate") if isinstance(raw, Mapping) else None
    return str(candidate or "router_candidate")


def dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value)[:240] for value in values if str(value)))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)
