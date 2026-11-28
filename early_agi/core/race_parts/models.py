from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...bench.contracts import FileEdit
DEFAULT_GATES = ("syntax_valid", "tests_passing")
DEFAULT_BRANCHES = ("module_cut", "dependency_cut", "test_locking_order", "inverse_replay")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Picklable live-model readiness config; never carries a client instance."""

    name: str = "offline_fixture"
    endpoint: str = ""
    model: str = ""
    seed: int = 0
    max_tokens_per_branch: int = 4096


@dataclass(frozen=True, slots=True)
class RaceFixtureSpec:
    """Picklable mirror of a benchmark fixture, including all candidate patches."""

    id: str
    task: str
    files: tuple[tuple[str, str], ...]
    editable_paths: tuple[str, ...]
    test_targets: tuple[str, ...]
    candidates: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...]
    probe_hint: str
    fix_index: int = 1

    @classmethod
    def from_fixture(cls, fixture: Any) -> RaceFixtureSpec:
        candidates = tuple(
            (
                str(candidate.name),
                tuple((str(edit.path), str(edit.content)) for edit in candidate.edits),
                str(candidate.rationale),
            )
            for candidate in fixture.candidates
        )
        return cls(
            id=str(fixture.id),
            task=str(fixture.task),
            files=tuple((str(path), str(content)) for path, content in fixture.files),
            editable_paths=tuple(str(path) for path in fixture.editable_paths),
            test_targets=tuple(str(path) for path in fixture.test_targets),
            candidates=candidates,
            probe_hint=str(fixture.probe_hint),
            fix_index=int(fixture.fix_index),
        )

    def to_bug_fixture(self) -> Any:
        from ...bench.contracts import FileEdit as _FileEdit
        from ...bench.fixtures import BugFixture, PatchCandidate

        return BugFixture(
            id=self.id,
            task=self.task,
            files=self.files,
            candidates=tuple(
                PatchCandidate(
                    name,
                    tuple(_FileEdit(path, content) for path, content in edits),
                    rationale,
                )
                for name, edits, rationale in self.candidates
            ),
            probe_hint=self.probe_hint,
            fix_index=self.fix_index,
        )

    @property
    def file_map(self) -> dict[str, str]:
        return dict(self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "files": [path for path, _ in self.files],
            "editable_paths": list(self.editable_paths),
            "test_targets": list(self.test_targets),
            "candidates": [name for name, _edits, _rationale in self.candidates],
            "fix_index": self.fix_index,
        }


@dataclass(frozen=True, slots=True)
class RaceConfig:
    """Configuration for one repeatable, strictly offline branch race."""

    fixture: RaceFixtureSpec
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    gates: tuple[str, ...] = DEFAULT_GATES
    trigger: str = "frustration"
    router_name: str = "offline_fixture"
    endpoint: str = ""
    model: str = ""
    seed: int = 0
    max_tokens_per_branch: int = 4096
    max_workers: int = 4
    executor: str = "process"
    timeout_s: float = 60.0
    user_id: str | None = None
    agent_id: str | None = None
    mera_path: str | None = None


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """The entire worker payload: primitives and nested dataclasses only."""

    branch_id: str
    strategy: str
    transform_basis: str
    branch_index: int
    candidate_index: int
    run_id: str
    parent_hash: str
    trigger: str
    source_root: str
    sandbox_parent: str
    sink_path: str
    user_id: str | None
    agent_id: str | None
    fixture: RaceFixtureSpec
    gates: tuple[str, ...]
    router: RouterConfig
    timeout_s: float
    lease_ttl_s: float
    overhead_tokens: int
    tool_overhead: float


@dataclass(frozen=True, slots=True)
class BranchResult:
    branch_id: str
    strategy: str
    transform_basis: str
    branch_index: int
    candidate_index: int
    candidate: str
    passed: bool
    eta_cog: float
    verified_drop_bits: float
    tokens: int
    uncapped_tokens: int
    cap_exceeded: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    directive: str
    gates: dict[str, str]
    constraints: tuple[str, ...]
    cost: dict[str, Any]
    edits: tuple[FileEdit, ...]
    candidate_hash: str
    pruned_hash: str
    error: str
    sandbox_mode: str
    scope: dict[str, str]
    lease_id: str
    worker_pid: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "strategy": self.strategy,
            "transform_basis": self.transform_basis,
            "branch_index": self.branch_index,
            "candidate_index": self.candidate_index,
            "candidate": self.candidate,
            "passed": self.passed,
            "eta_cog": round(self.eta_cog, 6),
            "verified_drop_bits": round(self.verified_drop_bits, 6),
            "tokens": self.tokens,
            "uncapped_tokens": self.uncapped_tokens,
            "cap_exceeded": self.cap_exceeded,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "directive": self.directive,
            "gates": dict(sorted(self.gates.items())),
            "constraints": list(self.constraints),
            "cost": dict(self.cost),
            "edits": [edit.as_dict() for edit in self.edits],
            "candidate_hash": self.candidate_hash,
            "pruned_hash": self.pruned_hash,
            "error": self.error,
            "sandbox_mode": self.sandbox_mode,
            "scope": dict(sorted(self.scope.items())),
        }


@dataclass(frozen=True, slots=True)
class BranchRaceResult:
    run_id: str
    parent_hash: str
    trigger: str
    executor: str
    branches: tuple[BranchResult, ...]
    winner: BranchResult | None
    winner_hash: str
    pruned_hashes: tuple[str, ...]
    lineage_edges: int
    total_tokens: int
    router_name: str
    endpoint: str
    model: str
    seed: int
    max_tokens_per_branch: int
    sandbox_modes: tuple[str, ...]
    mera_path: str

    @property
    def ok(self) -> bool:
        return self.winner is not None

    @property
    def winner_edits(self) -> tuple[FileEdit, ...]:
        return self.winner.edits if self.winner is not None else ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "offline": True,
            "network_calls": 0,
            "run_id": self.run_id,
            "parent_hash": self.parent_hash,
            "trigger": self.trigger,
            "executor": self.executor,
            "winner": self.winner.as_dict() if self.winner is not None else None,
            "winner_branch_id": self.winner.branch_id if self.winner is not None else "",
            "winner_hash": self.winner_hash,
            "pruned_branches": len(self.pruned_hashes),
            "pruned_hashes": list(self.pruned_hashes),
            "lineage_edges": self.lineage_edges,
            "branch_count": len(self.branches),
            "total_tokens": self.total_tokens,
            "tokens_by_branch": {
                result.branch_id: result.tokens
                for result in sorted(self.branches, key=lambda item: item.branch_id)
            },
            "router_name": self.router_name,
            "endpoint": self.endpoint,
            "model": self.model,
            "seed": self.seed,
            "max_tokens_per_branch": self.max_tokens_per_branch,
            "sandbox_modes": list(self.sandbox_modes),
            "branches": [result.as_dict() for result in self.branches],
        }
