"""Offline counterfactual branch racing with WAL-scoped single-winner commits.

The engine forks 2-4 isolated sandboxes when a ``volume_law`` or ``frustration``
trap fires. Workers construct their own scripted/mocked router from picklable
configuration; no client instance crosses a process boundary. All candidates
share one file-backed WAL :class:`ColdSink`, open optimistic parent-hash leases,
and only the green DarkStateGate branch with the highest ``eta_cog`` crystallises.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...memory.mera_graph import MERGED, SUPERSEDED, MeraGraph
from ..cold_sink import ColdSink, SinkError, SinkScope
from ..entropy_sensor import EntanglementReport
from ..governor import CognitiveStepCost, ThermodynamicGovernor

if TYPE_CHECKING:
    from ...bench.contracts import PatchRequest, PatchResponse

__all__ = [
    "BranchRaceResult",
    "BranchResult",
    "BranchSpec",
    "MockedOpenAICompatibleRouter",
    "RaceConfig",
    "RaceFixtureSpec",
    "RouterConfig",
    "race_fixture_spec",
    "run_branch_race",
]

from .models import BranchRaceResult, BranchResult, BranchSpec, RaceConfig, RaceFixtureSpec, RouterConfig
from .sandbox import _cleanup_sandbox, _materialise_sandbox, _safe_name

RACE_TRIGGERS = frozenset({"volume_law", "frustration"})
_TRANSFORM_BASES = ("module_cut", "dependency_cut", "test_locking_order", "inverse_replay")


class MockedOpenAICompatibleRouter:
    """Offline mocked transport that exercises the OpenAI-compatible request schema."""

    name = "mocked_openai_compatible"
    offline = True

    def __init__(
        self,
        fixture: Any,
        *,
        endpoint: str = "",
        model: str = "",
        seed: int = 0,
        max_tokens: int | None = None,
    ) -> None:
        self._fixture = fixture
        self.endpoint = str(endpoint)
        self.model = str(model)
        self.seed = int(seed)
        self.max_tokens = None if max_tokens is None else max(0, int(max_tokens))

    def propose(self, request: PatchRequest) -> PatchResponse:
        from ...bench.contracts import PatchResponse, validated_edits

        index = min(max(1, int(request.attempt)) - 1, len(self._fixture.candidates) - 1)
        candidate = self._fixture.candidates[index]
        edits = validated_edits(candidate.edits, request.allowed_paths)
        messages = request.messages()  # no network; verifies the live request shape
        prompt_chars = sum(len(message["content"]) for message in messages)
        completion_chars = sum(len(edit.path) + len(edit.content) for edit in edits)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, completion_chars // 4)
        uncapped_tokens = prompt_tokens + completion_tokens
        raw: dict[str, Any] = {
            "candidate": candidate.name,
            "model": self.model,
            "endpoint": self.endpoint,
            "seed": self.seed,
            "transport": "mocked",
            "offline": True,
            "network_calls": 0,
        }
        if self.max_tokens is not None:
            cap = self.max_tokens
            raw["token_cap"] = cap
            raw["uncapped_tokens"] = uncapped_tokens
            if uncapped_tokens > cap:
                prompt_tokens = min(prompt_tokens, cap // 2)
                completion_tokens = max(0, cap - prompt_tokens)
        latency_ms = round(1.0 + 0.01 * prompt_tokens + 0.02 * completion_tokens, 3)
        return PatchResponse(
            edits=edits,
            rationale=f"{candidate.rationale}; mocked seeded transport",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw=raw,
        )


def _branch_router(spec: BranchSpec, fixture: Any) -> Any:
    from ...bench.routers import OfflineFixtureRouter

    if spec.router.name in {"mocked_live", "mocked_openai_compatible"}:
        return MockedOpenAICompatibleRouter(
            fixture,
            endpoint=spec.router.endpoint,
            model=spec.router.model,
            seed=spec.router.seed,
            max_tokens=spec.router.max_tokens_per_branch,
        )
    return OfflineFixtureRouter((fixture,), max_tokens=spec.router.max_tokens_per_branch)


def _branch_request(spec: BranchSpec, fixture: Any) -> PatchRequest:
    from ...bench.contracts import PatchRequest

    return PatchRequest(
        fixture_id=fixture.id,
        task=fixture.task,
        attempt=spec.candidate_index + 1,
        orchestrator=f"race_{spec.strategy}",
        files=fixture.file_map,
        test_targets=fixture.test_targets,
        active_constraints=tuple(f"assert {gate}" for gate in spec.gates),
        failure_constraints=(),
        allowed_paths=fixture.editable_paths,
    )


def _error_branch_result(
    spec: BranchSpec,
    *,
    scope: SinkScope,
    error: str,
    lease_id: str,
    sandbox_mode: str,
) -> BranchResult:
    return BranchResult(
        branch_id=spec.branch_id,
        strategy=spec.strategy,
        transform_basis=spec.transform_basis,
        branch_index=spec.branch_index,
        candidate_index=spec.candidate_index,
        candidate="router_error",
        passed=False,
        eta_cog=0.0,
        verified_drop_bits=0.0,
        tokens=0,
        uncapped_tokens=0,
        cap_exceeded=False,
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
        directive="RECALL_SINK",
        gates={gate: "error" for gate in spec.gates},
        constraints=(f"assert not {_safe_name(error)}",),
        cost={},
        edits=(),
        candidate_hash="",
        pruned_hash="",
        error=str(error)[:240],
        sandbox_mode=sandbox_mode,
        scope=scope.as_dict(),
        lease_id=lease_id,
        worker_pid=os.getpid(),
    )


def _run_branch_worker(spec: BranchSpec, sink: ColdSink | None = None) -> BranchResult:
    """Top-level worker: picklable spec, locally constructed router and sink."""
    from ...bench import gates as bench_gates
    from ...bench.contracts import validated_edits

    scope = SinkScope(spec.user_id, spec.agent_id, spec.run_id, spec.branch_id)
    owns_sink = sink is None
    active_sink = sink or ColdSink(spec.sink_path, scope=scope)
    lease_id = ""
    sandbox: Path | None = None
    sandbox_mode = "not_created"
    try:
        try:
            lease_id = active_sink.open_branch_lease(
                run_id=spec.run_id,
                parent_hash=spec.parent_hash,
                branch_id=spec.branch_id,
                ttl_s=spec.lease_ttl_s,
                worker_pid=os.getpid(),
            )
        except SinkError as exc:
            return _error_branch_result(
                spec, scope=scope, error=str(exc), lease_id="", sandbox_mode="lease_failed"
            )
        sandbox, sandbox_mode = _materialise_sandbox(spec)
        try:
            fixture = spec.fixture.to_bug_fixture()
            router = _branch_router(spec, fixture)
            request = _branch_request(spec, fixture)
            response = router.propose(request)
            edits = validated_edits(response.edits, fixture.editable_paths)
            uncapped_tokens = int(response.raw.get("uncapped_tokens", response.tokens))
            cap_exceeded = uncapped_tokens > spec.router.max_tokens_per_branch
            if cap_exceeded:
                gate = bench_gates.GateReport(
                    passed=False,
                    verdicts={gate: "token_cap" for gate in spec.gates},
                    constraints=(f"assert tokens <= {spec.router.max_tokens_per_branch}",),
                    verified_drop_bits=0.0,
                    latency_ms=bench_gates.gate_latency_ms(fixture.test_targets),
                )
            else:
                bench_gates.apply_edits(sandbox, edits, allowed_paths=fixture.editable_paths)
                bench_gates._clear_pycache(sandbox)
                gate = bench_gates.evaluate_baseline_gates(
                    sandbox,
                    fixture.test_targets,
                    timeout_s=spec.timeout_s,
                )
            requested_green = all(gate.verdicts.get(name) == "pass" for name in spec.gates)
            passed = bool(gate.passed and requested_green and not cap_exceeded)
            cost = CognitiveStepCost(
                discarded_tokens=int(response.tokens) + spec.overhead_tokens,
                tool_cost=1.0 + len(edits) + len(fixture.test_targets) + spec.tool_overhead,
                pruned_branches=spec.branch_index,
                latency_ms=float(response.latency_ms) + float(gate.latency_ms),
            )
            eta_cog, directive = ThermodynamicGovernor().evaluate(
                float(gate.verified_drop_bits),
                cost,
                EntanglementReport(0.0, 0.0, 0, False),
            )
            eta_cog = round(float(eta_cog), 6)
            candidate = str(response.raw.get("candidate", "router_candidate"))
            payload = {
                "branch_id": spec.branch_id,
                "strategy": spec.strategy,
                "transform_basis": spec.transform_basis,
                "branch_index": spec.branch_index,
                "candidate_index": spec.candidate_index,
                "candidate": candidate,
                "trigger": spec.trigger,
                "run_id": spec.run_id,
                "parent_hash": spec.parent_hash,
                "fixture_id": fixture.id,
                "gates": dict(sorted(gate.verdicts.items())),
                "constraints": list(gate.constraints),
                "tokens": int(response.tokens),
                "uncapped_tokens": uncapped_tokens,
                "token_cap": spec.router.max_tokens_per_branch,
                "cap_exceeded": cap_exceeded,
                "eta_cog": eta_cog,
                "verified_drop_bits": round(float(gate.verified_drop_bits), 6),
                "passed": passed,
                "directive": directive.value,
                "router": spec.router.name,
                "endpoint": spec.router.endpoint,
                "model": spec.router.model,
                "seed": spec.router.seed,
                "offline": True,
                "network_calls": 0,
                "edits": [edit.as_dict() for edit in edits],
                "sandbox_mode": sandbox_mode,
                "error": "token_cap_exceeded" if cap_exceeded else "",
            }
            candidate_hash = active_sink.put(
                "branch_candidate",
                payload,
                scale=0,
                parent_hash=spec.parent_hash or None,
                eta_cog=eta_cog,
                scope=scope,
            )
            return BranchResult(
                branch_id=spec.branch_id,
                strategy=spec.strategy,
                transform_basis=spec.transform_basis,
                branch_index=spec.branch_index,
                candidate_index=spec.candidate_index,
                candidate=candidate,
                passed=passed,
                eta_cog=eta_cog,
                verified_drop_bits=round(float(gate.verified_drop_bits), 6),
                tokens=int(response.tokens),
                uncapped_tokens=uncapped_tokens,
                cap_exceeded=cap_exceeded,
                prompt_tokens=int(response.prompt_tokens),
                completion_tokens=int(response.completion_tokens),
                latency_ms=float(response.latency_ms),
                directive=directive.value,
                gates=dict(sorted(gate.verdicts.items())),
                constraints=tuple(str(item) for item in gate.constraints),
                cost=cost.as_dict(),
                edits=edits,
                candidate_hash=candidate_hash,
                pruned_hash="",
                error="token_cap_exceeded" if cap_exceeded else "",
                sandbox_mode=sandbox_mode,
                scope=scope.as_dict(),
                lease_id=lease_id,
                worker_pid=os.getpid(),
            )
        finally:
            if sandbox is not None:
                _cleanup_sandbox(spec, sandbox, sandbox_mode)
    except Exception as exc:
        return _error_branch_result(
            spec,
            scope=scope,
            error=f"{type(exc).__name__}: {exc}",
            lease_id=lease_id,
            sandbox_mode=sandbox_mode,
        )
    finally:
        if owns_sink:
            active_sink.close()


def _execute_specs(
    specs: tuple[BranchSpec, ...],
    *,
    sink: ColdSink,
    executor: str,
    max_workers: int,
) -> tuple[tuple[BranchResult, ...], str]:
    workers = max(1, min(int(max_workers), len(specs)))
    if executor == "thread":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_branch_worker, spec, sink) for spec in specs]
            return tuple(future.result() for future in futures), "thread"
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_run_branch_worker, spec, None) for spec in specs]
            return tuple(future.result() for future in futures), "process"
    except Exception:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_branch_worker, spec, sink) for spec in specs]
            return tuple(future.result() for future in futures), "thread_fallback"


def _branch_payload(
    result: BranchResult,
    *,
    run_id: str,
    parent_hash: str,
    trigger: str,
    token_cap: int,
) -> dict[str, Any]:
    return {
        "branch_id": result.branch_id,
        "strategy": result.strategy,
        "transform_basis": result.transform_basis,
        "branch_index": result.branch_index,
        "candidate": result.candidate,
        "candidate_hash": result.candidate_hash,
        "gates": dict(sorted(result.gates.items())),
        "constraints": list(result.constraints),
        "eta_cog": round(result.eta_cog, 6),
        "verified_drop_bits": round(result.verified_drop_bits, 6),
        "tokens": result.tokens,
        "uncapped_tokens": result.uncapped_tokens,
        "token_cap": token_cap,
        "cap_exceeded": result.cap_exceeded,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": round(result.latency_ms, 3),
        "cost": dict(result.cost),
        "edits": [edit.as_dict() for edit in result.edits],
        "run_id": run_id,
        "parent_hash": parent_hash,
        "trigger": trigger,
        "scope": dict(sorted(result.scope.items())),
        "offline": True,
        "network_calls": 0,
        "passed": result.passed,
        "error": result.error,
    }


def _prune_reason(result: BranchResult, *, winner_id: str) -> str:
    if result.branch_id == winner_id:
        return "winner"
    if result.cap_exceeded:
        return "token_cap_exceeded"
    if result.error:
        return result.error
    if not result.passed:
        return "gates_not_green"
    return "lower_eta_cog"


def run_branch_race(
    *,
    source_root: str | Path,
    config: RaceConfig,
    sink: ColdSink | None = None,
    sink_path: str | Path | None = None,
    run_id: str = "race",
    parent_hash: str | None = None,
    trigger: str | None = None,
    branches: Sequence[str] | None = None,
    gates: Sequence[str] | None = None,
    sandbox_parent: str | Path | None = None,
) -> BranchRaceResult:
    """Race 2-4 offline branches and commit at most one green WAL winner."""
    active_trigger = str(trigger or config.trigger)
    if active_trigger not in RACE_TRIGGERS:
        raise ValueError(f"branch race trigger must be one of {sorted(RACE_TRIGGERS)}")
    active_branches = tuple(str(item) for item in (branches or config.branches))
    active_gates = tuple(str(item) for item in (gates or config.gates))
    if not active_gates:
        raise ValueError("branch race requires at least one DarkStateGate")
    if not 2 <= len(active_branches) <= 4:
        raise ValueError("branch race requires 2 to 4 branches")
    if len(set(active_branches)) != len(active_branches):
        raise ValueError("branch race names must be distinct")

    root = str(Path(source_root).resolve())
    db_path = str(sink_path or (sink.db_path if sink is not None else ":memory:"))
    actual_parent = "" if parent_hash is None else str(parent_hash)
    race_run = str(run_id or "race")
    owns_sink = sink is None
    active_sink = sink or ColdSink(
        db_path,
        scope=SinkScope(config.user_id, config.agent_id, race_run),
    )
    executor = config.executor
    if db_path == ":memory:" and executor == "process":
        executor = "thread"
    if db_path == ":memory:" and sink is None:
        owns_sink = True

    temp_parent: tempfile.TemporaryDirectory[str] | None = None
    if sandbox_parent is None:
        temp_parent = tempfile.TemporaryDirectory(prefix="early-agi-race-")
        parent = Path(temp_parent.name)
    else:
        parent = Path(sandbox_parent)
    parent.mkdir(parents=True, exist_ok=True)

    router_config = RouterConfig(
        name=config.router_name,
        endpoint=config.endpoint,
        model=config.model,
        seed=config.seed,
        max_tokens_per_branch=config.max_tokens_per_branch,
    )
    candidate_count = max(1, len(config.fixture.candidates))
    specs = tuple(
        BranchSpec(
            branch_id=str(branch),
            strategy=f"{branch}:{_TRANSFORM_BASES[(index + config.seed) % len(_TRANSFORM_BASES)]}",
            transform_basis=_TRANSFORM_BASES[(index + config.seed) % len(_TRANSFORM_BASES)],
            branch_index=index,
            candidate_index=(index + config.seed) % candidate_count,
            run_id=race_run,
            parent_hash=actual_parent,
            trigger=active_trigger,
            source_root=root,
            sandbox_parent=str(parent),
            sink_path=db_path,
            user_id=config.user_id,
            agent_id=config.agent_id,
            fixture=config.fixture,
            gates=active_gates,
            router=router_config,
            timeout_s=config.timeout_s,
            lease_ttl_s=max(60.0, config.timeout_s * 2.0),
            overhead_tokens=index * 8,
            tool_overhead=index * 0.25,
        )
        for index, branch in enumerate(active_branches)
    )

    try:
        raw_results, executor_used = _execute_specs(
            specs,
            sink=active_sink,
            executor=executor,
            max_workers=config.max_workers,
        )
        green = [
            result
            for result in raw_results
            if result.passed and all(result.gates.get(name) == "pass" for name in active_gates)
        ]
        winner = (
            min(
                green,
                key=lambda item: (
                    -item.eta_cog,
                    item.tokens,
                    item.latency_ms,
                    item.branch_index,
                    item.branch_id,
                ),
            )
            if green
            else None
        )
        winner_id = winner.branch_id if winner is not None else ""

        winner_hash = ""
        if winner is not None:
            winner_payload = _branch_payload(
                winner,
                run_id=race_run,
                parent_hash=actual_parent,
                trigger=active_trigger,
                token_cap=config.max_tokens_per_branch,
            )
            winner_payload["kind"] = "race_winner"
            winner_payload["reason"] = "green_highest_eta_cog"
            winner_scope = SinkScope(config.user_id, config.agent_id, race_run, winner.branch_id)
            winner_hash = active_sink.put(
                "race_winner",
                winner_payload,
                scale=0,
                parent_hash=winner.candidate_hash or None,
                eta_cog=winner.eta_cog,
                scope=winner_scope,
            )
            if winner.lease_id:
                active_sink.claim_winner(
                    winner.lease_id,
                    winner_hash=winner_hash,
                    eta_cog=winner.eta_cog,
                )
        active_sink.prune_losers(
            race_run,
            actual_parent,
            winner_lease_id=winner.lease_id if winner is not None else "",
        )

        committed: list[BranchResult] = []
        pruned_hashes: list[str] = []
        for result in raw_results:
            if winner is not None and result.branch_id == winner.branch_id:
                committed.append(replace(result, pruned_hash=""))
                continue
            payload = _branch_payload(
                result,
                run_id=race_run,
                parent_hash=actual_parent,
                trigger=active_trigger,
                token_cap=config.max_tokens_per_branch,
            )
            payload["kind"] = "pruned_branch"
            payload["reason"] = _prune_reason(result, winner_id=winner_id)
            payload["winner_hash"] = winner_hash
            scope = SinkScope(config.user_id, config.agent_id, race_run, result.branch_id)
            pruned_hash = active_sink.put(
                "pruned_branch",
                payload,
                scale=0,
                parent_hash=result.candidate_hash or None,
                eta_cog=result.eta_cog,
                scope=scope,
            )
            pruned_hashes.append(pruned_hash)
            committed.append(replace(result, pruned_hash=pruned_hash))

        winner_committed = (
            next((item for item in committed if item.branch_id == winner_id), None)
            if winner is not None
            else None
        )
        mera_path = str(config.mera_path or (f"{db_path}.mera.db" if db_path != ":memory:" else ":memory:"))
        lineage_edges = 0
        if winner_hash and pruned_hashes:
            graph = MeraGraph(active_sink, mera_path)
            try:
                graph.record_race_lineage(winner_hash, pruned_hashes)
                current = set(pruned_hashes)
                lineage_edges = sum(
                    1
                    for edge in graph.edges()
                    if edge["src"] in current
                    and edge["dst"] == winner_hash
                    and edge["relation"] in {SUPERSEDED, MERGED}
                )
            finally:
                graph.close()

        return BranchRaceResult(
            run_id=race_run,
            parent_hash=actual_parent,
            trigger=active_trigger,
            executor=executor_used,
            branches=tuple(committed),
            winner=winner_committed,
            winner_hash=winner_hash,
            pruned_hashes=tuple(pruned_hashes),
            lineage_edges=lineage_edges,
            total_tokens=sum(item.tokens for item in committed),
            router_name=config.router_name,
            endpoint=config.endpoint,
            model=config.model,
            seed=config.seed,
            max_tokens_per_branch=config.max_tokens_per_branch,
            sandbox_modes=tuple(sorted({item.sandbox_mode for item in committed})),
            mera_path=mera_path,
        )
    finally:
        if temp_parent is not None:
            temp_parent.cleanup()
        if owns_sink:
            active_sink.close()


def race_fixture_spec(fixture: RaceFixtureSpec | Any) -> RaceFixtureSpec:
    """Coerce a benchmark fixture into its picklable race mirror."""
    return fixture if isinstance(fixture, RaceFixtureSpec) else RaceFixtureSpec.from_fixture(fixture)
