"""Head-to-head A/B repair benchmark with real local pytest gates.

The plain retry arm sees only failed-test feedback. The Netron arm runs the same
fixtures through `netron_plastic.gauge`, the bounded-χ backend, active manifold
constraints, and a shared `ReflexLibrary`. Successful Netron edits are archived
as gated ColdSink traces, compiled into exact-prestate reflexes, and replayed on
repeat tasks with zero router tokens.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..backends.bounded_chi import BoundedChiBackend
from ..core.cold_sink import ColdSink
from ..core.governor import CognitiveStepCost
from ..memory.mera_graph import MeraGraph
from ..memory.reflex import CompiledReflex, ReflexLibrary, content_sha
from .contracts import FileEdit, PatchRequest, PatchResponse, PatchRouter, validated_edits
from .fixtures import BugFixture, builtin_fixtures
from .gates import (
    GateReport,
    GaugeRunReport,
    apply_edits,
    attempt_cost,
    baseline_eta,
    capture_snapshot,
    evaluate_baseline_gates,
    gate_latency_ms,
    restore_snapshot,
    run_netron_gauge,
    write_workspace,
)
from .metrics import build_report
from .records import (
    ArmState,
    AttemptExecution,
    AttemptRecord,
    RetentionPoint,
    TaskRecord,
    candidate_name,
    dedupe_strings,
)
from .routers import OfflineFixtureRouter

__all__ = ["ABBenchmark", "run_ab_benchmark"]


class ABBenchmark:
    """Run a deterministic offline A/B benchmark or a live-router stress run."""

    def __init__(
        self,
        *,
        router: PatchRouter | None = None,
        fixtures: Sequence[BugFixture] | None = None,
        k: int = 4,
        sequential_tasks: int = 10,
        timeout_s: float = 60.0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if sequential_tasks < 1:
            raise ValueError("sequential_tasks must be >= 1")
        if timeout_s < 1.0:
            raise ValueError("timeout_s must be >= 1.0")
        self.fixtures = tuple(fixtures or builtin_fixtures())
        if not self.fixtures:
            raise ValueError("at least one fixture is required")
        self.fixture_by_id = {fixture.id: fixture for fixture in self.fixtures}
        self.k = int(k)
        self.sequential_tasks = int(sequential_tasks)
        self.timeout_s = float(timeout_s)
        self.task_ids = tuple(
            self.fixtures[index % len(self.fixtures)].id for index in range(self.sequential_tasks)
        )
        self.router = router or OfflineFixtureRouter(self.fixtures)
        self.sink = ColdSink(":memory:")
        self.backend = BoundedChiBackend()
        self.reflex_graph = MeraGraph(self.sink, ":memory:", backend=self.backend)
        self.reflexes = ReflexLibrary(self.reflex_graph)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.reflex_graph.close()
        self.sink.close()

    def run(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ABBenchmark is closed")
        arms = {
            "plain_retry": ArmState("plain_retry"),
            "netron": ArmState("netron"),
        }
        with TemporaryDirectory(prefix="early-agi-ab-") as tmp:
            base = Path(tmp)
            for task_index, fixture_id in enumerate(self.task_ids, start=1):
                fixture = self.fixture_by_id[fixture_id]
                for arm_name, state in arms.items():
                    root = base / f"{arm_name}-{task_index:02d}"
                    write_workspace(fixture, root)
                    first_seen = fixture.id not in state.seen
                    record = self._run_task(
                        arm_name=arm_name,
                        state=state,
                        task_index=task_index,
                        fixture=fixture,
                        root=root,
                        first_seen=first_seen,
                    )
                    state.tasks.append(record)
                    state.seen.add(fixture.id)
                    if record.success:
                        state.solved_workspaces.append((root, fixture))
                    point = self._retention_point(state, task_index)
                    state.retention.append(point)
                    record.retention = point
        return self._report(arms)

    # -- task execution -----------------------------------------------------
    def _run_task(
        self,
        *,
        arm_name: str,
        state: ArmState,
        task_index: int,
        fixture: BugFixture,
        root: Path,
        first_seen: bool,
    ) -> TaskRecord:
        attempts: list[AttemptRecord] = []
        failure_constraints: tuple[str, ...] = ()
        active_constraints: tuple[str, ...] = ()
        reflex_hit = False
        preflight: dict[str, Any] | None = None
        success: AttemptExecution | None = None

        if arm_name == "netron" and not first_seen:
            reflex = self.reflexes.match(root, task=fixture.task)
            if reflex is not None:
                reflex_hit = True
                replay = self._run_reflex_attempt(
                    reflex=reflex,
                    task_index=task_index,
                    fixture=fixture,
                    root=root,
                )
                attempts.append(replay.record)
                if replay.record.passed:
                    return self._task_record(
                        task_index=task_index,
                        fixture=fixture,
                        first_seen=first_seen,
                        attempts=attempts,
                        reflex_hit=reflex_hit,
                        preflight=preflight,
                    )
                failure_constraints = dedupe_strings(
                    (*failure_constraints, *replay.record.constraints, fixture.probe_hint)
                )
                if len(attempts) >= self.k:
                    return self._task_record(
                        task_index=task_index,
                        fixture=fixture,
                        first_seen=first_seen,
                        attempts=attempts,
                        reflex_hit=reflex_hit,
                        preflight=preflight,
                    )

        if arm_name == "netron":
            preflight_report = self._preflight(fixture=fixture, root=root)
            state.preflight_probes += 1
            preflight = {
                "probe": preflight_report.preflight_probe,
                "operators": list(preflight_report.probe_operators),
                "constraints": list(preflight_report.constraints),
                "directive": preflight_report.directive,
                "eta_cog": preflight_report.eta_cog,
                "gates": dict(preflight_report.gates),
                "backend": preflight_report.backend,
                "violation_count": preflight_report.violation_count,
            }
            active_constraints = dedupe_strings((*preflight_report.constraints, fixture.probe_hint))

        while len(attempts) < self.k:
            request = self._request(
                arm_name=arm_name,
                fixture=fixture,
                attempt=len(attempts) + 1,
                active_constraints=active_constraints if arm_name == "netron" else (),
                failure_constraints=failure_constraints,
            )
            execution = self._run_router_attempt(
                request=request,
                arm_name=arm_name,
                task_index=task_index,
                fixture=fixture,
                root=root,
            )
            attempts.append(execution.record)
            if execution.record.passed:
                success = execution
                break
            hint = fixture.probe_hint if arm_name == "netron" else ""
            failure_constraints = dedupe_strings((*failure_constraints, *execution.record.constraints, hint))

        if success is not None and arm_name == "netron" and first_seen:
            state.reflex_compiles += self._archive_reflex(
                success,
                fixture=fixture,
                task_index=task_index,
                eta_cog=success.record.eta_cog,
                directive=success.record.directive,
            )
        return self._task_record(
            task_index=task_index,
            fixture=fixture,
            first_seen=first_seen,
            attempts=attempts,
            reflex_hit=reflex_hit,
            preflight=preflight,
        )

    def _request(
        self,
        *,
        arm_name: str,
        fixture: BugFixture,
        attempt: int,
        active_constraints: Sequence[str],
        failure_constraints: Sequence[str],
    ) -> PatchRequest:
        return PatchRequest(
            fixture_id=fixture.id,
            task=fixture.task,
            attempt=attempt,
            orchestrator="netron_gauge" if arm_name == "netron" else arm_name,
            files=fixture.file_map,
            test_targets=fixture.test_targets,
            active_constraints=tuple(active_constraints),
            failure_constraints=tuple(failure_constraints),
            allowed_paths=fixture.editable_paths,
        )

    def _preflight(self, *, fixture: BugFixture, root: Path) -> GaugeRunReport:
        cost = attempt_cost(
            tokens=0,
            latency_ms=0.0,
            edit_count=0,
            target_count=len(fixture.test_targets),
            attempt=1,
            gate_latency=gate_latency_ms(fixture.test_targets),
        )
        return run_netron_gauge(
            root,
            sink=self.sink,
            backend=self.backend,
            metered_cost=cost,
            timeout_s=self.timeout_s,
        )

    # -- attempts -----------------------------------------------------------
    def _run_router_attempt(
        self,
        *,
        request: PatchRequest,
        arm_name: str,
        task_index: int,
        fixture: BugFixture,
        root: Path,
    ) -> AttemptExecution:
        snapshot: Mapping[str, bytes | None] = {}
        edits: tuple[FileEdit, ...] = ()
        try:
            response = self.router.propose(request)
            edits = validated_edits(response.edits, fixture.editable_paths)
            snapshot = capture_snapshot(root, edits, allowed_paths=fixture.editable_paths)
            apply_edits(root, edits, allowed_paths=fixture.editable_paths)
        except Exception as exc:
            restore_snapshot(root, snapshot)
            record = self._error_record(
                attempt=request.attempt,
                arm_name=arm_name,
                fixture=fixture,
                reason=type(exc).__name__,
                rollback=bool(snapshot),
            )
            return AttemptExecution(record=record, edits=(), before={})
        cost = attempt_cost(
            tokens=response.tokens,
            latency_ms=response.latency_ms,
            edit_count=len(edits),
            target_count=len(fixture.test_targets),
            attempt=request.attempt,
            gate_latency=gate_latency_ms(fixture.test_targets),
        )
        if arm_name == "plain_retry":
            gate = evaluate_baseline_gates(
                root,
                fixture.test_targets,
                timeout_s=self.timeout_s,
            )
            eta_cog, directive = baseline_eta(gate, cost)
            passed = gate.passed
            rollback = not passed
            if rollback:
                restore_snapshot(root, snapshot)
            record = self._gate_record(
                attempt=request.attempt,
                arm_name=arm_name,
                fixture=fixture,
                response=response,
                cost=cost,
                gate=gate,
                eta_cog=eta_cog,
                directive=directive,
                backend="plain_retry",
                passed=passed,
                rollback=rollback,
                reflex=False,
                preflight_probe=False,
                violation_count=0,
            )
        else:
            gauge = run_netron_gauge(
                root,
                sink=self.sink,
                backend=self.backend,
                metered_cost=cost,
                timeout_s=self.timeout_s,
            )
            passed = gauge.passed
            rollback = not passed
            if rollback:
                restore_snapshot(root, snapshot)
            record = self._gauge_record(
                attempt=request.attempt,
                arm_name=arm_name,
                fixture=fixture,
                response=response,
                cost=cost,
                gauge=gauge,
                passed=passed,
                rollback=rollback,
                reflex=False,
            )
        return AttemptExecution(record=record, edits=edits, before=snapshot)

    def _run_reflex_attempt(
        self,
        *,
        reflex: CompiledReflex,
        task_index: int,
        fixture: BugFixture,
        root: Path,
    ) -> AttemptExecution:
        edits = tuple(FileEdit(edit.path, edit.content) for edit in reflex.edits)
        cost = attempt_cost(
            tokens=0,
            latency_ms=0.0,
            edit_count=len(edits),
            target_count=len(fixture.test_targets),
            attempt=1,
            gate_latency=gate_latency_ms(fixture.test_targets),
        )
        snapshot: Mapping[str, bytes | None] = {}
        try:
            edits = validated_edits(edits, fixture.editable_paths)
            snapshot = capture_snapshot(root, edits, allowed_paths=fixture.editable_paths)
            apply_edits(root, edits, allowed_paths=fixture.editable_paths)
            gauge = run_netron_gauge(
                root,
                sink=self.sink,
                backend=self.backend,
                metered_cost=cost,
                timeout_s=self.timeout_s,
            )
            passing_gates = {name for name, verdict in gauge.gates.items() if verdict == "pass"}
            satisfied = set(reflex.expected_gates).issubset(passing_gates)
            passed = gauge.passed and satisfied
            if not passed:
                restore_snapshot(root, snapshot)
            self.reflexes.record(
                reflex,
                ok=passed,
                eta_cog=gauge.eta_cog,
                parent_hash=None,
            )
            response = PatchResponse(
                edits=edits,
                rationale=f"reflex {reflex.name}: 0 router tokens",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                raw={"candidate": reflex.name, "offline": True},
            )
            record = self._gauge_record(
                attempt=1,
                arm_name="netron",
                fixture=fixture,
                response=response,
                cost=cost,
                gauge=gauge,
                passed=passed,
                rollback=not passed,
                reflex=True,
            )
            return AttemptExecution(record=record, edits=edits, before=snapshot)
        except Exception as exc:
            restore_snapshot(root, snapshot)
            with contextlib.suppress(Exception):
                self.reflexes.record(reflex, ok=False, eta_cog=0.0, parent_hash=None)
            record = self._error_record(
                attempt=1,
                arm_name="netron",
                fixture=fixture,
                reason=type(exc).__name__,
                rollback=bool(snapshot),
                reflex=True,
            )
            return AttemptExecution(record=record, edits=(), before={})

    # -- records ------------------------------------------------------------
    def _error_record(
        self,
        *,
        attempt: int,
        arm_name: str,
        fixture: BugFixture,
        reason: str,
        rollback: bool,
        reflex: bool = False,
    ) -> AttemptRecord:
        cost = attempt_cost(
            tokens=0,
            latency_ms=0.0,
            edit_count=0,
            target_count=len(fixture.test_targets),
            attempt=attempt,
            gate_latency=gate_latency_ms(fixture.test_targets),
        )
        return AttemptRecord(
            attempt=attempt,
            orchestrator=arm_name,
            candidate="router_error",
            tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=cost.latency_ms,
            eta_cog=0.0,
            verified_drop_bits=0.0,
            gates={},
            constraints=("assert patch_safe",),
            cost=cost.as_dict(),
            directive="",
            backend="bounded_chi" if arm_name == "netron" else "plain_retry",
            passed=False,
            rollback=rollback,
            reflex=reflex,
            preflight_probe=False,
            violation_count=0,
            error=reason[:120],
        )

    def _gate_record(
        self,
        *,
        attempt: int,
        arm_name: str,
        fixture: BugFixture,
        response: PatchResponse,
        cost: CognitiveStepCost,
        gate: GateReport,
        eta_cog: float,
        directive: str,
        backend: str,
        passed: bool,
        rollback: bool,
        reflex: bool,
        preflight_probe: bool,
        violation_count: int,
    ) -> AttemptRecord:
        return AttemptRecord(
            attempt=attempt,
            orchestrator=arm_name,
            candidate=candidate_name(response),
            tokens=response.tokens,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=cost.latency_ms,
            eta_cog=eta_cog,
            verified_drop_bits=gate.verified_drop_bits,
            gates=gate.verdicts,
            constraints=gate.constraints,
            cost=cost.as_dict(),
            directive=directive,
            backend=backend,
            passed=passed,
            rollback=rollback,
            reflex=reflex,
            preflight_probe=preflight_probe,
            violation_count=violation_count,
        )

    def _gauge_record(
        self,
        *,
        attempt: int,
        arm_name: str,
        fixture: BugFixture,
        response: PatchResponse,
        cost: CognitiveStepCost,
        gauge: GaugeRunReport,
        passed: bool,
        rollback: bool,
        reflex: bool,
    ) -> AttemptRecord:
        constraints = gauge.constraints or (fixture.probe_hint,)
        return AttemptRecord(
            attempt=attempt,
            orchestrator=arm_name,
            candidate=candidate_name(response),
            tokens=response.tokens,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=cost.latency_ms,
            eta_cog=gauge.eta_cog,
            verified_drop_bits=gauge.verified_drop_bits,
            gates=gauge.gates,
            constraints=constraints,
            cost=cost.as_dict(),
            directive=gauge.directive,
            backend=gauge.backend,
            passed=passed,
            rollback=rollback,
            reflex=reflex,
            preflight_probe=gauge.preflight_probe,
            violation_count=gauge.violation_count,
        )

    def _task_record(
        self,
        *,
        task_index: int,
        fixture: BugFixture,
        first_seen: bool,
        attempts: list[AttemptRecord],
        reflex_hit: bool,
        preflight: dict[str, Any] | None,
    ) -> TaskRecord:
        return TaskRecord(
            task_index=task_index,
            fixture_id=fixture.id,
            first_seen=first_seen,
            attempts=list(attempts),
            reflex_hit=reflex_hit,
            preflight=preflight,
        )

    # -- reflex memory ------------------------------------------------------
    def _archive_reflex(
        self,
        execution: AttemptExecution,
        *,
        fixture: BugFixture,
        task_index: int,
        eta_cog: float,
        directive: str,
    ) -> int:
        if not execution.edits:
            return 0
        tick_payload = {
            "task": fixture.task,
            "state": {"task": fixture.task, "fixture": fixture.id},
            "gates": {"syntax_valid": "pass", "tests_passing": "pass"},
            "ok": True,
            "has_regressions": False,
            "directive": directive or "CRYSTALLIZE",
            "eta_cog": round(float(eta_cog), 6),
            "fixture": fixture.id,
            "run": task_index,
            "arm": "netron",
        }
        tick_hash = self.sink.put(
            "uv_state",
            tick_payload,
            scale=0,
            eta_cog=eta_cog,
        )
        for edit in execution.edits:
            self.sink.put(
                "applied_edit",
                {
                    "path": edit.path,
                    "content": edit.content,
                    "before_sha": content_sha(execution.before.get(edit.path)),
                    "after_sha": content_sha(edit.content),
                    "source": "benchmark_router",
                    "tick": tick_hash,
                    "fixture": fixture.id,
                    "run": task_index,
                },
                scale=0,
                parent_hash=tick_hash,
                eta_cog=eta_cog,
            )
        report = self.reflexes.compile(min_support=1)
        return int(report.reflexes)

    # -- curves -------------------------------------------------------------
    def _retention_point(self, state: ArmState, task_index: int) -> RetentionPoint:
        checked = 0
        passed = 0
        for root, fixture in state.solved_workspaces:
            checked += 1
            report = evaluate_baseline_gates(
                root,
                fixture.test_targets,
                timeout_s=self.timeout_s,
            )
            if report.passed:
                passed += 1
        rate = 1.0 if checked == 0 else passed / checked
        previous = state.retention[-1].candidate_gate_pass_rate if state.retention else 1.0
        regressions = checked - passed
        return RetentionPoint(
            task_index=task_index,
            checked_workspaces=checked,
            passed_workspaces=passed,
            historical_gate_pass_rate=previous,
            candidate_gate_pass_rate=rate,
            regressions=regressions,
            zero_regression=regressions == 0 and rate + 1e-9 >= previous,
        )

    def _report(self, arms: Mapping[str, ArmState]) -> dict[str, Any]:
        return build_report(self, arms)


def run_ab_benchmark(
    *,
    router: PatchRouter | None = None,
    fixtures: Sequence[BugFixture] | None = None,
    k: int = 4,
    sequential_tasks: int = 10,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Run one A/B benchmark and close every SQLite handle it opened."""
    benchmark = ABBenchmark(
        router=router,
        fixtures=fixtures,
        k=k,
        sequential_tasks=sequential_tasks,
        timeout_s=timeout_s,
    )
    try:
        return benchmark.run()
    finally:
        benchmark.close()
