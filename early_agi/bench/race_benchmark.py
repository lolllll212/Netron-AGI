"""Three-arm offline A/B benchmark with a parallel WAL branch-racing arm.

The default two-arm report is left untouched; this module adds ``netron_race``
only when explicitly requested. Tangled multi-module fixtures force the plain
retry loop through several failed candidates while the race arm evaluates four
reversible-transform bases concurrently and crystallises one green winner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..core.branch_race import (
    RaceConfig,
    RaceFixtureSpec,
    run_branch_race,
)
from ..core.cold_sink import ColdSink, SinkScope
from .contracts import PatchRouter
from .escalation import StallTracker
from .fixtures import BugFixture, tangled_fixtures
from .gates import (
    apply_edits,
    attempt_cost,
    capture_snapshot,
    gate_latency_ms,
    write_workspace,
)
from .metrics import build_race_report
from .records import ArmState, AttemptExecution, AttemptRecord
from .routers import OfflineFixtureRouter
from .runner import ABBenchmark

__all__ = ["RaceABBenchmark", "run_race_ab_benchmark"]


class RaceABBenchmark(ABBenchmark):
    """Run plain_retry, netron, and netron_race arms over tangled fixtures."""

    def __init__(
        self,
        *,
        router: PatchRouter | None = None,
        fixtures: Sequence[BugFixture] | None = None,
        k: int = 4,
        sequential_tasks: int = 6,
        timeout_s: float = 60.0,
        endpoint: str = "",
        model: str = "",
        seed: int = 0,
        max_tokens_per_branch: int = 4096,
        max_workers: int = 4,
    ) -> None:
        selected = tuple(fixtures or tangled_fixtures())
        super().__init__(
            router=router or OfflineFixtureRouter(selected),
            fixtures=selected,
            k=k,
            sequential_tasks=sequential_tasks,
            timeout_s=timeout_s,
        )
        if max_tokens_per_branch < 1:
            raise ValueError("max_tokens_per_branch must be >= 1")
        self.race_endpoint = str(endpoint)
        self.race_model = str(model)
        self.race_seed = int(seed)
        self.race_max_tokens_per_branch = int(max_tokens_per_branch)
        self.race_max_workers = max(2, min(4, int(max_workers)))
        self.race_summaries: list[dict[str, Any]] = []
        self.race_config = RaceConfig(
            fixture=RaceFixtureSpec.from_fixture(self.fixtures[0]),
            endpoint=self.race_endpoint,
            model=self.race_model,
            seed=self.race_seed,
            max_tokens_per_branch=self.race_max_tokens_per_branch,
            max_workers=self.race_max_workers,
            executor="process",
            timeout_s=self.timeout_s,
            user_id="benchmark",
            agent_id="netron_race",
            router_name="mocked_live" if self.race_endpoint else "offline_fixture",
        )

    def run(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ABBenchmark is closed")
        arms = {
            "plain_retry": ArmState("plain_retry"),
            "netron": ArmState("netron"),
        }
        race_state = ArmState("netron_race")
        self.race_summaries = []
        with (
            _SequentialRaceBenchmark(
                router=self.router,
                fixtures=self.fixtures,
                k=self.k,
                sequential_tasks=self.sequential_tasks,
                timeout_s=self.timeout_s,
            ) as sequential,
            TemporaryDirectory(prefix="early-agi-race-ab-") as tmp,
        ):
            base = Path(tmp)
            wal_path = base / "race-wal.db"
            mera_path = base / "race-mera.db"
            sandboxes = base / "sandboxes"
            sandboxes.mkdir(parents=True, exist_ok=True)
            ColdSink(
                str(wal_path),
                scope=SinkScope(user_id="benchmark", agent_id="netron_race"),
            ).close()
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

                race_root = base / f"netron_race-{task_index:02d}"
                write_workspace(fixture, race_root)
                # Use a separate governor/reflex archive so the comparison arm
                # cannot learn from the sequential arm's earlier run.
                first_seen = fixture.id not in race_state.seen
                task = sequential._run_task(
                    arm_name="netron",
                    state=race_state,
                    task_index=task_index,
                    fixture=fixture,
                    root=race_root,
                    first_seen=first_seen,
                )
                # Failed verification alone is not an escalation signal. Require
                # an actual governance stall and leave room in the turn budget.
                stalled = not task.success and task.attempts and bool((task.preflight or {}).get("stall"))
                if stalled and self.k - len(task.attempts) >= 2:
                    config = replace(
                        self.race_config,
                        fixture=RaceFixtureSpec.from_fixture(fixture),
                        mera_path=str(mera_path),
                        branches=self.race_config.branches[: self.k - len(task.attempts)],
                    )
                    result = run_branch_race(
                        source_root=race_root,
                        config=config,
                        sink_path=str(wal_path),
                        run_id=f"race-{task_index:02d}",
                        trigger="frustration",
                        sandbox_parent=sandboxes,
                    )
                    before = capture_snapshot(
                        race_root, result.winner_edits, allowed_paths=fixture.editable_paths
                    )
                    if result.ok:
                        apply_edits(race_root, result.winner_edits, allowed_paths=fixture.editable_paths)
                    summary = result.as_dict()
                    self.race_summaries.append(summary)
                    task.attempts.append(
                        replace(
                            self._race_attempt_record(
                                result=result,
                                fixture=fixture,
                                task_index=task_index,
                            ),
                            attempt=len(task.attempts) + len(result.branches),
                        )
                    )
                    if result.ok and first_seen:
                        attempt = task.attempts[-1]
                        race_state.reflex_compiles += sequential._archive_reflex(
                            AttemptExecution(attempt, result.winner_edits, before),
                            fixture=fixture,
                            task_index=task_index,
                            eta_cog=attempt.eta_cog,
                            directive=attempt.directive,
                        )
                    task.preflight = {**(task.preflight or {}), "race": summary}
                race_state.tasks.append(task)
                race_state.seen.add(fixture.id)
                if task.success:
                    race_state.solved_workspaces.append((race_root, fixture))
                point = self._retention_point(race_state, task_index)
                race_state.retention.append(point)
                task.retention = point

        return build_race_report(
            self,
            arms,
            race_state=race_state,
            race_summaries=self.race_summaries,
            race_config=self.race_config,
        )

    def _race_attempt_record(
        self,
        *,
        result: Any,
        fixture: BugFixture,
        task_index: int,
    ) -> AttemptRecord:
        winner = result.winner
        passed = bool(result.ok)
        tokens = int(result.total_tokens)
        prompt_tokens = sum(int(item.prompt_tokens) for item in result.branches)
        completion_tokens = sum(int(item.completion_tokens) for item in result.branches)
        latency_ms = sum(float(item.latency_ms) for item in result.branches)
        cost = attempt_cost(
            tokens=tokens,
            latency_ms=latency_ms,
            edit_count=len(result.winner_edits),
            target_count=len(fixture.test_targets),
            attempt=1,
            gate_latency=gate_latency_ms(fixture.test_targets),
        )
        return AttemptRecord(
            attempt=1,
            orchestrator="netron_race",
            candidate=winner.candidate if winner is not None else "no_green_branch",
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=cost.latency_ms,
            eta_cog=winner.eta_cog if winner is not None else 0.0,
            verified_drop_bits=winner.verified_drop_bits if winner is not None else 0.0,
            gates=winner.gates if winner is not None else {},
            constraints=winner.constraints if winner is not None else ("assert green_branch",),
            cost=cost.as_dict(),
            directive="CRYSTALLIZE" if passed else "RECALL_SINK",
            backend="branch_race+bounded_chi",
            passed=passed,
            rollback=False,
            reflex=False,
            preflight_probe=False,
            violation_count=0,
            error="" if passed else "no_green_branch",
        )


def run_race_ab_benchmark(
    *,
    router: PatchRouter | None = None,
    fixtures: Sequence[BugFixture] | None = None,
    k: int = 4,
    sequential_tasks: int = 6,
    timeout_s: float = 60.0,
    endpoint: str = "",
    model: str = "",
    seed: int = 0,
    max_tokens_per_branch: int = 4096,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run the explicit three-arm branch-racing benchmark and close SQLite handles."""
    benchmark = RaceABBenchmark(
        router=router,
        fixtures=fixtures,
        k=k,
        sequential_tasks=sequential_tasks,
        timeout_s=timeout_s,
        endpoint=endpoint,
        model=model,
        seed=seed,
        max_tokens_per_branch=max_tokens_per_branch,
        max_workers=max_workers,
    )
    try:
        return benchmark.run()
    finally:
        benchmark.close()


class _SequentialRaceBenchmark(ABBenchmark):
    """Same fast/reflex path, yielding early only on a failed governance stall."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _run_task(self, **kwargs):
        budget = self.k
        self.tracker = StallTracker()
        try:
            task = super()._run_task(**kwargs)
            if self.tracker.recall_streak:
                task.preflight = {
                    **(task.preflight or {}),
                    "stall": {
                        "trigger": "frustration",
                        "recall_streak": self.tracker.recall_streak,
                        "evidence": "repeated_failed_gates_or_failed_shift",
                    },
                }
            return task
        finally:
            self.k = budget

    def _run_router_attempt(self, **kwargs):
        result = super()._run_router_attempt(**kwargs)
        if self.tracker.observe(result.record) and self.k - kwargs["request"].attempt >= 2:
            # Preserve the failed observation; the coordinator may now escalate.
            self.k = kwargs["request"].attempt
        return result
