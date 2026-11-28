"""Escalation policy stress, not evidence about real model solve rates."""

from dataclasses import replace
from unittest.mock import patch

from early_agi.bench import run_race_ab_benchmark
from early_agi.bench.fixtures import builtin_fixtures, hard_stall_fixtures
from early_agi.bench.routers import OfflineFixtureRouter


def test_easy_tasks_never_fork_and_preserve_token_savings():
    with patch(
        "early_agi.bench.race_benchmark.run_branch_race", side_effect=AssertionError("unexpected fork")
    ):
        report = run_race_ab_benchmark(fixtures=builtin_fixtures(), sequential_tasks=4)
    assert report["summary"]["netron_race_tokens_saved"] > 0
    assert report["summary"]["netron_race_tokens_saved"] == report["summary"]["netron_tokens_saved"]
    assert report["summary"]["branch_race_tokens"] == 0


class GreedyStallRouter(OfflineFixtureRouter):
    """Simulate repeated local-cut fixation; every arm has the same candidates.

    The core racer explores cuts rather than retrying this greedy strategy. This
    demonstrates recovery under a scripted policy, not a reserved golden fix.
    """

    def propose(self, request):
        return super().propose(
            replace(request, attempt=1, orchestrator="greedy", active_constraints=(), failure_constraints=())
        )


def test_hard_stall_escalates_and_keeps_failed_attempt_cost():
    fixtures = hard_stall_fixtures()[:2]
    report = run_race_ab_benchmark(
        router=GreedyStallRouter(fixtures), fixtures=fixtures, sequential_tasks=2, k=4
    )
    assert report["summary"]["baseline_solved"] == 0
    assert report["summary"]["netron_solved"] == 0
    assert report["summary"]["netron_race_solved"] == 2
    for task in report["arms"]["netron_race"]["tasks"]:
        assert not task["attempts"][0]["passed"]
        assert task["attempts"][0]["tokens"] > 0
        assert task["preflight"]["race"]["branch_count"] <= 3
        assert task["success"]


def test_recovered_hard_task_replays_without_another_race():
    fixtures = hard_stall_fixtures()[:1]
    report = run_race_ab_benchmark(
        router=GreedyStallRouter(fixtures), fixtures=fixtures, sequential_tasks=2, k=4
    )
    tasks = report["arms"]["netron_race"]["tasks"]
    assert tasks[0]["success"] and tasks[1]["success"]
    assert tasks[1]["reflex_hit"]
    assert tasks[1]["tokens"] == 0
    assert len(report["arms"]["netron_race"]["race_summaries"]) == 1
