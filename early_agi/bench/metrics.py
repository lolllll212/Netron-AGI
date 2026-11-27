"""Metric curves and final JSON report for the A/B benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .records import ArmState, mean, ratio

__all__ = ["build_report"]


def build_report(benchmark: Any, arms: Mapping[str, ArmState]) -> dict[str, Any]:
    baseline = arms["plain_retry"]
    netron = arms["netron"]
    curves = {
        "discovery@k": discovery_curve(benchmark, baseline, netron),
        "zero_forgetting_retention": retention_curve(baseline, netron),
        "repeat_task_reflex_speedup": repeat_curve(baseline, netron),
    }
    zero_regression = bool(curves["zero_forgetting_retention"]["zero_regression"])
    repeat_token_speedup = float(curves["repeat_task_reflex_speedup"]["token_speedup"])
    offline = bool(getattr(benchmark.router, "offline", False))
    summary = {
        "ok": (
            baseline.solved_count == len(benchmark.task_ids)
            and netron.solved_count == len(benchmark.task_ids)
            and zero_regression
            and netron.reflex_hits > 0
            and repeat_token_speedup > 1.0
        ),
        "offline": offline,
        "sequential_tasks": len(benchmark.task_ids),
        "baseline_solved": baseline.solved_count,
        "netron_solved": netron.solved_count,
        "netron_tokens_saved": baseline.tokens_spent - netron.tokens_spent,
        "netron_rollbacks_avoided": baseline.rollbacks - netron.rollbacks,
        "reflex_hits": netron.reflex_hits,
        "repeat_token_speedup": round(repeat_token_speedup, 6),
        "zero_regression": zero_regression,
    }
    return {
        "offline": offline,
        "network_calls": 0 if offline else None,
        "config": {
            "router": str(getattr(benchmark.router, "name", type(benchmark.router).__name__)),
            "k": benchmark.k,
            "sequential_tasks": benchmark.sequential_tasks,
            "timeout_s": benchmark.timeout_s,
            "fixtures": [fixture.as_dict() for fixture in benchmark.fixtures],
            "baseline": {
                "controller": "plain_retry_loop",
                "gate": "ast_syntax_gate+pytest_gate",
            },
            "netron": {
                "controller": "netron_plastic_gauge",
                "program": "netron_plastic.gauge",
                "backend": "bounded_chi",
                "reflex_library": "ReflexLibrary",
                "gate": "GaugeGate(ast_check+pytest_run)",
            },
        },
        "arms": {
            "plain_retry": {**baseline.as_dict(), "controller": "plain_retry_loop"},
            "netron": {
                **netron.as_dict(),
                "controller": "netron_plastic_gauge",
                "program": "netron_plastic.gauge",
                "backend": "bounded_chi",
                "reflex_library": True,
            },
        },
        "curves": curves,
        "summary": summary,
    }


def discovery_curve(benchmark: Any, baseline: ArmState, netron: ArmState) -> dict[str, Any]:
    baseline_first = [task for task in baseline.tasks if task.first_seen]
    netron_first = [task for task in netron.tasks if task.first_seen]
    grid = list(range(1, benchmark.k + 1))
    baseline_den = max(1, len(baseline_first))
    netron_den = max(1, len(netron_first))
    baseline_rates = [
        sum(
            1
            for task in baseline_first
            if task.discovery_attempt is not None and task.discovery_attempt <= point
        )
        / baseline_den
        for point in grid
    ]
    netron_rates = [
        sum(
            1
            for task in netron_first
            if task.discovery_attempt is not None and task.discovery_attempt <= point
        )
        / netron_den
        for point in grid
    ]
    baseline_attempts = [task.discovery_attempt or benchmark.k + 1 for task in baseline_first]
    netron_attempts = [task.discovery_attempt or benchmark.k + 1 for task in netron_first]
    baseline_tokens = sum(task.tokens for task in baseline_first)
    netron_tokens = sum(task.tokens for task in netron_first)
    baseline_eta = mean([task.eta_cog for task in baseline_first if task.success])
    netron_eta = mean([task.eta_cog for task in netron_first if task.success])
    return {
        "k": benchmark.k,
        "attempt_grid": grid,
        "plain_retry_cumulative_success": [round(value, 6) for value in baseline_rates],
        "netron_cumulative_success": [round(value, 6) for value in netron_rates],
        "plain_retry_discovery_attempts": baseline_attempts,
        "netron_discovery_attempts": netron_attempts,
        "plain_retry_mean_attempt": round(mean(baseline_attempts), 6),
        "netron_mean_attempt": round(mean(netron_attempts), 6),
        "attempt_speedup": round(ratio(mean(baseline_attempts), mean(netron_attempts)), 6),
        "plain_retry_tokens": baseline_tokens,
        "netron_tokens": netron_tokens,
        "token_speedup": round(ratio(baseline_tokens, netron_tokens), 6),
        "plain_retry_eta_cog": round(baseline_eta, 6),
        "netron_eta_cog": round(netron_eta, 6),
        "eta_cog_gain": round(netron_eta - baseline_eta, 6),
    }


def retention_curve(baseline: ArmState, netron: ArmState) -> dict[str, Any]:
    baseline_points = baseline.retention
    netron_points = netron.retention
    baseline_regressions = sum(point.regressions for point in baseline_points)
    netron_regressions = sum(point.regressions for point in netron_points)
    return {
        "tasks": [point.task_index for point in baseline_points],
        "plain_retry_historical_gate_pass_rate": [
            round(point.historical_gate_pass_rate, 6) for point in baseline_points
        ],
        "plain_retry_candidate_gate_pass_rate": [
            round(point.candidate_gate_pass_rate, 6) for point in baseline_points
        ],
        "netron_historical_gate_pass_rate": [
            round(point.historical_gate_pass_rate, 6) for point in netron_points
        ],
        "netron_candidate_gate_pass_rate": [
            round(point.candidate_gate_pass_rate, 6) for point in netron_points
        ],
        "plain_retry_checked_workspaces": [point.checked_workspaces for point in baseline_points],
        "netron_checked_workspaces": [point.checked_workspaces for point in netron_points],
        "plain_retry_regressions": baseline_regressions,
        "netron_regressions": netron_regressions,
        "zero_regression": all(point.zero_regression for point in (*baseline_points, *netron_points)),
    }


def repeat_curve(baseline: ArmState, netron: ArmState) -> dict[str, Any]:
    pairs = [
        (baseline_task, netron_task)
        for baseline_task, netron_task in zip(
            baseline.tasks,
            netron.tasks,
            strict=True,
        )
        if not baseline_task.first_seen
    ]
    plain_tokens = [task.tokens for task, _ in pairs]
    netron_tokens = [netron_task.tokens for _, netron_task in pairs]
    plain_attempts = [len(task.attempts) for task, _ in pairs]
    netron_attempts = [len(netron_task.attempts) for _, netron_task in pairs]
    plain_rollbacks = [task.rollbacks for task, _ in pairs]
    netron_rollbacks = [netron_task.rollbacks for _, netron_task in pairs]
    plain_eta = [task.eta_cog for task, _ in pairs]
    netron_eta = [netron_task.eta_cog for _, netron_task in pairs]
    return {
        "tasks": [task.task_index for task, _ in pairs],
        "plain_retry_tokens": plain_tokens,
        "netron_tokens": netron_tokens,
        "plain_retry_attempts": plain_attempts,
        "netron_attempts": netron_attempts,
        "plain_retry_rollbacks": plain_rollbacks,
        "netron_rollbacks": netron_rollbacks,
        "plain_retry_eta_cog": [round(value, 6) for value in plain_eta],
        "netron_eta_cog": [round(value, 6) for value in netron_eta],
        "netron_reflex_hits": [netron_task.reflex_hit for _, netron_task in pairs],
        "token_speedup": round(ratio(sum(plain_tokens), sum(netron_tokens)), 6),
        "attempt_speedup": round(ratio(mean(plain_attempts), mean(netron_attempts)), 6),
        "rollbacks_avoided": sum(plain_rollbacks) - sum(netron_rollbacks),
        "eta_cog_gain": round(mean(netron_eta) - mean(plain_eta), 6),
    }
