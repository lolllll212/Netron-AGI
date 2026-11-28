"""Metric curves and final JSON report for the A/B benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .records import ArmState, mean, ratio

__all__ = ["build_race_report", "build_report"]


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


def build_race_report(
    benchmark: Any,
    arms: Mapping[str, ArmState],
    *,
    race_state: ArmState,
    race_summaries: Sequence[Mapping[str, Any]],
    race_config: Any,
) -> dict[str, Any]:
    """Extend the untouched two-arm report with a deterministic netron_race arm."""
    report = build_report(benchmark, arms)
    baseline = arms["plain_retry"]
    summaries = [dict(item) for item in race_summaries]
    race_zero_regression = all(point.zero_regression for point in race_state.retention)
    report["config"]["netron_race"] = {
        "controller": "stall_gated_parallel_wal_branch_race",
        "program": "RACE_BRANCHES",
        "gate": "ast_syntax_gate+pytest_gate",
        "trigger": "volume_law_or_frustration",
        "router_name": str(race_config.router_name),
        "endpoint": str(race_config.endpoint),
        "model": str(race_config.model),
        "branches": list(race_config.branches),
        "gates": list(race_config.gates),
        "seed": int(race_config.seed),
        "max_tokens_per_branch": int(race_config.max_tokens_per_branch),
        "max_workers": int(race_config.max_workers),
        "executor": str(race_config.executor),
        "offline": True,
    }
    report["arms"]["netron_race"] = {
        **race_state.as_dict(),
        "controller": "stall_gated_parallel_wal_branch_race",
        "race_summaries": summaries,
    }
    winner_eta = [float((item.get("winner") or {}).get("eta_cog", 0.0) or 0.0) for item in summaries]
    winner_branches = [str(item.get("winner_branch_id", "")) for item in summaries]
    pruned = [int(item.get("pruned_branches", 0)) for item in summaries]
    tokens = [int(item.get("total_tokens", 0)) for item in summaries]
    lineage = [int(item.get("lineage_edges", 0)) for item in summaries]
    report["curves"]["branch_race"] = {
        "tasks": [
            task.task_index for task in race_state.tasks if task.preflight and "race" in task.preflight
        ],
        "winner_eta_cog": [round(value, 6) for value in winner_eta],
        "winner_branch_ids": winner_branches,
        "pruned_branches": pruned,
        "total_tokens": tokens,
        "lineage_edges": lineage,
        "green_winners": [str(item.get("winner_hash", "")) != "" for item in summaries],
        "executors": [str(item.get("executor", "")) for item in summaries],
        "sandbox_modes": [list(item.get("sandbox_modes", ())) for item in summaries],
        "network_calls": 0,
        "offline": True,
    }
    summary = report["summary"]
    summary["netron_race_solved"] = race_state.solved_count
    summary["netron_race_tokens_saved"] = baseline.tokens_spent - race_state.tokens_spent
    summary["netron_race_rollbacks_avoided"] = baseline.rollbacks - race_state.rollbacks
    summary["branch_race_pruned_branches"] = sum(pruned)
    summary["branch_race_lineage_edges"] = sum(lineage)
    summary["branch_race_tokens"] = sum(tokens)
    summary["zero_regression"] = bool(summary["zero_regression"] and race_zero_regression)
    summary["ok"] = bool(
        summary["ok"]
        and race_state.solved_count == len(benchmark.task_ids)
        and race_zero_regression
        and all(bool(item.get("ok", False)) for item in summaries)
        and all(int(item.get("network_calls", 1)) == 0 for item in summaries)
    )
    return report


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


# Live evaluation scoring is intentionally separate from build_report(): the
# deterministic two-arm report remains byte-compatible with its published lock.
LIVE_SCORING = {
    "name": "external_task_v1",
    "formula": "verified_task_bits / (0.001*tokens + 0.0001*latency_ms + tool_cost)",
    "verified_task_bits": 6.0,
    "required_gates": ["syntax_valid", "tests_passing"],
    "credit": "one fixed six-unit completion credit per fully verified task; zero for partial/failed tasks",
    "cost": "sum all candidate costs, including failures and pruned branches; exclude internal Gauge penalties",
    "latency": "provider latency plus standardized external grader charge; not end-to-end wall-clock speed",
    "mean_eta_cog_alias": "overall_mean_eta_cog",
}


def _live_closed(attempt: Mapping[str, Any]) -> bool:
    return bool(
        attempt.get("passed")
        and not attempt.get("error")
        and all(attempt.get("gates", {}).get(gate) == "pass" for gate in LIVE_SCORING["required_gates"])
    )


def _live_cost(attempts: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    tokens = sum(float(a.get("tokens", 0)) for a in attempts)
    latency = sum(float(a.get("cost", {}).get("latency_ms", a.get("latency_ms", 0))) for a in attempts)
    tools = sum(float(a.get("cost", {}).get("tool_cost", 0)) for a in attempts)
    return {
        "tokens": tokens,
        "latency_ms": latency,
        "tool_cost": tools,
        "weighted_cost": 0.001 * tokens + 0.0001 * latency + tools,
    }


def score_live_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a live task without mutating its internal controller evidence.

    Six units are a fixed benchmark completion convention, not a measurement of
    physical entropy or an assertion that the starting code failed both gates.
    The same resource model applies to every arm, including exact reflex replay.
    """
    import copy

    result = copy.deepcopy(dict(task))
    if "governance" not in result:
        result["governance"] = {
            "eta_cog": result.get("eta_cog", 0),
            "eta_cog_curve": result.get("eta_cog_curve", []),
        }
    preflight = result.get("preflight") or {}
    preflight["constraints"] = list(
        dict.fromkeys(str(c).strip() for c in preflight.get("constraints", []) if str(c).strip())
    )
    if result.get("preflight") is not None:
        result["preflight"] = preflight
    attempts = result.get("attempts", [])
    curve = []
    for index, attempt in enumerate(attempts):
        if "governance" not in attempt:
            attempt["governance"] = {
                key: attempt.get(key)
                for key in ("eta_cog", "verified_drop_bits", "directive", "preflight_probe")
            }
        attempt["constraints"] = list(
            dict.fromkeys(str(c).strip() for c in attempt.get("constraints", []) if str(c).strip())
        )
        attempt["preflight_probe"] = bool(index == 0 and not attempt.get("reflex") and preflight.get("probe"))
        closed = _live_closed(attempt)
        if closed:
            attempt["directive"] = "CRYSTALLIZE"
        bits = 6.0 if closed else 0.0
        attempt["verified_drop_bits"] = bits
        own_cost = _live_cost([attempt])
        attempt["external_cost"] = own_cost
        attempt["eta_cog"] = round(bits / max(1e-9, own_cost["weighted_cost"]), 6)
        prefix = attempts[: index + 1]
        prefix_bits = 6.0 if any(_live_closed(a) for a in prefix) else 0.0
        curve.append(round(prefix_bits / max(1e-9, _live_cost(prefix)["weighted_cost"]), 6))
    cost = _live_cost(attempts)
    bits = 6.0 if result.get("success") and any(_live_closed(a) for a in attempts) else 0.0
    result.update(
        {
            "scoring": "external_task_v1",
            "verified_task_bits": bits,
            "external_cost": cost,
            "eta_cog": round(bits / max(1e-9, cost["weighted_cost"]), 6),
            "eta_cog_curve": curve,
        }
    )
    return result


def live_efficiency_summary(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose first-pass, repeat-pass and overall means without dropping failures."""
    first = [t for t in tasks if t.get("first_seen")]
    second = [t for t in tasks if not t.get("first_seen")]
    overall = round(mean([float(t["eta_cog"]) for t in tasks]), 6)
    first_tokens = sum(int(t.get("tokens", 0)) for t in first)
    second_tokens = sum(int(t.get("tokens", 0)) for t in second)
    return {
        "mean_eta_cog": overall,
        "first_pass_mean_eta_cog": round(mean([float(t["eta_cog"]) for t in first]), 6),
        "overall_mean_eta_cog": overall,
        "second_pass_mean_eta_cog": round(mean([float(t["eta_cog"]) for t in second]), 6) if second else None,
        "overall_mean_governance_eta_cog": round(
            mean([float(t.get("governance", {}).get("eta_cog", 0)) for t in tasks]), 6
        ),
        "first_pass_tokens": first_tokens,
        "second_pass_tokens": second_tokens,
        "second_pass_token_speedup": first_tokens / second_tokens if second_tokens else None,
        "second_pass_zero_token_replay": bool(
            second and second_tokens == 0 and all(t.get("success") and t.get("reflex_hit") for t in second)
        ),
    }
