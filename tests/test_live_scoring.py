"""Common live scoring and terminal telemetry, independent of Gauge control math."""

from copy import deepcopy

import pytest

from early_agi.bench.metrics import live_efficiency_summary, score_live_task
from early_agi.bench.rescore_live import rescore_report


def attempt(*, tokens=1000, latency=2000, tools=5, passed=True, reflex=False):
    return {
        "passed": passed,
        "error": "",
        "tokens": tokens,
        "latency_ms": latency,
        "eta_cog": 0.29,
        "verified_drop_bits": 3,
        "directive": "EPISTEMIC_PROBE",
        "preflight_probe": True,
        "reflex": reflex,
        "constraints": ["", "  ", " assert x ", "assert x"],
        "gates": {"syntax_valid": "pass", "tests_passing": "pass" if passed else "fail"},
        "cost": {"tool_cost": tools, "latency_ms": latency, "pruned_branches": 10},
    }


def task(*attempts, preflight=None, first_seen=True):
    return {
        "attempts": list(attempts),
        "success": any(a["passed"] for a in attempts),
        "first_seen": first_seen,
        "preflight": preflight,
        "tokens": sum(a["tokens"] for a in attempts),
        "eta_cog": 0.29,
        "eta_cog_curve": [0.29],
        "reflex_hit": any(a["reflex"] for a in attempts),
    }


def test_identical_resources_have_identical_scores_across_controllers():
    baseline = attempt()
    baseline.update(eta_cog=0.98, verified_drop_bits=6, directive="SHIFT_COORDINATE", preflight_probe=False)
    normal = task(baseline)
    netron = task(attempt(), preflight={"probe": True})
    replay = task(attempt(reflex=True))
    results = [score_live_task(t) for t in (normal, netron, replay)]
    assert {r["eta_cog"] for r in results} == {round(6 / (1 + 0.2 + 5), 6)}
    assert all(r["verified_task_bits"] == 6 for r in results)
    assert all(r["attempts"][0]["directive"] == "CRYSTALLIZE" for r in results)
    assert results[1]["attempts"][0]["preflight_probe"]
    assert not results[2]["attempts"][0]["preflight_probe"]
    assert results[1]["attempts"][0]["governance"]["verified_drop_bits"] == 3
    assert results[1]["attempts"][0]["governance"]["directive"] == "EPISTEMIC_PROBE"
    assert results[1]["attempts"][0]["constraints"] == ["assert x"]
    assert netron["attempts"][0]["constraints"][0] == ""  # no controller mutation


def test_task_score_charges_failed_attempts_once_and_credits_only_full_completion():
    failed = attempt(tokens=500, passed=False)
    solved = attempt(tokens=1000)
    scored = score_live_task(task(failed, solved))
    assert scored["external_cost"]["weighted_cost"] == pytest.approx(1.5 + 0.4 + 10)
    assert scored["eta_cog"] == round(6 / 11.9, 6)
    assert scored["verified_task_bits"] == 6
    assert scored["eta_cog_curve"][0] == 0
    assert score_live_task(task(failed))["eta_cog"] == 0
    not_verified = attempt()
    not_verified["gates"]["tests_passing"] = "unknown"
    assert score_live_task(task(not_verified))["verified_task_bits"] == 0


def test_reflex_gains_are_in_overall_mean_and_no_invented_speedup():
    first = score_live_task(task(attempt()))
    second = score_live_task(task(attempt(tokens=0, latency=12, reflex=True), first_seen=False))
    stats = live_efficiency_summary([first, second])
    assert second["eta_cog"] > first["eta_cog"]
    assert stats["first_pass_mean_eta_cog"] == first["eta_cog"]
    assert stats["overall_mean_eta_cog"] == round((first["eta_cog"] + second["eta_cog"]) / 2, 6)
    assert stats["mean_eta_cog"] == stats["overall_mean_eta_cog"]
    assert stats["second_pass_token_speedup"] is None
    assert stats["second_pass_zero_token_replay"]
    assert not second["attempts"][0]["preflight_probe"]


def test_race_score_charges_aggregate_branch_cost_not_only_winner_latency():
    aggregate = attempt(tokens=3000, latency=100, tools=15)
    aggregate["cost"]["latency_ms"] = 6000  # aggregate branch resource accounting
    scored = score_live_task(task(aggregate))
    assert scored["external_cost"]["weighted_cost"] == pytest.approx(3 + 0.6 + 15)
    assert scored["eta_cog"] == round(6 / 18.6, 6)


def test_rescoring_is_offline_idempotent_and_preserves_evidence(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call HTTP"))
    original = {
        "schema_version": 2,
        "offline": False,
        "valid_comparison": False,
        "arms": {"netron": {"tasks": [task(attempt())], "tokens": 1000}},
    }
    before = deepcopy(original)
    scored = rescore_report(original)
    assert original == before
    assert scored == rescore_report(scored)
    assert scored["valid_comparison"] is False
    assert scored["arms"]["netron"]["tokens"] == 1000
    assert scored["schema_version"] == 3
    assert "no new model calls" in scored["scoring_provenance"]["method"]
