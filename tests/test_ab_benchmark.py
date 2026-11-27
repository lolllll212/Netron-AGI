"""Offline A/B repair benchmark: real fixtures, three curves, and router seam."""

from __future__ import annotations

import json
import socket
import urllib.request
from importlib.resources import files
from pathlib import Path

import pytest

from early_agi.bench import (
    LiveRouterDisabled,
    OfflineFixtureRouter,
    OpenAICompatiblePatchRouter,
    PatchRequest,
    builtin_fixtures,
    evaluate_baseline_gates,
)
from early_agi.bench.cli import main
from early_agi.bench.gates import apply_edits, write_workspace


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("offline A/B benchmark must not call external services")


def test_offline_ab_cli_three_curves_without_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr("early_agi.llm.client.OpenAICompatibleClient.chat_json", _forbidden)

    output = tmp_path / "ab-report.json"
    status = main(
        [
            "--tasks",
            "5",
            "--k",
            "3",
            "--timeout",
            "60",
            "--compact",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["offline"] is True
    assert report["network_calls"] == 0
    assert set(report["curves"]) == {
        "discovery@k",
        "zero_forgetting_retention",
        "repeat_task_reflex_speedup",
    }
    assert report["config"]["netron"] == {
        "controller": "netron_plastic_gauge",
        "program": "netron_plastic.gauge",
        "backend": "bounded_chi",
        "reflex_library": "ReflexLibrary",
        "gate": "GaugeGate(ast_check+pytest_run)",
    }
    assert report["summary"]["ok"] is True
    assert report["summary"]["baseline_solved"] == 5
    assert report["summary"]["netron_solved"] == 5
    assert report["summary"]["reflex_hits"] == 2
    assert report["summary"]["netron_tokens_saved"] > 0
    assert report["summary"]["netron_rollbacks_avoided"] > 0

    discovery = report["curves"]["discovery@k"]
    assert discovery["k"] == 3
    assert discovery["netron_cumulative_success"][0] > discovery["plain_retry_cumulative_success"][0]
    assert discovery["attempt_speedup"] > 1.0
    assert discovery["token_speedup"] > 1.0
    assert discovery["netron_eta_cog"] > 0.0

    retention = report["curves"]["zero_forgetting_retention"]
    assert len(retention["tasks"]) == 5
    assert retention["zero_regression"] is True
    assert retention["plain_retry_regressions"] == 0
    assert retention["netron_regressions"] == 0
    assert retention["netron_candidate_gate_pass_rate"][-1] == 1.0

    repeat = report["curves"]["repeat_task_reflex_speedup"]
    assert repeat["netron_tokens"] == [0, 0]
    assert repeat["netron_reflex_hits"] == [True, True]
    assert repeat["token_speedup"] > 1.0
    assert repeat["attempt_speedup"] > 1.0
    assert repeat["rollbacks_avoided"] > 0
    assert all(value > 0.0 for value in repeat["netron_eta_cog"])

    assert report["arms"]["plain_retry"]["tokens_spent"] > 0
    assert report["arms"]["netron"]["tokens_spent"] > 0
    assert report["arms"]["netron"]["preflight_probes"] >= 3
    assert report["arms"]["netron"]["reflex_compiles"] > 0
    for arm in report["arms"].values():
        for task in arm["tasks"]:
            for attempt in task["attempts"]:
                if attempt["passed"]:
                    assert attempt["eta_cog"] > 0.0


def test_builtin_fixtures_are_real_multifile_bugs(tmp_path: Path) -> None:
    for fixture in builtin_fixtures():
        root = write_workspace(fixture, tmp_path / fixture.id)
        assert len(fixture.fix_candidate.edits) >= 2
        assert fixture.test_targets
        assert set(fixture.editable_paths).isdisjoint(fixture.test_targets)

        before = evaluate_baseline_gates(root, fixture.test_targets, timeout_s=60.0)
        assert before.passed is False

        apply_edits(root, fixture.fix_candidate.edits, allowed_paths=fixture.editable_paths)
        after = evaluate_baseline_gates(root, fixture.test_targets, timeout_s=60.0)
        assert after.passed is True
        assert after.verdicts == {"syntax_valid": "pass", "tests_passing": "pass"}


def test_router_seam_builds_live_prompt_but_stays_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("early_agi.llm.client.OpenAICompatibleClient.chat_json", _forbidden)
    fixture = builtin_fixtures()[0]
    request = PatchRequest(
        fixture_id=fixture.id,
        task=fixture.task,
        attempt=1,
        orchestrator="netron_gauge",
        files=fixture.file_map,
        test_targets=fixture.test_targets,
        active_constraints=(fixture.probe_hint,),
        allowed_paths=fixture.editable_paths,
    )
    messages = request.messages()
    payload = json.loads(messages[-1]["content"])
    assert messages[0]["role"] == "system"
    assert payload["fixture_id"] == fixture.id
    assert payload["orchestrator"] == "netron_gauge"
    assert fixture.probe_hint in payload["active_constraints"]
    assert "tests/test_pricing.py" in payload["files"]
    assert "pricing/cart.py" in payload["allowed_paths"]

    offline = OfflineFixtureRouter((fixture,))
    response = offline.propose(request)
    assert response.raw["candidate"] == fixture.fix_candidate.name
    assert response.raw["probe_guided"] is True

    live = OpenAICompatiblePatchRouter(
        base_url="http://localhost:11434/v1",
        model="local-model",
        enabled=False,
    )
    assert live.offline is True
    with pytest.raises(LiveRouterDisabled):
        live.propose(request)


def test_packaged_plastic_program_matches_verified_example() -> None:
    resource = files("early_agi.bench").joinpath("netron_plastic.gauge")
    example_path = Path(__file__).parents[1] / "examples" / "netron_plastic.gauge"
    packaged = resource.read_text(encoding="utf-8")
    example = example_path.read_text(encoding="utf-8")
    assert packaged == example
    assert resource.read_bytes() == example_path.read_bytes()
    assert "ast_check(p.files)" in packaged
    assert "pytest_run(p.targets)" in packaged
