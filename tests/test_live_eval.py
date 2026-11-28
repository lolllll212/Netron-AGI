"""Entire live runner exercised through mocked stdlib HTTP only."""

import io
import json
from unittest.mock import patch

from early_agi.bench.fixtures import builtin_fixtures
from early_agi.bench.live_eval import evaluation_pack, run_live_evaluation


def test_eight_tasks_and_mocked_three_arm_ledger():
    pack = evaluation_pack()
    assert len(pack) == len({f.id for f in pack}) == 8
    fixture = builtin_fixtures()[0]
    prompts = []

    def transport(request, **kwargs):
        if request.full_url.endswith("/models"):
            return io.BytesIO(b'{"data":[{"id":"fast"},{"id":"deep"}]}')
        payload = json.loads(request.data)
        prompts.append(payload)
        content = {"edits": [{"path": e.path, "content": e.content} for e in fixture.fix_candidate.edits]}
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                }
            ).encode()
        )

    with patch("urllib.request.urlopen", side_effect=transport), patch("time.sleep"):
        report = run_live_evaluation(
            endpoint="https://mock/v1",
            api_key="never-publish",
            model="fast",
            deep_model="deep",
            fixtures=[fixture],
            verify_models=True,
        )
    assert set(report["arms"]) == {"baseline", "netron_plastic", "netron_race"}
    assert "never-publish" not in json.dumps(report)
    for payload in prompts:
        user = json.loads(payload["messages"][1]["content"])
        assert user["active_constraints"] == []
        assert all(not path.startswith("tests/") and path != "conftest.py" for path in user["files"])
        assert user["fixture_id"] == "task"
    for arm in report["arms"].values():
        assert arm["solve_rate"] == 1
        assert arm["tokens"] > 0
        assert arm["rate_limit_retries"] == 0
        assert arm["races"] == 0
    assert report["arms"]["netron_race"]["second_pass_reflex_hits"] == 1


def test_mocked_stall_uses_bounded_parallel_workers_and_accounts_all_requests():
    from early_agi.bench.fixtures import hard_stall_fixtures

    fixture = hard_stall_fixtures()[0]

    def transport(request, **kwargs):
        payload = json.loads(request.data)
        # A scripted executor fixates on the projection-only repair. The deep
        # candidate can repair both invariants. All generation is mocked HTTP.
        candidate = fixture.fix_candidate if payload["model"] == "deep" else fixture.candidates[0]
        content = {"edits": [{"path": e.path, "content": e.content} for e in candidate.edits]}
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                }
            ).encode()
        )

    with patch("urllib.request.urlopen", side_effect=transport), patch("time.sleep"):
        report = run_live_evaluation(
            endpoint="https://mock/v1",
            api_key="mock",
            model="fast",
            deep_model="deep",
            fixtures=[fixture],
            max_workers=8,
        )
    assert report["budgets"]["max_workers"] == 2
    race = report["arms"]["netron_race"]
    assert race["races"] == 1
    assert race["solve_rate"] == 1
    assert race["requests_sent"] == 4
    assert race["tokens"] == 800
    assert race["rollbacks"] == 3
    assert race["second_pass_reflex_hits"] == 1


def test_budget_exhaustion_cannot_issue_http():
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not send")) as send:
        report = run_live_evaluation(
            endpoint="https://mock/v1",
            api_key="mock",
            model="fast",
            deep_model="deep",
            fixtures=[builtin_fixtures()[0]],
            token_budget=1,
        )
    assert send.call_count == 0
    assert all(arm["tokens"] == 0 and arm["solve_rate"] == 0 for arm in report["arms"].values())


def test_cli_live_dispatch_requires_opt_in_and_writes_ledger(tmp_path, monkeypatch):
    import pytest

    from early_agi.bench.cli import main

    target = tmp_path / "summary.json"
    with pytest.raises(SystemExit):
        main(["--live-eval", "--deep-model", "deep", "--output", str(target)])
    monkeypatch.setenv("NVIDIA_API_KEY", "mock-key")
    with patch(
        "early_agi.bench.live_eval.run_live_evaluation",
        return_value={"offline": False, "status": "completed"},
    ) as run:
        assert (
            main(
                [
                    "--live-eval",
                    "--allow-live",
                    "--model",
                    "fast",
                    "--deep-model",
                    "deep",
                    "--endpoint",
                    "https://mock/v1",
                    "--output",
                    str(target),
                ]
            )
            == 0
        )
    assert json.loads(target.read_text()) == {"offline": False, "status": "completed"}
    assert run.call_args.kwargs["max_workers"] == 2
    assert run.call_args.kwargs["rpm_limit"] == 40


def test_live_report_uses_external_scores_and_honest_reflex_telemetry():
    fixture = builtin_fixtures()[0]

    def transport(request, **kwargs):
        edits = [{"path": e.path, "content": e.content} for e in fixture.fix_candidate.edits]
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps({"edits": edits})}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                }
            ).encode()
        )

    with patch("urllib.request.urlopen", side_effect=transport), patch("time.sleep"):
        report = run_live_evaluation(
            endpoint="https://mock/v1", api_key="mock", model="fast", deep_model="deep", fixtures=[fixture]
        )
    assert report["schema_version"] == 3
    for arm in report["arms"].values():
        assert arm["mean_eta_cog"] == arm["overall_mean_eta_cog"]
        assert "first_pass_mean_eta_cog" in arm
        for task in arm["tasks"]:
            assert task["verified_task_bits"] == 6
            for attempt in task["attempts"]:
                assert attempt["directive"] == "CRYSTALLIZE"
                assert attempt["verified_drop_bits"] == 6
                assert "governance" in attempt
                assert not any(not c.strip() for c in attempt["constraints"])
                if task["preflight"] is None:
                    assert attempt["preflight_probe"] is False
    for name in ("netron_plastic", "netron_race"):
        assert report["arms"][name]["second_pass_zero_token_replay"]
        assert report["arms"][name]["second_pass_token_speedup"] is None
