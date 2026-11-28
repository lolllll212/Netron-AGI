"""Offline regression tests for visible, bounded and checkpointed live runs."""

import io
import json
import threading
from unittest.mock import patch

import pytest

from early_agi.bench.fixtures import builtin_fixtures
from early_agi.bench.live_eval import run_live_evaluation
from early_agi.bench.live_monitor import LiveMonitor, checkpoint


def options(tmp_path):
    return dict(
        endpoint="https://mock/v1",
        api_key="nvapi-secret",
        model="fast",
        deep_model="deep",
        fixtures=builtin_fixtures()[:2],
        checkpoint_path=tmp_path / "report.json",
    )


def completion():
    fixture = builtin_fixtures()[0]
    edits = [{"path": e.path, "content": e.content} for e in fixture.fix_candidate.edits]
    return io.BytesIO(
        json.dumps(
            {
                "choices": [{"message": {"content": json.dumps({"edits": edits})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }
        ).encode()
    )


def test_timeouts_stop_remaining_arms_and_preserve_redacted_detail(tmp_path):
    logs = []
    with patch("urllib.request.urlopen", side_effect=TimeoutError("read timed out nvapi-secret")) as send:
        report = run_live_evaluation(**options(tmp_path), progress=logs.append)
    assert send.call_count == 2  # Default: no hidden retries per candidate.
    assert report["status"] == "transport_failed"
    assert not report["valid_comparison"]
    assert set(report["arms"]) == {"baseline"}
    assert report["arms"]["baseline"]["completed_tasks"] == 1
    assert report["arms"]["baseline"]["planned_tasks"] == 4
    assert "timed out" in report["errors"][0]["message"]
    assert report["arms"]["baseline"]["unresolved_token_reservations"] > 0
    stored = (tmp_path / "report.json").read_text()
    assert json.loads(stored) == report
    assert "nvapi-secret" not in stored + str(logs)
    assert any("circuit_open" in message for message in logs)


def test_interrupt_retains_completed_task_and_marks_partial_report(tmp_path):
    with patch("urllib.request.urlopen", side_effect=[completion(), KeyboardInterrupt()]):
        report = run_live_evaluation(**options(tmp_path), arms=["baseline"], passes=1)
    assert report["status"] == "interrupted"
    assert not report["valid_comparison"]
    assert report["arms"]["baseline"]["tasks"][0]["success"]
    assert report["arms"]["baseline"]["completed_tasks"] == 1
    assert report["current_task"]["index"] == 2
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "interrupted"


def test_task_checkpoint_exists_before_next_request(tmp_path):
    calls = 0

    def http(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completion()
        saved = json.loads((tmp_path / "report.json").read_text())
        assert saved["arms"]["baseline"]["completed_tasks"] == 1
        assert saved["arms"]["baseline"]["tasks"][0]["success"]
        raise KeyboardInterrupt

    with patch("urllib.request.urlopen", side_effect=http):
        run_live_evaluation(**options(tmp_path), arms=["baseline"], passes=1)


def test_heartbeat_is_visible_without_extra_provider_requests():
    seen = threading.Event()
    messages = []

    def progress(message):
        messages.append(message)
        if "request_waiting" in message:
            seen.set()

    monitor = LiveMonitor(api_key="secret", progress=progress, heartbeat=0.01)
    with monitor.waiting(model="fast", timeout=180):
        assert seen.wait(timeout=2)
    assert any("request_end" in message for message in messages)


def test_smoke_cli_is_exactly_one_task_one_pass_one_arm(tmp_path, monkeypatch):
    from early_agi.bench.cli import main

    monkeypatch.setenv("NVIDIA_API_KEY", "mock")
    with patch("early_agi.bench.live_eval.run_live_evaluation", return_value={"status": "completed"}) as run:
        assert (
            main(
                [
                    "--live-eval",
                    "--allow-live",
                    "--model",
                    "fast",
                    "--deep-model",
                    "deep",
                    "--smoke",
                    "--output",
                    str(tmp_path / "smoke.json"),
                ]
            )
            == 0
        )
    assert run.call_args.kwargs["k"] == 1
    assert run.call_args.kwargs["task_limit"] == 1
    assert run.call_args.kwargs["passes"] == 1
    assert run.call_args.kwargs["arms"] == ["baseline"]
    assert run.call_args.kwargs["timeout"] == 180
    assert run.call_args.kwargs["request_retries"] == 0


def test_checkpoint_failure_does_not_destroy_last_good_report(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint(path, {"status": "running"})
    with pytest.raises(TypeError):
        checkpoint(path, {"bad": object()})
    assert json.loads(path.read_text()) == {"status": "running"}


def test_preflight_failure_is_checkpointed_without_starting_arms(tmp_path):
    with patch("urllib.request.urlopen", side_effect=TimeoutError("model list timed out")) as send:
        report = run_live_evaluation(**options(tmp_path), verify_models=True)
    assert send.call_count == 1
    assert report["status"] == "failed"
    assert report["arms"] == {}
    assert "model list timed out" in report["errors"][0]["message"]
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("status,code", [("transport_failed", 1), ("interrupted", 130)])
def test_cli_does_not_return_success_for_incomplete_runs(tmp_path, monkeypatch, status, code):
    from early_agi.bench.cli import main

    monkeypatch.setenv("NVIDIA_API_KEY", "mock")
    with patch("early_agi.bench.live_eval.run_live_evaluation", return_value={"status": status}):
        assert (
            main(
                [
                    "--live-eval",
                    "--allow-live",
                    "--deep-model",
                    "deep",
                    "--smoke",
                    "--output",
                    str(tmp_path / "report.json"),
                ]
            )
            == code
        )
