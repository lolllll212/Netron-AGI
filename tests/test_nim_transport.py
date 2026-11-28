"""Provider compatibility without any network traffic."""

import io
import json
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

from early_agi.config import Settings
from early_agi.llm.client import LLMError, OpenAICompatibleClient, sanitize_messages
from early_agi.llm.pacing import RequestPacer, retry_after


def completion():
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_after_and_telemetry(status):
    headers = Message()
    headers["Retry-After"] = "3"
    error = urllib.error.HTTPError("https://mock/v1", status, "retry", headers, io.BytesIO(b"busy"))
    with (
        patch("urllib.request.urlopen", side_effect=[error, completion()]) as send,
        patch("time.sleep") as sleep,
    ):
        client = OpenAICompatibleClient("model", base_url="https://mock/v1")
        client.chat([{"role": "developer", "content": "JSON"}])
    assert sleep.call_args.args[0] >= 3
    assert client.rate_limit_retries == (1 if status == 429 else 0)
    assert client.requests_sent == 2
    payload = json.loads(send.call_args.args[0].data)
    assert payload["messages"] == [{"role": "system", "content": "JSON"}]


def test_no_retry_on_bad_request_and_reject_tools():
    with pytest.raises(LLMError, match="unsupported"):
        sanitize_messages([{"role": "tool", "content": "result"}])
    error = urllib.error.HTTPError("https://mock", 400, "bad", {}, io.BytesIO(b"bad"))
    with patch("urllib.request.urlopen", side_effect=error) as send, pytest.raises(LLMError):
        OpenAICompatibleClient("model").chat([{"role": "user", "content": "x"}])
    assert send.call_count == 1


def test_model_discovery_and_env_aliases():
    settings = Settings.from_env(
        {"OPENAI_BASE_URL": "https://mock/v1", "NVIDIA_API_KEY": "secret"}, load_dotenv=False
    )
    assert settings.base_url == "https://mock/v1"
    assert settings.api_key == "secret"
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"data":[{"id":"model"}]}')):
        assert OpenAICompatibleClient("model").verify_models(["model"]) == ("model",)
    assert retry_after("invalid") == 0
    assert retry_after("Wed, 01 Jan 2020 00:00:00 GMT") == 0


def test_shared_pacer_reserves_every_start():
    pacer = RequestPacer(40)
    with patch("time.monotonic", return_value=100), patch("time.sleep") as sleep:
        pacer.wait()
        pacer.wait()
        pacer.wait()
    assert [call.args[0] for call in sleep.call_args_list] == [1.5, 3.0]


def test_retry_exhaustion_is_bounded_and_model_preflight_rejects_missing():
    def exhausted(*args, **kwargs):
        raise urllib.error.HTTPError("https://mock", 429, "busy", {}, io.BytesIO(b"busy"))

    with patch("urllib.request.urlopen", side_effect=exhausted) as send, patch("time.sleep"):
        client = OpenAICompatibleClient("model", max_retries=2)
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "x"}])
    assert send.call_count == 3
    assert client.rate_limit_retries == 2
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"data":[]}')), pytest.raises(LLMError):
        client.verify_models(["unavailable"])
