from __future__ import annotations

import pytest

from early_agi.core.kernel import ShiftArtifact
from early_agi.llm.client import LLMReply, extract_json
from early_agi.llm.router import (
    DualTierRouter,
    RouterError,
    ScriptedRouter,
    TurnProposal,
    parse_shift,
    parse_turn,
)


def test_parse_turn_rejects_unsafe_paths() -> None:
    proposal = parse_turn(
        {
            "edits": [
                {"path": "../escape.py", "content": "x"},
                {"path": "/abs.py", "content": "x"},
                {"path": "ok.py", "content": "x"},
            ],
            "rationale": "r",
        }
    )
    assert proposal.paths == ("ok.py",)
    assert len(proposal.rejected) == 2


def test_parse_turn_accepts_a_single_edit_object() -> None:
    assert parse_turn({"path": "a.py", "content": "x = 1\n"}).paths == ("a.py",)


def test_parse_shift_rejects_bad_artifacts() -> None:
    with pytest.raises(RouterError):
        parse_shift({"kind": "patch", "payload": {"diff": "..."}})
    with pytest.raises(RouterError):
        parse_shift({"kind": "decomposition", "payload": {"steps": ["only one"]}})


def test_extract_json_handles_fenced_and_noisy_output() -> None:
    assert extract_json('Sure!\n```json\n{"edits": []}\n```') == {"edits": []}
    assert extract_json('prose {"a": 1} trailing') == {"a": 1}


class FakeClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.seen: list[dict[str, str]] | None = None

    def chat_json(self, messages: list[dict[str, str]], **kwargs: object) -> tuple[object, LLMReply]:
        self.seen = messages
        return self.payload, LLMReply(text="", model="fake", usage={"total_tokens": 12}, latency_ms=3.0)


def test_dual_tier_router_parses_file_edits() -> None:
    client = FakeClient({"edits": [{"path": "a.py", "content": "x = 1\n"}], "rationale": "fix"})
    router = DualTierRouter(client, client)  # type: ignore[arg-type]

    proposal = router.execute({"task": "t"})

    assert isinstance(proposal, TurnProposal)
    assert proposal.tokens == 12
    assert proposal.paths == ("a.py",)
    assert client.seen is not None and client.seen[0]["role"] == "system"


def test_scripted_router_cycles_then_falls_back() -> None:
    router = ScriptedRouter(
        executions=[{"edits": [], "rationale": "noop"}],
        shifts=[{"kind": "invariant", "payload": {"check": "True"}}],
    )
    assert router.execute({}).rationale == "noop"
    assert router.execute({}).rationale.startswith("scripted")
    assert isinstance(router.reason({}), ShiftArtifact)
    assert router.calls == ["execute", "execute", "reason"]
