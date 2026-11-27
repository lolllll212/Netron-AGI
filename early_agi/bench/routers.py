"""Patch routers for the A/B benchmark.

`OfflineFixtureRouter` is the default and never touches the network. Both arms
use the same `propose(PatchRequest)` seam, so `OpenAICompatiblePatchRouter` can
later point the exact same fixtures and metrics at a local OpenAI-compatible
endpoint without changing the benchmark runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .contracts import FileEdit, PatchRequest, PatchResponse, validated_edits
from .fixtures import BugFixture, builtin_fixtures

__all__ = [
    "LiveRouterDisabled",
    "OfflineFixtureRouter",
    "OpenAICompatiblePatchRouter",
]


class LiveRouterDisabled(RuntimeError):
    """Raised when a live router is selected without explicit opt-in."""


class OfflineFixtureRouter:
    """Deterministic stand-in for a local model; no API or network calls."""

    name = "offline_fixture_v1"
    offline = True

    def __init__(
        self,
        fixtures: Sequence[BugFixture] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> None:
        selected = tuple(fixtures or builtin_fixtures())
        self._fixtures = {fixture.id: fixture for fixture in selected}
        self.max_tokens = None if max_tokens is None else max(0, int(max_tokens))

    def propose(self, request: PatchRequest) -> PatchResponse:
        fixture = self._fixtures.get(request.fixture_id)
        if fixture is None:
            raise KeyError(f"unknown benchmark fixture {request.fixture_id!r}")
        constraints = tuple(dict.fromkeys([*request.active_constraints, *request.failure_constraints]))
        probe_guided = request.orchestrator == "netron_gauge" and bool(constraints)
        if probe_guided:
            candidate = fixture.fix_candidate
        else:
            index = min(max(1, request.attempt) - 1, len(fixture.candidates) - 1)
            candidate = fixture.candidates[index]
        edits = validated_edits(candidate.edits, request.allowed_paths)
        messages = request.messages()
        prompt_chars = sum(len(message["content"]) for message in messages)
        completion_chars = sum(len(edit.path) + len(edit.content) for edit in edits)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, completion_chars // 4)
        uncapped_tokens = prompt_tokens + completion_tokens
        raw: dict[str, Any] = {
            "candidate": candidate.name,
            "constraints_used": list(constraints),
            "probe_guided": probe_guided,
            "offline": True,
        }
        if self.max_tokens is not None:
            cap = self.max_tokens
            raw["token_cap"] = cap
            raw["uncapped_tokens"] = uncapped_tokens
            if uncapped_tokens > cap:
                prompt_tokens = min(prompt_tokens, cap // 2)
                completion_tokens = max(0, cap - prompt_tokens)
        latency_ms = round(1.0 + 0.01 * prompt_tokens + 0.02 * completion_tokens, 3)
        rationale = candidate.rationale
        if probe_guided:
            rationale = f"{rationale}; guided by {len(constraints)} distilled constraint(s)"
        return PatchResponse(
            edits=edits,
            rationale=rationale,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw=raw,
        )


@dataclass(slots=True)
class OpenAICompatiblePatchRouter:
    """Opt-in live seam using the repository's stdlib chat client."""

    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_s: float = 90.0
    temperature: float = 0.0
    enabled: bool = False

    name: ClassVar[str] = "openai_compatible_live"

    @property
    def offline(self) -> bool:
        return not self.enabled

    def propose(self, request: PatchRequest) -> PatchResponse:
        if not self.enabled:
            raise LiveRouterDisabled("live benchmark routing requires enabled=True")
        from ..llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=0,
            temperature=self.temperature,
        )
        data, reply = client.chat_json(
            request.messages(),
            json_mode=True,
            timeout=self.timeout_s,
        )
        edits = _edits_from_payload(data, request.allowed_paths)
        usage = dict(reply.usage)
        prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"))
        completion_tokens = _nonnegative_int(usage.get("completion_tokens"))
        if prompt_tokens + completion_tokens == 0:
            prompt_chars = sum(len(message["content"]) for message in request.messages())
            completion_chars = sum(len(edit.path) + len(edit.content) for edit in edits)
            prompt_tokens = max(1, prompt_chars // 4)
            completion_tokens = max(1, completion_chars // 4)
        rationale = data.get("rationale", "") if isinstance(data, Mapping) else ""
        return PatchResponse(
            edits=edits,
            rationale=str(rationale)[:500],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=float(reply.latency_ms),
            raw={
                "model": reply.model,
                "usage": usage,
                "offline": False,
            },
        )


def _edits_from_payload(data: Any, allowed_paths: Sequence[str]) -> tuple[FileEdit, ...]:
    if not isinstance(data, Mapping) or not isinstance(data.get("edits"), list):
        raise ValueError("live router response must be an object with an edits list")
    edits: list[FileEdit] = []
    for item in data["edits"]:
        if not isinstance(item, Mapping) or "path" not in item or "content" not in item:
            raise ValueError("each live edit needs path and content")
        edits.append(FileEdit(str(item["path"]), str(item["content"])))
    return validated_edits(edits, allowed_paths)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
