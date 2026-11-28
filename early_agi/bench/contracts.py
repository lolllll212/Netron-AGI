"""Shared contracts for the offline/live A/B patch-router seam.

The benchmark never imports a model client on the default offline path. A live
OpenAI-compatible endpoint can be plugged in later by implementing
:class:`PatchRouter`; both benchmark arms receive the same ``PatchRequest``
schema, so only the router changes when moving from deterministic fixtures to a
local model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "FileEdit",
    "PatchRequest",
    "PatchResponse",
    "PatchRouter",
    "is_safe_relative",
    "validated_edits",
]

_SYSTEM_PROMPT = """You are a code-repair candidate generator for an A/B benchmark.

Return JSON only, shaped exactly as:
{"edits": [{"path": "relative/path.py", "content": "FULL FILE CONTENT"}], "rationale": "short reason"}

Rules:
- Edit only paths listed in allowed_paths. Never edit tests or conftest.py.
- Return complete file contents, never diffs or patches.
- Use active_constraints and failure_constraints as hard requirements.
- Keep the change minimal, syntax-valid, and locally testable.
"""


@dataclass(frozen=True, slots=True)
class FileEdit:
    """One full-file replacement."""

    path: str
    content: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": len(self.content.encode("utf-8"))}


@dataclass(frozen=True, slots=True)
class PatchRequest:
    """One deterministic model turn shared by the baseline and Netron arms."""

    fixture_id: str
    task: str
    attempt: int
    orchestrator: str
    files: Mapping[str, str]
    test_targets: tuple[str, ...]
    active_constraints: tuple[str, ...] = ()
    failure_constraints: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        """Return OpenAI-compatible chat messages for the current request."""
        payload = {
            "fixture_id": self.fixture_id,
            "task": self.task,
            "attempt": self.attempt,
            "orchestrator": self.orchestrator,
            "test_targets": list(self.test_targets),
            "active_constraints": list(self.active_constraints),
            "failure_constraints": list(self.failure_constraints),
            "allowed_paths": list(self.allowed_paths),
            "files": dict(self.files),
        }
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]


@dataclass(frozen=True, slots=True)
class PatchResponse:
    """A router proposal plus deterministic/provider-reported usage."""

    edits: tuple[FileEdit, ...]
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return int(self.prompt_tokens + self.completion_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "edits": [edit.as_dict() for edit in self.edits],
            "rationale": self.rationale,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens": self.tokens,
            "latency_ms": round(float(self.latency_ms), 3),
        }


class PatchRouter(Protocol):
    """The only model-facing seam; offline and live routers are interchangeable."""

    name: str
    offline: bool

    def propose(self, request: PatchRequest) -> PatchResponse: ...


def is_safe_relative(path: str) -> bool:
    """Reject absolute paths, drive letters, and parent traversal."""
    raw = str(path).replace("\\", "/")
    if not raw or raw.startswith(("/", "\\")):
        return False
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    return bool(parts) and ".." not in parts and ":" not in parts[0]


def validated_edits(
    edits: Sequence[FileEdit],
    allowed_paths: Sequence[str],
) -> tuple[FileEdit, ...]:
    """Return edits only when every path is explicitly allowlisted."""
    allowed = set(allowed_paths)
    checked: list[FileEdit] = []
    for edit in edits:
        if not isinstance(edit.path, str) or not isinstance(edit.content, str):
            raise ValueError("edit path and content must be strings")
        if not is_safe_relative(edit.path) or edit.path not in allowed:
            raise ValueError(f"edit path is not allowlisted: {edit.path!r}")
        checked.append(edit)
    return tuple(checked)
