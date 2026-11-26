"""Dual-tier routing: Fast Executor for CRYSTALLIZE, Deep Reasoner for SHIFT.

Two hard rules, both enforced by the kernel rather than by prompt etiquette:

* the fast tier may only emit file edits;
* when coordinates must shift, edits are rejected outright and the deep tier
  must emit a decomposition, an invariant check, or a new operator.

The router is the producer; the kernel is the gatekeeper. Validation lives in
:mod:`early_agi.core.kernel` so a model response can never widen the loop's
invariants.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.cold_sink import canonical_json
from ..core.kernel import ShiftArtifact
from .client import OpenAICompatibleClient

__all__ = [
    "DualTierRouter",
    "Edit",
    "Router",
    "RouterError",
    "ScriptedRouter",
    "TurnProposal",
    "parse_shift",
    "parse_turn",
]

EXECUTOR_SYSTEM = """You are the FAST EXECUTOR of a code-repair kernel.

Emit at most a few surgical file edits that reduce the entropy of the task.
Rules:
- Never invent paths. Only edit files listed in "files".
- Return complete file contents, not diffs.
- Obey "writes_locked": when it is true, return {"edits": [], "rationale": "writes isolated"}.
- Respond with JSON only, shaped exactly:
{"edits": [{"path": "relative/path.py", "content": "..."}], "rationale": "..."}
"""

REASONER_SYSTEM = """You are the DEEP REASONER of a code-repair kernel.

Local patching has stalled or the coupling is volume-law. You may NOT patch.
Produce exactly one coordinate shift as JSON:
{"kind": "decomposition", "payload": {"steps": ["...", "..."]}, "rationale": "..."}
{"kind": "invariant", "payload": {"check": "<python expression that must stay true>"}, "rationale": "..."}
{"kind": "new_operator", "payload": {"name": "<identifier>", "source": "<python source defining it>"}, "rationale": "..."}
Choose the smallest artefact that actually disentangles the coupling.
"""


class RouterError(RuntimeError):
    """The model produced something the loop cannot use."""


class Router(Protocol):
    def execute(self, brief: Mapping[str, Any]) -> TurnProposal: ...

    def reason(self, brief: Mapping[str, Any]) -> ShiftArtifact: ...


@dataclass(frozen=True)
class Edit:
    path: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "content": self.content}


@dataclass(frozen=True)
class TurnProposal:
    edits: tuple[Edit, ...] = ()
    rationale: str = ""
    rejected: tuple[Mapping[str, Any], ...] = ()
    tokens: int = 0
    latency_ms: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(edit.path for edit in self.edits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "edits": [edit.as_dict() for edit in self.edits],
            "rationale": self.rationale,
            "rejected": list(self.rejected),
            "tokens": self.tokens,
        }


def _is_safe_relative(path: str) -> bool:
    if not path or path.startswith(("/", "\\")) or ":" in path.split("/")[0]:
        return False
    parts = [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]
    return bool(parts) and ".." not in parts


def parse_turn(data: Any, *, tokens: int = 0, latency_ms: float = 0.0) -> TurnProposal:
    """Validate a fast-tier response into a proposal."""
    if isinstance(data, list):
        data = {"edits": data}
    if not isinstance(data, Mapping):
        raise RouterError(f"executor returned {type(data).__name__}, expected an object")

    raw_edits = data.get("edits")
    if raw_edits is None and "path" in data and "content" in data:
        raw_edits = [data]
    raw_edits = raw_edits or []
    if not isinstance(raw_edits, Sequence) or isinstance(raw_edits, (str, bytes)):
        raise RouterError("'edits' must be a list")

    edits: list[Edit] = []
    rejected: list[Mapping[str, Any]] = []
    for item in raw_edits:
        if not isinstance(item, Mapping):
            rejected.append({"reason": "edit is not an object", "item": str(item)[:120]})
            continue
        path, content = item.get("path"), item.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            rejected.append({"reason": "edit needs string 'path' and 'content'", "path": path})
            continue
        if not _is_safe_relative(path):
            rejected.append({"reason": "path is not a safe relative path", "path": path})
            continue
        edits.append(Edit(path=path, content=content))

    rationale = data.get("rationale", "")
    return TurnProposal(
        edits=tuple(edits),
        rationale=str(rationale)[:2000],
        rejected=tuple(rejected),
        tokens=int(tokens),
        latency_ms=float(latency_ms),
        raw=dict(data),
    )


def parse_shift(data: Any) -> ShiftArtifact:
    """Validate a deep-tier response into a shift artefact."""
    if not isinstance(data, Mapping):
        raise RouterError(f"reasoner returned {type(data).__name__}, expected an object")
    artifact = ShiftArtifact.from_mapping(data)
    ok, detail = artifact.validate()
    if not ok:
        raise RouterError(f"rejected coordinate shift: {detail}")
    return artifact


class DualTierRouter:
    """Fast tier for execution, deep tier for reframing."""

    def __init__(self, fast: OpenAICompatibleClient, deep: OpenAICompatibleClient) -> None:
        self.fast = fast
        self.deep = deep

    @classmethod
    def from_settings(cls, settings: Any = None) -> DualTierRouter:
        from ..config import Settings

        settings = settings or Settings.from_env()
        return cls(
            OpenAICompatibleClient.from_settings("fast", settings),
            OpenAICompatibleClient.from_settings("deep", settings),
        )

    def execute(self, brief: Mapping[str, Any]) -> TurnProposal:
        data, reply = self.fast.chat_json(self._messages(EXECUTOR_SYSTEM, brief))
        return parse_turn(data, tokens=reply.tokens, latency_ms=reply.latency_ms)

    def reason(self, brief: Mapping[str, Any]) -> ShiftArtifact:
        data, _reply = self.deep.chat_json(self._messages(REASONER_SYSTEM, brief))
        return parse_shift(data)

    @staticmethod
    def _messages(system: str, brief: Mapping[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical_json(brief)},
        ]


class ScriptedRouter:
    """Deterministic stand-in: the same interface, no network."""

    def __init__(
        self,
        executions: Sequence[TurnProposal | Mapping[str, Any]] = (),
        shifts: Sequence[ShiftArtifact | Mapping[str, Any]] = (),
        *,
        strict: bool = False,
    ) -> None:
        self._executions: deque[Any] = deque(executions)
        self._shifts: deque[Any] = deque(shifts)
        self.strict = strict
        self.calls: list[str] = []

    def execute(self, brief: Mapping[str, Any]) -> TurnProposal:
        self.calls.append("execute")
        if not self._executions:
            if self.strict:
                raise RouterError("no scripted execution left")
            return TurnProposal(rationale="scripted: no-op")
        item = self._executions.popleft()
        return item if isinstance(item, TurnProposal) else parse_turn(item)

    def reason(self, brief: Mapping[str, Any]) -> ShiftArtifact:
        self.calls.append("reason")
        if not self._shifts:
            if self.strict:
                raise RouterError("no scripted shift left")
            return ShiftArtifact(
                kind="invariant", payload={"check": "True"}, rationale="scripted: no-op reframe"
            )
        item = self._shifts.popleft()
        return item if isinstance(item, ShiftArtifact) else parse_shift(item)
