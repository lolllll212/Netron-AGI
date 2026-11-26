"""Phase-3 reflex compilation: a recurring win becomes a zero-LLM procedure.

Consolidation (:mod:`early_agi.memory.mera_graph`) tells you *that* a pattern
recurs. Compilation turns that into something executable: the edits that
repeatedly produced a verified entropy drop are stored as a reflex, so the next
matching turn costs no tokens at all.

The trust model is the whole point, and it is deliberately narrow:

* a reflex compiles only when the same edit signature won at least
  ``min_support`` times *and* every one of those wins passed at least one
  dark-state gate — otherwise there is no effect worth caching;
* it fires only when the target files are byte-identical to a pre-state it was
  actually learned on (content hash, never "looks similar");
* its output still faces the same gates as a model edit, and if the gates it was
  compiled against do not pass, the files are restored immediately;
* enough misses retire it.

Memory can therefore make the loop cheaper, never less verified. Nothing here
writes to the sink except through the append-only store, so every reflex carries
a lineage back to the traces that justified it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.cold_sink import SinkRecord, canonical_json
from .mera_graph import MeraGraph

__all__ = [
    "CompiledReflex",
    "ReflexCompileReport",
    "ReflexEdit",
    "ReflexLibrary",
    "compile_reflexes",
    "content_sha",
    "task_signature",
]

_STOPWORDS = frozenset({"a", "an", "and", "for", "from", "in", "of", "or", "the", "to", "with"})


def content_sha(text: str | bytes | None) -> str | None:
    """Content hash used to pin a reflex to the exact pre-state it learned on."""
    if text is None:
        return None
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return hashlib.sha256(raw).hexdigest()


def task_signature(task: str | None) -> str | None:
    """Stable, order-insensitive signature of a task's meaningful tokens."""
    if not task:
        return None
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", task.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }
    if not tokens:
        return None
    canonical = " ".join(sorted(tokens))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ReflexEdit:
    path: str
    content: str
    before_sha: tuple[str | None, ...] = ()

    def accepts(self, root: str | Path) -> bool:
        """True only if the file is in a state this reflex was compiled from."""
        base = Path(root).resolve()
        target = (base / self.path).resolve()
        if target != base and base not in target.parents:
            return False
        if not target.exists():
            return None in self.before_sha
        try:
            return content_sha(target.read_bytes()) in self.before_sha
        except OSError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content, "before_sha": list(self.before_sha)}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> ReflexEdit:
        path, content = data.get("path"), data.get("content")
        if not isinstance(path, str) or not path or not isinstance(content, str):
            raise ValueError("reflex edit needs a 'path' and 'content'")
        raw = data.get("before_sha") or []
        return cls(
            path=path,
            content=content,
            before_sha=tuple(item if isinstance(item, str) else None for item in raw),
        )


@dataclass(frozen=True)
class CompiledReflex:
    hash: str
    name: str
    edits: tuple[ReflexEdit, ...]
    expected_gates: tuple[str, ...]
    pattern: tuple[str, ...]
    support: int
    rule: str = ""
    hits: int = 0
    misses: int = 0
    task_signatures: tuple[str, ...] = ()

    @property
    def short(self) -> str:
        return self.hash[:10]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(edit.path for edit in self.edits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hash": self.short,
            "paths": list(self.paths),
            "expected_gates": list(self.expected_gates),
            "pattern": list(self.pattern),
            "support": self.support,
            "hits": self.hits,
            "misses": self.misses,
            "task_signatures": list(self.task_signatures),
        }


@dataclass(frozen=True)
class ReflexCompileReport:
    batches: int = 0
    signatures: int = 0
    reflexes: int = 0
    skipped: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "signatures": self.signatures,
            "reflexes": self.reflexes,
            "skipped": list(self.skipped),
        }


def _passing(payload: Mapping[str, Any]) -> set[str]:
    gates = payload.get("gates") or {}
    return {str(name) for name, verdict in gates.items() if verdict == "pass"}


def _payload_task(payload: Mapping[str, Any]) -> str:
    state = payload.get("state")
    if isinstance(state, Mapping):
        return str(state.get("task", ""))
    return str(payload.get("task", ""))


def _features(payload: Mapping[str, Any]) -> set[str]:
    features = {f"gate:{name}" for name in _passing(payload)}
    directive = payload.get("directive")
    if directive:
        features.add(f"directive:{directive}")
    return features


def _best_rule(graph: MeraGraph, parents: list[str], min_support: int) -> str:
    """The consolidated rule that most of these traces were absorbed into."""
    best, best_overlap = "", 0
    for rule in graph.rules(min_support=min_support):
        overlap = len(set(parents) & set(graph.members(rule["hash"])))
        if overlap > best_overlap:
            best, best_overlap = rule["hash"], overlap
    return best if best_overlap >= min_support else ""


def compile_reflexes(
    graph: MeraGraph,
    *,
    min_support: int = 2,
    limit: int = 2000,
) -> ReflexCompileReport:
    """Compile recurring, gated edit batches into stored reflexes.

    Input is ``applied_edit`` records (archived by the session only for edits that
    survived the gates), grouped by the tick they belong to. A group compiles when
    its edit signature repeats across ``min_support`` distinct ticks and those
    ticks share at least one passing gate.
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")

    records = graph.sink.records(kinds=("applied_edit",), limit=limit)
    if not records:
        return ReflexCompileReport(skipped=("no applied_edit traces",))

    traces: dict[str, SinkRecord] = {r.hash: r for r in graph.sink.records(kinds=("uv_state",))}
    skipped: list[str] = []

    batches: dict[str, list[SinkRecord]] = {}
    for record in records:
        # ``tick`` is part of the payload because the sink deduplicates identical
        # payloads; the parent column alone would keep only the first occurrence.
        parent = record.payload.get("tick") or record.parent_hash
        if not isinstance(parent, str) or parent not in traces:
            skipped.append(f"edit {record.short} is not anchored to a known tick")
            continue
        batches.setdefault(parent, []).append(record)

    groups: dict[str, list[tuple[str, list[SinkRecord]]]] = {}
    for parent, batch in batches.items():
        signature: list[tuple[str, str]] = []
        for record in sorted(batch, key=lambda r: str(r.payload.get("path"))):
            path, content = record.payload.get("path"), record.payload.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                signature = []
                skipped.append(f"edit {record.short} has no usable path/content")
                break
            if record.payload.get("after_sha") != content_sha(content):
                signature = []
                skipped.append(f"edit {record.short} failed its own content hash")
                break
            signature.append((path, content))
        if not signature:
            continue
        digest = hashlib.sha256(canonical_json(sorted(signature)).encode("utf-8")).hexdigest()
        groups.setdefault(digest, []).append((parent, batch))

    created: set[str] = set()
    for signature, occurrences in sorted(groups.items()):
        parents = sorted({parent for parent, _ in occurrences})
        if len(parents) < min_support:
            continue

        passed_sets = [_passing(traces[parent].payload) for parent in parents]
        expected = sorted(set.intersection(*passed_sets)) if passed_sets else []
        if not expected:
            skipped.append(
                f"{signature[:8]}: the winning edits were never gated, so there is no effect to replay"
            )
            continue
        pattern = sorted(set.intersection(*(_features(traces[parent].payload) for parent in parents)))
        task_signatures = sorted(
            {
                task_sig
                for parent in parents
                if (task_sig := task_signature(_payload_task(traces[parent].payload)))
            }
        )

        # Union of the pre-states every occurrence was learned against: a reflex
        # may fire on any state it has actually seen, never on an unseen one.
        before: dict[tuple[str, str], set[str | None]] = {}
        for _parent, batch in occurrences:
            for record in batch:
                key = (str(record.payload["path"]), str(record.payload["content"]))
                before.setdefault(key, set()).add(record.payload.get("before_sha"))

        edits = []
        for path, content in sorted(before):
            states = before[(path, content)]
            entries: list[str | None] = sorted(state for state in states if isinstance(state, str))
            if None in states:
                entries.append(None)
            edits.append({"path": path, "content": content, "before_sha": entries})

        name = f"reflex/{signature[:8]}"
        scale = max(traces[parent].scale for parent in parents) + 1
        digest = graph.put_node(
            "reflex",
            {
                "name": name,
                "signature": signature,
                "edits": edits,
                "expected_gates": expected,
                "pattern": pattern,
                "task_signatures": task_signatures,
            },
            scale=scale,
            eta_cog=None,
        )
        for parent in parents:
            graph.link(parent, digest, "compiled_from")
        rule = _best_rule(graph, parents, min_support)
        if rule:
            graph.link(rule, digest, "compiled_into")
        created.add(digest)

    return ReflexCompileReport(
        batches=len(batches),
        signatures=len(groups),
        reflexes=len(created),
        skipped=tuple(skipped),
    )


class ReflexLibrary:
    """Read-side view over compiled reflexes, plus the fire/retire bookkeeping."""

    def __init__(self, graph: MeraGraph, *, miss_limit: int = 2) -> None:
        self.graph = graph
        self.miss_limit = max(1, miss_limit)

    def compile(self, *, min_support: int = 2, limit: int = 2000) -> ReflexCompileReport:
        return compile_reflexes(self.graph, min_support=min_support, limit=limit)

    # -- loading ------------------------------------------------------------
    def _outcomes(self) -> dict[str, list[bool]]:
        outcomes: dict[str, list[bool]] = {}
        for record in self.graph.sink.records(kinds=("reflex_outcome",)):
            target = record.payload.get("reflex")
            if isinstance(target, str):
                outcomes.setdefault(target, []).append(bool(record.payload.get("ok")))
        return outcomes

    def reflexes(self) -> list[CompiledReflex]:
        outcomes = self._outcomes()
        compiled: list[CompiledReflex] = []
        for node in self.graph.nodes(kind="reflex"):
            payload = node["payload"]
            try:
                edits = tuple(ReflexEdit.from_payload(item) for item in payload.get("edits", []))
            except ValueError:
                continue
            if not edits:
                continue
            history = outcomes.get(node["hash"], [])
            rules = self.graph.sources(node["hash"], "compiled_into")
            compiled.append(
                CompiledReflex(
                    hash=node["hash"],
                    name=str(payload.get("name") or f"reflex/{node['hash'][:8]}"),
                    edits=edits,
                    expected_gates=tuple(str(g) for g in payload.get("expected_gates", [])),
                    pattern=tuple(str(feature) for feature in payload.get("pattern", [])),
                    support=len(self.graph.sources(node["hash"], "compiled_from")),
                    rule=rules[0] if rules else "",
                    hits=sum(1 for ok in history if ok),
                    misses=sum(1 for ok in history if not ok),
                    task_signatures=tuple(
                        str(item) for item in payload.get("task_signatures", ()) if isinstance(item, str)
                    ),
                )
            )
        return sorted(compiled, key=lambda reflex: (len(reflex.expected_gates), reflex.support), reverse=True)

    # -- execution ----------------------------------------------------------
    def match(
        self,
        root: str | Path,
        *,
        task: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> CompiledReflex | None:
        """Best live reflex whose task and learned pre-states all still hold."""
        requested_task = task if task is not None else (context or {}).get("task")
        signature = task_signature(str(requested_task)) if requested_task is not None else None
        best: CompiledReflex | None = None
        for reflex in self.reflexes():
            if reflex.misses >= self.miss_limit:
                continue
            if signature is not None and signature not in reflex.task_signatures:
                continue
            if not all(edit.accepts(root) for edit in reflex.edits):
                continue
            if best is None or (len(reflex.expected_gates), reflex.support) > (
                len(best.expected_gates),
                best.support,
            ):
                best = reflex
        return best

    def record(
        self,
        reflex: CompiledReflex,
        *,
        ok: bool,
        eta_cog: float | None = None,
        parent_hash: str | None = None,
    ) -> str:
        """Archive the outcome; the sink stays the single source of truth."""
        return self.graph.sink.put(
            "reflex_outcome",
            {
                "reflex": reflex.hash,
                "name": reflex.name,
                "ok": bool(ok),
                "tick": parent_hash,
                "eta_cog": None if eta_cog is None else round(float(eta_cog), 6),
            },
            scale=0,
            parent_hash=parent_hash,
            eta_cog=eta_cog,
        )
