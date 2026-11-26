"""Non-commuting operators and the commutator side-effect matrix.

Tools are not interchangeable functions here. Each one declares the resources it
reads and writes, which makes ``[T_i, T_j] != 0`` concrete: running the tests
before mutating a coupled file produces a different state than mutating first,
so the verification you just paid for is stale.

Two guards come out of that:

* ``isolate_writes`` — the kernel trips this on a coordinate shift, so the fast
  executor physically cannot patch its way out of a SHIFT_COORDINATE interrupt.
* ``audit_turn`` — flags any mutating call that consumed a verification of the
  same resource within the same turn.
"""

from __future__ import annotations

import ast
import difflib
import io
import keyword
import shlex
import subprocess
import time
import tokenize
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "READ",
    "VERIFY",
    "WILDCARD",
    "WRITE",
    "CommutatorViolation",
    "OperatorEvent",
    "OperatorRegistry",
    "OperatorSpec",
    "WriteLocked",
    "build_default_registry",
]

WILDCARD = "*"
READ = "read"
WRITE = "write"
VERIFY = "verify"


class WriteLocked(RuntimeError):
    """Raised when a mutating operator is invoked while writes are isolated."""


def _scope_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == WILDCARD or right == WILDCARD:
        return True
    if left.endswith("/*"):
        return right == left[:-2] or right.startswith(left[:-1])
    if right.endswith("/*"):
        return left == right[:-2] or left.startswith(right[:-1])
    if left.endswith("/"):
        return right.startswith(left)
    if right.endswith("/"):
        return left.startswith(right)
    return False


def _overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    return any(_scope_matches(a, b) for a in left for b in right)


def _render(template: Iterable[str], args: Mapping[str, Any]) -> frozenset[str]:
    """Substitute call arguments into scope templates; unknown -> wildcard."""
    rendered: set[str] = set()
    for item in template:
        try:
            rendered.add(item.format(**args))
        except (KeyError, IndexError, ValueError):
            rendered.add(WILDCARD)
    return frozenset(rendered)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    fn: Callable[..., Any]
    kind: str = READ
    reads: frozenset[str] = field(default_factory=frozenset)
    writes: frozenset[str] = field(default_factory=frozenset)
    cost: float = 0.0
    description: str = ""

    @property
    def mutates(self) -> bool:
        return self.kind == WRITE or bool(self.writes)


@dataclass(frozen=True)
class OperatorEvent:
    seq: int
    turn: int
    name: str
    kind: str
    targets: frozenset[str]
    ok: bool
    blocked: bool
    elapsed_ms: float = 0.0
    detail: str = ""

    @property
    def mutates(self) -> bool:
        return self.kind == WRITE


@dataclass(frozen=True)
class CommutatorViolation:
    turn: int
    kind: str  # "stale_verification" | "write_race" | "blocked_write"
    first: str
    second: str
    targets: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"turn {self.turn}: {self.kind} — {self.first} -> {self.second} "
            f"on {', '.join(self.targets) or '<any>'}"
        )


class OperatorRegistry:
    """Stateful tool registry with a live commutator log."""

    def __init__(self, *, log_limit: int = 4096) -> None:
        self._ops: dict[str, OperatorSpec] = {}
        self._log: list[OperatorEvent] = []
        self._log_limit = max(64, log_limit)
        self._seq = 0
        self.turn = 0
        self._lock_reason: str = ""

    # -- registration -------------------------------------------------------
    def register(self, spec: OperatorSpec) -> OperatorSpec:
        if not spec.name:
            raise ValueError("operator needs a name")
        if spec.name in self._ops:
            raise ValueError(f"operator {spec.name!r} is already registered")
        self._ops[spec.name] = spec
        return spec

    def register_many(self, specs: Iterable[OperatorSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def operator(self, name: str) -> OperatorSpec:
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(f"unknown operator {name!r}; registered: {', '.join(sorted(self._ops))}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._ops))

    def __contains__(self, name: object) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    # -- write isolation (Trap 2) ------------------------------------------
    @property
    def writes_locked(self) -> bool:
        return bool(self._lock_reason)

    @property
    def lock_reason(self) -> str:
        return self._lock_reason

    def isolate_writes(self, *, reason: str = "coordinate shift") -> None:
        self._lock_reason = reason or "coordinate shift"

    def release_writes(self) -> None:
        self._lock_reason = ""

    def consume_turn(self) -> int:
        self.turn += 1
        return self.turn

    # -- invocation ---------------------------------------------------------
    def apply(self, name: str, /, **kwargs: Any) -> Any:
        spec = self.operator(name)
        targets = _render(spec.writes or spec.reads, kwargs) if (spec.writes or spec.reads) else frozenset()

        if spec.mutates and self.writes_locked:
            self._record(spec, targets, ok=False, blocked=True, detail=self._lock_reason)
            raise WriteLocked(f"operator {name!r} is mutating but writes are isolated: {self._lock_reason}")

        started = time.perf_counter()
        try:
            result = spec.fn(**kwargs)
        except Exception as exc:
            self._record(
                spec,
                targets,
                ok=False,
                blocked=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        # A verifier that ran but reported failure is not a successful
        # verification; the reordering audit depends on that distinction.
        ok = True
        if spec.kind == VERIFY and isinstance(result, Mapping):
            verdict = result.get("passed")
            if verdict is not None:
                ok = bool(verdict)
        self._record(
            spec,
            targets,
            ok=ok,
            blocked=False,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return result

    def _record(
        self,
        spec: OperatorSpec,
        targets: frozenset[str],
        *,
        ok: bool,
        blocked: bool,
        elapsed_ms: float = 0.0,
        detail: str = "",
    ) -> None:
        self._seq += 1
        self._log.append(
            OperatorEvent(
                seq=self._seq,
                turn=self.turn,
                name=spec.name,
                kind=spec.kind,
                targets=targets,
                ok=ok,
                blocked=blocked,
                elapsed_ms=elapsed_ms,
                detail=detail,
            )
        )
        if len(self._log) > self._log_limit:
            del self._log[: len(self._log) - self._log_limit]

    @property
    def log(self) -> tuple[OperatorEvent, ...]:
        return tuple(self._log)

    # -- commutators --------------------------------------------------------
    def commutator(self, first: str, second: str, /, **kwargs: Any) -> float:
        """0.0 when the operators commute, 2.0 for write/write, 1.0 for write/read."""
        a, b = self.operator(first), self.operator(second)
        a_writes = _render(a.writes, kwargs)
        b_writes = _render(b.writes, kwargs)
        a_reads = _render(a.reads, kwargs)
        b_reads = _render(b.reads, kwargs)
        if a_writes and b_writes and _overlap(a_writes, b_writes):
            return 2.0
        if a_writes and (b_reads or b_writes) and _overlap(a_writes, b_reads | b_writes):
            return 1.0
        if b_writes and (a_reads or a_writes) and _overlap(b_writes, a_reads | a_writes):
            return 1.0
        return 0.0

    def commutes(self, first: str, second: str, /, **kwargs: Any) -> bool:
        return self.commutator(first, second, **kwargs) == 0.0

    def matrix(
        self, names: Sequence[str] | None = None, *, path: str = "<path>"
    ) -> dict[str, dict[str, float]]:
        """Symmetric commutator side-effect matrix for the given operators."""
        chosen = tuple(names) if names else self.names()
        args = {"path": path, "command": path, "target": path}
        return {a: {b: self.commutator(a, b, **args) for b in chosen} for a in chosen}

    def as_table(self, names: Sequence[str] | None = None, *, path: str = "<path>") -> str:
        matrix = self.matrix(names, path=path)
        columns = list(matrix)
        width = max([len(name) for name in columns] + [4]) + 2
        header = "".ljust(width) + "".join(name.rjust(width) for name in columns)
        rows = [header]
        for row_name, row in matrix.items():
            rows.append(row_name.ljust(width) + "".join(f"{row[col]:.1f}".rjust(width) for col in columns))
        return "\n".join(rows)

    # -- reordering audit ---------------------------------------------------
    def audit_turn(self, turn: int | None = None) -> list[CommutatorViolation]:
        """Flag verifications invalidated by a later mutation in the same turn."""
        target_turn = self.turn if turn is None else turn
        events = [event for event in self._log if event.turn == target_turn]
        violations: list[CommutatorViolation] = []
        verified: dict[str, str] = {}  # target -> verifier name
        written: dict[str, str] = {}

        for event in events:
            if event.blocked:
                violations.append(
                    CommutatorViolation(
                        target_turn, "blocked_write", event.name, event.name, tuple(sorted(event.targets))
                    )
                )
                continue
            if event.kind == VERIFY and event.ok:
                for target in event.targets:
                    verified[target] = event.name
                continue
            if event.mutates:
                stale = sorted(t for t in event.targets if any(_scope_matches(t, v) for v in verified))
                if stale:
                    violations.append(
                        CommutatorViolation(
                            target_turn,
                            "stale_verification",
                            verified.get(stale[0], "verify"),
                            event.name,
                            tuple(stale),
                        )
                    )
                raced = sorted(t for t in event.targets if any(_scope_matches(t, w) for w in written))
                if raced:
                    violations.append(
                        CommutatorViolation(
                            target_turn, "write_race", written[raced[0]], event.name, tuple(raced)
                        )
                    )
                for target in event.targets:
                    written[target] = event.name
        return violations

    def reset_log(self) -> None:
        self._log.clear()
        self._seq = 0


# ---------------------------------------------------------------------------
# Built-in stateful operators (read / edit / shell / ast_refactor / verifiers)
# ---------------------------------------------------------------------------
def _safe_join(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise PermissionError(f"path escapes repository root: {relative!r}")
    return target


def _rename_identifiers(source: str, mapping: Mapping[str, str]) -> tuple[str, int]:
    """Rename identifiers in place using token positions (formatting preserved).

    Token spans are used rather than AST spans because a ``FunctionDef`` node
    starts at the ``def`` keyword; splicing on that span would delete it.
    """
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    edits: list[tuple[int, int, str]] = []
    stream = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in stream:
        if token.type != tokenize.NAME:
            continue
        if keyword.iskeyword(token.string) or token.string not in mapping:
            continue
        start = offsets[token.start[0] - 1] + token.start[1]
        stop = offsets[token.end[0] - 1] + token.end[1]
        edits.append((start, stop, mapping[token.string]))

    if not edits:
        return source, 0
    out = source
    for start, stop, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + replacement + out[stop:]
    ast.parse(out)  # never hand back source that no longer parses
    return out, len(edits)


def build_default_registry(root: str | Path) -> OperatorRegistry:
    """Wire the six Phase-2 tools with explicit side-effect footprints."""
    base = Path(root)

    def read_file(path: str) -> str:
        return _safe_join(base, path).read_text(encoding="utf-8")

    def write_file(path: str, content: str, *, expected: str | None = None) -> dict[str, Any]:
        target = _safe_join(base, path)
        previous = target.read_text(encoding="utf-8") if target.exists() else ""
        if expected is not None and previous != expected:
            raise RuntimeError(f"refusing to overwrite {path!r}: content changed since it was read")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "path": path,
            "created": not previous,
            "bytes": len(content.encode("utf-8")),
            "diff": "".join(
                difflib.unified_diff(
                    previous.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            ),
        }

    def ast_refactor(path: str, mapping: Mapping[str, str]) -> dict[str, Any]:
        target = _safe_join(base, path)
        source = target.read_text(encoding="utf-8")
        updated, count = _rename_identifiers(source, mapping)
        if count:
            # Parse before writing: a refactor that breaks syntax must not land.
            ast.parse(updated)
            target.write_text(updated, encoding="utf-8")
        return {"path": path, "renames": count}

    def run_shell(command: str, *, timeout: float = 60.0) -> dict[str, Any]:
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    def run_pytest(paths: Sequence[str] = (), *, timeout: float = 180.0) -> dict[str, Any]:
        from .dark_state import run_pytest as _run_pytest

        return _run_pytest(base, paths, timeout=timeout)

    def z3_check(smt2: str, *, timeout_ms: int = 5000) -> dict[str, Any]:
        from .dark_state import run_z3_smt2

        return run_z3_smt2(smt2, timeout_ms=timeout_ms)

    registry = OperatorRegistry()
    registry.register_many(
        [
            OperatorSpec(
                name="read_file",
                fn=read_file,
                kind=READ,
                reads=frozenset({"{path}"}),
                cost=0.1,
                description="read a repository file",
            ),
            OperatorSpec(
                name="write_file",
                fn=write_file,
                kind=WRITE,
                writes=frozenset({"{path}"}),
                reads=frozenset({"{path}"}),
                cost=1.0,
                description="replace a file (optionally CAS-guarded by expected content)",
            ),
            OperatorSpec(
                name="ast_refactor",
                fn=ast_refactor,
                kind=WRITE,
                writes=frozenset({"{path}"}),
                reads=frozenset({"{path}"}),
                cost=1.5,
                description="AST-position rename preserving formatting",
            ),
            OperatorSpec(
                name="run_shell",
                fn=run_shell,
                kind=WRITE,
                writes=frozenset({WILDCARD}),
                reads=frozenset({WILDCARD}),
                cost=2.0,
                description="arbitrary process: unconstrained side effects",
            ),
            OperatorSpec(
                name="run_pytest",
                fn=run_pytest,
                kind=VERIFY,
                reads=frozenset({"*"}),
                cost=1.0,
                description="run the test suite (dark-state verifier)",
            ),
            OperatorSpec(
                name="check_z3",
                fn=z3_check,
                kind=VERIFY,
                reads=frozenset(),
                cost=0.5,
                description="prove/refute an SMT-LIB2 constraint (dark-state verifier)",
            ),
        ]
    )
    return registry
