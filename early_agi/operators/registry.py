"""Non-commuting operators and the commutator side-effect matrix."""

from __future__ import annotations

import ast
import difflib
import io
import keyword
import os
import shlex
import subprocess
import time
import tokenize
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EPISTEMIC_PROBE",
    "READ",
    "READ_ONLY_KINDS",
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
READ_ONLY_KINDS = frozenset({READ, VERIFY})
EPISTEMIC_PROBE = "EPISTEMIC_PROBE"


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
    rendered: set[str] = set()
    for item in template:
        try:
            rendered.add(item.format(**args))
        except (KeyError, IndexError, ValueError):
            rendered.add(WILDCARD)
    return frozenset(rendered)


def _matching_name(target: str, mapping: Mapping[str, str]) -> str | None:
    for key, value in mapping.items():
        if _scope_matches(target, key):
            return value
    return None


def _verifier_ok(result: Any) -> bool | None:
    """Map verifier results to pass / fail / unverifiable."""
    if not isinstance(result, Mapping):
        return True
    if "passed" in result:
        verdict = result.get("passed")
        return None if verdict is None else bool(verdict)
    status = result.get("status")
    if status == "unsat":
        return True
    if status == "sat":
        return False
    if status in {"unknown", "error"}:
        return None
    return True


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
        return str(self.kind).casefold() == WRITE or bool(self.writes)


@dataclass(frozen=True)
class OperatorEvent:
    seq: int
    turn: int
    name: str
    kind: str
    targets: frozenset[str]
    ok: bool | None
    blocked: bool
    mutates: bool
    elapsed_ms: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class CommutatorViolation:
    turn: int
    kind: str
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
        self._directive: str = ""

    @property
    def active_directive(self) -> str:
        """Current kernel directive, if any, constraining operator selection."""
        return self._directive

    def set_directive(self, directive: str) -> None:
        self._directive = str(directive)

    def clear_directive(self) -> None:
        self._directive = ""

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

    def replace(self, spec: OperatorSpec) -> OperatorSpec:
        """Install or replace a spec; used by compiled Gauge programs."""
        if not spec.name:
            raise ValueError("operator needs a name")
        self._ops[spec.name] = spec
        return spec

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

    def epistemic_probe(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Run cheap read-only operators while temporarily isolating writes."""
        if calls is None:
            calls = (
                ("inspect_deps", {}),
                ("run_pytest", {"paths": (), "timeout": 30.0}),
                ("check_z3", {"smt2": "(check-sat)", "timeout_ms": 1000}),
            )
        previous = self._lock_reason
        previous_directive = self._directive
        self.isolate_writes(reason="EPISTEMIC_PROBE (read-only measurements only)")
        self.set_directive(EPISTEMIC_PROBE)
        results: list[dict[str, Any]] = []
        try:
            for name, kwargs in calls:
                spec = self.operator(name)
                if spec.mutates or str(spec.kind).casefold() not in READ_ONLY_KINDS:
                    results.append({"operator": name, "skipped": "mutating operator"})
                    continue
                try:
                    value = self.apply(name, **dict(kwargs))
                    results.append(
                        {
                            "operator": name,
                            "ok": True,
                            "value": value if isinstance(value, Mapping) else str(value)[:500],
                        }
                    )
                except Exception as exc:  # probing must never abort the turn
                    results.append({"operator": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._lock_reason = previous
            self._directive = previous_directive
        return {"operators": [name for name, _ in calls], "results": results}

    def snapshot(self) -> dict[str, Any]:
        """Capture mutable bookkeeping for a transactional session turn."""
        return {
            "turn": self.turn,
            "seq": self._seq,
            "log": list(self._log),
            "lock_reason": self._lock_reason,
            "directive": self._directive,
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.turn = int(checkpoint.get("turn", 0))
        self._seq = int(checkpoint.get("seq", 0))
        self._log = list(checkpoint.get("log", ()))
        self._lock_reason = str(checkpoint.get("lock_reason", ""))
        self._directive = str(checkpoint.get("directive", ""))

    def apply(self, name: str, /, **kwargs: Any) -> Any:
        spec = self.operator(name)
        targets = _render(spec.writes or spec.reads, kwargs) if (spec.writes or spec.reads) else frozenset()

        if spec.mutates and self.writes_locked:
            self._record(spec, targets, ok=False, blocked=True, detail=self._lock_reason)
            raise WriteLocked(f"operator {name!r} is mutating but writes are isolated: {self._lock_reason}")
        if self._directive == EPISTEMIC_PROBE and str(spec.kind).casefold() not in READ_ONLY_KINDS:
            detail = f"{EPISTEMIC_PROBE} allows only READ/VERIFY operators"
            self._record(spec, targets, ok=False, blocked=True, detail=detail)
            raise WriteLocked(f"operator {name!r} is blocked: {detail}")

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
        ok: bool | None = True
        if str(spec.kind).casefold() == VERIFY:
            ok = _verifier_ok(result)
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
        ok: bool | None,
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
                mutates=spec.mutates,
                elapsed_ms=elapsed_ms,
                detail=detail,
            )
        )
        if len(self._log) > self._log_limit:
            del self._log[: len(self._log) - self._log_limit]

    @property
    def log(self) -> tuple[OperatorEvent, ...]:
        return tuple(self._log)

    def commutator(self, first: str, second: str, /, **kwargs: Any) -> float:
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

    def audit_turn(self, turn: int | None = None) -> list[CommutatorViolation]:
        target_turn = self.turn if turn is None else turn
        events = [event for event in self._log if event.turn == target_turn]
        violations: list[CommutatorViolation] = []
        verified: dict[str, str] = {}
        written: dict[str, str] = {}

        for event in events:
            if event.blocked:
                violations.append(
                    CommutatorViolation(
                        target_turn, "blocked_write", event.name, event.name, tuple(sorted(event.targets))
                    )
                )
                continue
            if event.kind == VERIFY and event.ok is True:
                for target in event.targets:
                    verified[target] = event.name
                continue
            if event.mutates:
                stale = sorted(t for t in event.targets if any(_scope_matches(t, v) for v in verified))
                if stale:
                    first = _matching_name(stale[0], verified) or "verify"
                    violations.append(
                        CommutatorViolation(
                            target_turn,
                            "stale_verification",
                            first,
                            event.name,
                            tuple(stale),
                        )
                    )
                raced = sorted(t for t in event.targets if any(_scope_matches(t, w) for w in written))
                if raced:
                    first = _matching_name(raced[0], written) or event.name
                    violations.append(
                        CommutatorViolation(
                            target_turn,
                            "write_race",
                            first,
                            event.name,
                            tuple(raced),
                        )
                    )
                for target in event.targets:
                    written[target] = event.name
        return violations

    def reset_log(self) -> None:
        self._log.clear()
        self._seq = 0


def _safe_join(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise PermissionError(f"path escapes repository root: {relative!r}")
    return target


def _rename_identifiers(source: str, mapping: Mapping[str, str]) -> tuple[str, int]:
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
    ast.parse(out)
    return out, len(edits)


def build_default_registry(root: str | Path, *, allow_shell: bool = False) -> OperatorRegistry:
    """Wire the Phase-2 tools with explicit side-effect footprints.

    ``run_shell`` is unconstrained by construction. It is registered with a
    wildcard footprint for commutator auditing, but disabled by default so a
    coordinate shift cannot accidentally turn a text edit into process escape.
    """
    base = Path(root)

    def read_file(path: str) -> str:
        return _safe_join(base, path).read_text(encoding="utf-8", errors="replace")

    def inspect_deps() -> dict[str, Any]:
        """Read-only AST motif census used by epistemic probing."""
        ignored = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}
        files: list[str] = []
        imports: set[str] = set()
        functions = 0
        calls = 0
        for target in sorted(base.rglob("*.py")):
            relative = target.relative_to(base).as_posix()
            if any(part in ignored for part in target.relative_to(base).parts):
                continue
            files.append(relative)
            try:
                tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module or "")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions += 1
                elif isinstance(node, ast.Call):
                    calls += 1
        return {
            "modules": len(files),
            "imports": sorted(item for item in imports if item),
            "functions": functions,
            "calls": calls,
            "files": files[:32],
        }

    def _expected_bytes(expected: str | bytes | None) -> bytes | None:
        if expected is None:
            return None
        return expected.encode("utf-8") if isinstance(expected, str) else bytes(expected)

    def write_file(path: str, content: str | bytes, *, expected: str | bytes | None = None) -> dict[str, Any]:
        target = _safe_join(base, path)
        previous = target.read_bytes() if target.exists() else b""
        expect = _expected_bytes(expected)
        if expect is not None and previous != expect:
            raise RuntimeError(f"refusing to overwrite {path!r}: content changed since it was read")
        encoded = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        return {
            "path": path,
            "created": not previous,
            "bytes": len(encoded),
            "diff": "".join(
                difflib.unified_diff(
                    previous.decode("utf-8", errors="replace").splitlines(keepends=True),
                    encoded.decode("utf-8", errors="replace").splitlines(keepends=True),
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
            ast.parse(updated)
            target.write_text(updated, encoding="utf-8")
        return {"path": path, "renames": count}

    def run_shell(command: str, *, timeout: float = 60.0) -> dict[str, Any]:
        if not allow_shell:
            raise PermissionError("run_shell is disabled; enable it explicitly in a sandbox")
        completed = subprocess.run(
            shlex.split(command, posix=(os.name != "nt")),
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
                name="inspect_deps",
                fn=inspect_deps,
                kind=READ,
                reads=frozenset({"*"}),
                cost=0.2,
                description="read-only AST import/function/call census for epistemic probing",
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
