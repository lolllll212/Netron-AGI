"""Deterministic dark-state verifiers with true tri-state semantics."""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.adaptability import distilled_constraint

__all__ = [
    "DarkStateGate",
    "GateResult",
    "ast_syntax_gate",
    "build_standard_gates",
    "pytest_gate",
    "ruff_gate",
    "run_pytest",
    "run_z3_smt2",
    "verifier_availability",
    "z3_gate",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool | None
    bits: float = 1.0
    detail: str = ""
    constraint: str | None = None

    @property
    def verdict(self) -> str:
        return {True: "pass", False: "fail", None: "unverifiable"}[self.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "bits": self.bits,
            "detail": self.detail,
            "constraint": self.constraint,
        }


Check = Callable[[Mapping[str, Any]], bool | GateResult | None]


def _fail(name: str, detail: str = "", *, constraint: str | None = None) -> GateResult:
    """Return a failure carrying only one distilled active-manifold constraint."""
    minimal = constraint or distilled_constraint(
        name,
        {"passed": False, "verdict": "fail", "detail": detail},
    )
    minimal = minimal or f"assert not {name}_failed"
    return GateResult("<gate>", False, 0.0, minimal, minimal)


class DarkStateGate:
    def __init__(self, name: str, check: Check, bits: float = 1.0) -> None:
        if not name:
            raise ValueError("gate needs a name")
        self.name = name
        self.bits = float(bits)
        self._check = check

    def evaluate(self, state: Mapping[str, Any]) -> GateResult:
        try:
            raw = self._check(state)
        except Exception as exc:
            return GateResult(self.name, None, self.bits, f"{type(exc).__name__}: {exc}")
        if isinstance(raw, GateResult):
            constraint = raw.constraint
            if constraint is None and raw.passed is False:
                constraint = distilled_constraint(self.name, raw.as_dict())
            return GateResult(self.name, raw.passed, self.bits, raw.detail, constraint)
        if raw is None:
            return GateResult(self.name, None, self.bits, "verifier unavailable", None)
        failed = not bool(raw)
        return GateResult(
            self.name,
            bool(raw),
            self.bits,
            "",
            f"assert not {self.name}_failed" if failed else None,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"DarkStateGate({self.name!r}, bits={self.bits})"


def _root(state: Mapping[str, Any]) -> Path:
    root = state.get("repo_root")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _targets(state: Mapping[str, Any]) -> list[Path]:
    root = _root(state)
    sources: Mapping[str, str] | None = state.get("sources")  # type: ignore[assignment]
    if sources:
        return [root / relative for relative in sources]
    explicit = state.get("paths")
    if explicit:
        out: list[Path] = []
        for item in explicit:
            path = Path(item)
            out.append(path if path.is_absolute() else root / path)
        return out
    return sorted(root.rglob("*.py"))


def _sources(state: Mapping[str, Any]) -> dict[Path, str]:
    root = _root(state)
    provided: Mapping[str, str] | None = state.get("sources")  # type: ignore[assignment]
    if provided:
        return {root / relative: text for relative, text in provided.items()}
    out: dict[Path, str] = {}
    for path in _targets(state):
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            out[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def _ignored(path: Path, root: Path) -> bool:
    try:
        parts = set(path.relative_to(root).parts)
    except ValueError:
        return True
    return bool(parts & {".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".git"})


def _pytest_assertion(stdout: str, stderr: str) -> str:
    """Extract the smallest failing assertion; never return a traceback."""
    text = stdout or stderr
    assertions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E "):
            stripped = stripped[2:].strip()
        if stripped.startswith("assert "):
            assertions.append(stripped[:240])
    if assertions:
        return assertions[-1]
    match = re.search(r"AssertionError:\s*(.+?)(?:\s{2,}|$)", text)
    if match:
        return f"assert {match.group(1).strip()[:240]}"
    return "assert tests_pass"


def _z3_counterexample(model_text: str) -> dict[str, str]:
    """Normalize a Z3 model into a compact, path-independent counterexample."""
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_!.]*)\s*=\s*([^,\]\n]+)", model_text)
    return {str(key): str(value).strip() for key, value in pairs if str(value).strip()}


def ast_syntax_gate(name: str = "syntax_ok", bits: float = 2.0) -> DarkStateGate:
    def check(state: Mapping[str, Any]) -> bool | None:
        sources = _sources(state)
        root = _root(state)
        checked = 0
        errors: list[str] = []
        for path, text in sources.items():
            if path.suffix != ".py" or _ignored(path, root):
                continue
            checked += 1
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(f"{path.name}:{exc.lineno}: {exc.msg}")
        if checked == 0:
            return None
        if errors:
            return _fail(name, "; ".join(errors[:5]))
        return True

    return DarkStateGate(name, check, bits)


def _ruff_command() -> list[str] | None:
    if shutil.which("ruff"):
        return ["ruff"]
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return None


def ruff_gate(name: str = "lint_ok", bits: float = 1.5, timeout: float = 30.0) -> DarkStateGate:
    def check(state: Mapping[str, Any]) -> bool | None:
        command = _ruff_command()
        if command is None:
            return None
        root = _root(state)
        targets = [p for p in _targets(state) if p.is_file() and not _ignored(p, root)]
        if not targets:
            return None
        completed = subprocess.run(
            [*command, "check", "--quiet", "--output-format=concise", *[str(p) for p in targets]],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode == 0:
            return True
        if completed.returncode == 1:
            return _fail(name, (completed.stdout or completed.stderr).strip()[:2000])
        return None

    return DarkStateGate(name, check, bits)


def run_pytest(
    root: str | Path,
    paths: Sequence[str] = (),
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", *paths]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "passed": None,
            "stdout": "",
            "stderr": f"pytest timed out after {timeout}s",
        }
    stdout = (completed.stdout or "")[-4000:]
    stderr = (completed.stderr or "")[-4000:]
    if completed.returncode == 0:
        passed: bool | None = True
    elif completed.returncode == 1:
        passed = False
    else:
        passed = None
    assertion = _pytest_assertion(stdout, stderr) if passed is False else ""
    constraint = f"assert {assertion.removeprefix('assert ')}" if assertion else None
    return {
        "returncode": completed.returncode,
        "passed": passed,
        "stdout": stdout,
        "stderr": stderr,
        "assertion": assertion,
        "constraint": constraint,
        "failures": [{"assertion": assertion}] if assertion else [],
    }


def pytest_gate(
    name: str = "tests_pass",
    bits: float = 4.0,
    targets: Sequence[str] = (),
    timeout: float = 180.0,
) -> DarkStateGate:
    def check(state: Mapping[str, Any]) -> bool | None:
        result = run_pytest(_root(state), targets, timeout=timeout)
        if result["passed"] is True:
            return True
        if result["passed"] is False:
            return _fail(
                name,
                "",
                constraint=str(
                    result.get("constraint") or _pytest_assertion(result["stdout"], result["stderr"])
                ),
            )
        return None

    return DarkStateGate(name, check, bits)


def run_z3_smt2(smt2: str, *, timeout_ms: int = 5000) -> dict[str, Any]:
    """Solve SMT-LIB2. ``passed`` is True for unsat, False for sat, None otherwise."""
    try:
        import z3  # type: ignore import-not-found
    except ImportError as exc:
        raise RuntimeError("z3-solver is not installed; pip install 'early-agi[verify]'") from exc

    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    try:
        solver.from_string(smt2)
        outcome = solver.check()
    except z3.Z3Exception as exc:
        return {"status": "error", "passed": None, "detail": str(exc)}
    status = str(outcome)
    model: dict[str, str] = {}
    counterexample = ""
    if status == "unsat":
        passed: bool | None = True
        detail = ""
    elif status == "sat":
        passed = False
        detail = str(solver.model())
        model = _z3_counterexample(detail)
        counterexample = ", ".join(f"{key} == {value}" for key, value in sorted(model.items()))
        detail = f"counterexample: {counterexample}" if counterexample else "constraint refuted"
    else:
        passed = None
        detail = f"solver returned {status}"
    return {
        "status": status,
        "passed": passed,
        "detail": detail,
        "model": model,
        "counterexample": counterexample,
        "constraint": f"counterexample: {counterexample}" if counterexample else None,
    }


def z3_gate(
    builder: Callable[[Mapping[str, Any]], str | bool | None],
    name: str = "constraints_proved",
    bits: float = 3.0,
    *,
    timeout_ms: int = 5000,
) -> DarkStateGate:
    """A real Z3 gate. The builder returns SMT-LIB2 text, a bool, or None."""

    def check(state: Mapping[str, Any]) -> bool | None:
        if importlib.util.find_spec("z3") is None:
            return None
        target = builder(state)
        if target is None:
            return None
        if isinstance(target, bool):
            return target
        if not isinstance(target, str):
            return None
        result = run_z3_smt2(target, timeout_ms=timeout_ms)
        if result["passed"] is False:
            return _fail(
                name,
                "",
                constraint=str(result.get("constraint") or result.get("detail", "constraint refuted"))[:1500],
            )
        return result["passed"]

    return DarkStateGate(name, check, bits)


smt2_gate = z3_gate


def build_standard_gates(
    *,
    include_pytest: bool = True,
    include_ruff: bool = True,
    pytest_targets: Sequence[str] = (),
    include_z3: bool = False,
    z3_builder: Callable[[Mapping[str, Any]], str | bool | None] | None = None,
) -> list[DarkStateGate]:
    gates = [ast_syntax_gate()]
    if include_ruff:
        gates.append(ruff_gate())
    if include_pytest:
        gates.append(pytest_gate(targets=pytest_targets))
    if include_z3:
        if z3_builder is None:
            raise ValueError("include_z3=True requires z3_builder")
        gates.append(z3_gate(z3_builder))
    return gates


def verifier_availability() -> dict[str, bool]:
    return {
        "ast": True,
        "ruff": _ruff_command() is not None,
        "pytest": importlib.util.find_spec("pytest") is not None,
        "z3": importlib.util.find_spec("z3") is not None,
    }
