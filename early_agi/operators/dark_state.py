"""Deterministic dark-state verifiers.

A dark state is one a constraint kills: ``L_k |psi*> = 0``. The engineering
version of that statement is cheaper and stricter — the gate is an executable
predicate over the workspace, and only a *verified* collapse of uncertainty is
allowed to pay entropy back into ``eta_cog``.

Gates are tri-state on purpose:

* ``True``  — the invariant holds; award its bits (once per satisfaction).
* ``False`` — the invariant is broken; charge its bits retroactively.
* ``None``  — the verifier could not run (missing tool, no tests, timeout). It
  awards nothing and breaks nothing, so a missing ``z3`` cannot manufacture
  efficiency and a crashed linter cannot fake a regression.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    @property
    def verdict(self) -> str:
        return {True: "pass", False: "fail", None: "unverifiable"}[self.passed]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verdict": self.verdict, "bits": self.bits, "detail": self.detail}


Check = Callable[[Mapping[str, Any]], bool | GateResult | None]


def _fail(detail: str) -> GateResult:
    """Signal a *verified* negative from inside a gate check.

    Raising would be indistinguishable from a broken verifier, which the gate
    deliberately treats as neutral. The name/bits are overwritten by callers.
    """
    return GateResult("<gate>", False, 0.0, detail)


class DarkStateGate:
    """A named predicate with an entropy weight."""

    def __init__(self, name: str, check: Check, bits: float = 1.0) -> None:
        if not name:
            raise ValueError("gate needs a name")
        self.name = name
        self.bits = float(bits)
        self._check = check

    def evaluate(self, state: Mapping[str, Any]) -> GateResult:
        try:
            raw = self._check(state)
        except Exception as exc:  # a verifier crash must be neutral, not a failure
            return GateResult(self.name, None, self.bits, f"{type(exc).__name__}: {exc}")
        if isinstance(raw, GateResult):
            return GateResult(self.name, raw.passed, self.bits, raw.detail)
        if raw is None:
            return GateResult(self.name, None, self.bits, "verifier unavailable")
        return GateResult(self.name, bool(raw), self.bits, "")

    def __repr__(self) -> str:  # pragma: no cover - cosmetics
        return f"DarkStateGate({self.name!r}, bits={self.bits})"


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------
def _root(state: Mapping[str, Any]) -> Path:
    root = state.get("repo_root")
    return Path(root) if root else Path.cwd()


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
        except OSError:
            continue
    return out


def _ignored(path: Path, root: Path) -> bool:
    parts = set(path.relative_to(root).parts)
    return bool(parts & {".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".git"})


# ---------------------------------------------------------------------------
# AST syntax gate
# ---------------------------------------------------------------------------
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
            return _fail("; ".join(errors[:5]))
        return True

    return DarkStateGate(name, check, bits)


# ---------------------------------------------------------------------------
# Ruff gate
# ---------------------------------------------------------------------------
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
            return _fail((completed.stdout or completed.stderr).strip()[:2000])
        return None

    return DarkStateGate(name, check, bits)


# ---------------------------------------------------------------------------
# Pytest gate
# ---------------------------------------------------------------------------
def run_pytest(
    root: str | Path,
    paths: Sequence[str] = (),
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Run pytest in ``root``; ``passed`` is None when the outcome is unknown."""
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
    if completed.returncode == 0:
        passed: bool | None = True
    elif completed.returncode == 1:
        passed = False
    elif completed.returncode == 5:
        passed = None  # no tests collected
    else:
        passed = None  # interrupted / usage error / internal error
    return {
        "returncode": completed.returncode,
        "passed": passed,
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-4000:],
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
            return _fail((result["stdout"] or result["stderr"]).strip()[-1500:])
        return None

    return DarkStateGate(name, check, bits)


# ---------------------------------------------------------------------------
# Z3 gate / SMT runner
# ---------------------------------------------------------------------------
def run_z3_smt2(smt2: str, *, timeout_ms: int = 5000) -> dict[str, Any]:
    """Solve an SMT-LIB2 script. Raises if z3-solver is not installed."""
    try:
        import z3  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("z3-solver is not installed; pip install 'early-agi[verify]'") from exc

    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    try:
        solver.from_string(smt2)
        outcome = solver.check()
    except z3.Z3Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return {
        "status": str(outcome),
        "detail": str(solver.model()) if outcome == z3.sat else "",
    }


def z3_gate(
    builder: Callable[[Mapping[str, Any]], bool],
    name: str = "constraints_proved",
    bits: float = 3.0,
) -> DarkStateGate:
    def check(state: Mapping[str, Any]) -> bool | None:
        if importlib.util.find_spec("z3") is None:
            return None
        return bool(builder(state))

    return DarkStateGate(name, check, bits)


# ---------------------------------------------------------------------------
# Convenience wiring
# ---------------------------------------------------------------------------
def build_standard_gates(
    *,
    include_pytest: bool = True,
    include_ruff: bool = True,
    pytest_targets: Sequence[str] = (),
) -> list[DarkStateGate]:
    gates = [ast_syntax_gate()]
    if include_ruff:
        gates.append(ruff_gate())
    if include_pytest:
        gates.append(pytest_gate(targets=pytest_targets))
    return gates


def verifier_availability() -> dict[str, bool]:
    return {
        "ast": True,
        "ruff": _ruff_command() is not None,
        "pytest": importlib.util.find_spec("pytest") is not None,
        "z3": importlib.util.find_spec("z3") is not None,
    }
