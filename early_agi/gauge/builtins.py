"""Safe Gauge built-ins, symbolic resolution, and dark-state gate bridge.

The VM owns control flow; this module owns the small, auditable vocabulary that
Gauge expressions may call. No Python ``eval`` or arbitrary builtins are exposed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..operators.dark_state import GateResult, run_pytest
from .expressions import ExpressionError, evaluate
from .runtime import ManifoldState

__all__ = ["GaugeBuiltins", "GaugeGate", "SymbolResolver"]

_SYMBOLIC_NAMES = frozenset(
    {
        "identity",
        "dependency_cut",
        "module_cut",
        "cut_graph",
        "coordinate_shift",
        "sink",
        "none",
        "all",
        "true",
        "false",
    }
)


class SymbolResolver:
    """Resolve symbolic enum-like identifiers without exposing Python names."""

    def lookup(self, name: str) -> Any:
        if name in _SYMBOLIC_NAMES:
            return None if name == "none" else name
        raise ExpressionError(f"unknown symbolic identifier {name!r}")

    def call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        raise ExpressionError(f"unknown function {name!r}")


@dataclass
class GaugeGate:
    """Bridge a Gauge predicate into the kernel's tri-state gate protocol."""

    name: str
    predicate: Any
    vm: Any
    bits: float = 1.0

    def evaluate(self, state: Mapping[str, Any]) -> GateResult:
        try:
            value = evaluate(self.predicate, self.vm.execution.evaluation_context(state))
        except Exception as exc:
            return GateResult(self.name, None, self.bits, f"{type(exc).__name__}: {exc}")
        if value is None:
            return GateResult(self.name, None, self.bits, "predicate could not be verified")
        return GateResult(self.name, bool(value), self.bits, "")


class GaugeBuiltins:
    """Factory for expression functions bound to one VM execution."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    def functions(self) -> dict[str, Any]:
        execution = self.vm.execution

        def ast_check(files: Any = None) -> bool | None:
            candidates = (
                list(files)
                if isinstance(files, Sequence) and not isinstance(files, str)
                else list(execution.state.files)
            )
            checked = 0
            for relative in candidates:
                relative = str(relative)
                if not relative.endswith(".py"):
                    continue
                source = execution.state.sources.get(relative)
                if source is None:
                    try:
                        source = (self.vm.root / relative).read_bytes().decode("utf-8", errors="replace")
                    except OSError:
                        return None
                checked += 1
                try:
                    compile(source, relative, "exec")
                except SyntaxError:
                    return False
            return True if checked else None

        def pytest_run(targets: Any = None) -> bool | None:
            candidates = [
                str(item)
                for item in (targets if targets is not None else execution.state.targets)
                if str(item).endswith(".py")
                and ("test" in Path(str(item)).name or "tests" in Path(str(item)).parts)
            ]
            if not candidates:
                return None
            result = run_pytest(self.vm.root, candidates, timeout=min(self.vm.settings.timeout_s, 60.0))
            return result["passed"]

        return {
            "ast_check": ast_check,
            "syntax_check": ast_check,
            "pytest_run": pytest_run,
            "tests_pass": pytest_run,
            "entropy": lambda *_: (
                execution.entanglement.spectral_entropy_bits if execution.entanglement else None
            ),
            "eta": lambda *_: execution.eta,
            "chi": lambda *_: self.vm.kernel.chi_max,
            "truncation": lambda *_: (
                execution.entanglement.truncation_error if execution.entanglement else None
            ),
            "rank": lambda *_: execution.entanglement.effective_rank if execution.entanglement else None,
            "volume_law": lambda *_: execution.entanglement.is_volume_law if execution.entanglement else None,
            "regressions": lambda *_: execution.state.has_regressions,
            "len": len,
            "abs": abs,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "dict": dict,
            "list": list,
        }

    def transform_functions(self, manifold: ManifoldState) -> dict[str, Any]:
        """Return reversible coordinate functions for a transform body."""

        def reencode(target: Any, basis: str = "identity") -> Any:
            subject = target if isinstance(target, ManifoldState) else manifold
            subject.basis = str(basis)
            subject.data["basis"] = str(basis)
            subject.data["reencodes"] = int(subject.data.get("reencodes", 0)) + 1
            return subject

        return {
            "reencode": reencode,
            "rotate_basis": reencode,
            "relabel": reencode,
        }
