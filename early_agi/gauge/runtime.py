"""Runtime objects shared by the Gauge VM and expression evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import ast as g


def _constraint_atom(node: Any) -> str:
    if isinstance(node, Mapping):
        for key in ("name", "attr", "value"):
            value = node.get(key)
            if value is not None:
                return str(value)
        return str(node)[:120]
    return str(node)


def _constraint_view(item: Any) -> str:
    """A compact, JSON-safe rendering; never a raw AST/stack-trace dump."""
    if isinstance(item, str):
        return item[:240]
    if isinstance(item, Mapping):
        left = _constraint_atom(item.get("left"))
        right = _constraint_atom(item.get("right"))
        operator = str(item.get("op", ""))
        if left or right:
            return f"{left} {operator} {right}".strip()[:240]
    return str(item)[:240]


@dataclass(frozen=True, slots=True)
class CandidateView:
    files: tuple[str, ...]
    targets: tuple[str, ...]
    has_regressions: bool = False


@dataclass(slots=True)
class GaugeState:
    root: Path
    files: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    has_regressions: bool = False
    task: str = ""
    scale: int = 0

    @property
    def ast(self) -> SimpleNamespace:
        return SimpleNamespace(modules=tuple(self.files))

    @property
    def repo(self) -> SimpleNamespace:
        return SimpleNamespace(
            root=str(self.root),
            ast=self.ast,
            modules=tuple(self.files),
        )

    @property
    def candidate(self) -> CandidateView:
        return CandidateView(tuple(self.files), tuple(self.targets), self.has_regressions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.root),
            "paths": list(self.files),
            "sources": dict(self.sources),
            "task": self.task,
            "scale": self.scale,
            "repo": self.repo,
            "p": self.candidate,
            "scope": list(self.files),
            "max_tokens": 0,
            "entanglement_bound": 0.0,
            "has_regressions": self.has_regressions,
        }


@dataclass(slots=True)
class ManifoldState:
    name: str
    space: str
    constraints: list[Any]
    scale: int = 0
    basis: str = "identity"
    collapsed: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "data": dict(self.data),
            "collapsed": self.collapsed,
            "constraints": list(self.constraints),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.basis = str(snapshot.get("basis", "identity"))
        self.data = dict(snapshot.get("data", {}))
        self.collapsed = bool(snapshot.get("collapsed", False))
        self.constraints = list(snapshot.get("constraints", ()))

    def active_constraints(self) -> list[str]:
        return [
            _constraint_view(item)
            for item in self.constraints
            if isinstance(item, str) and item.startswith(("assert ", "counterexample:"))
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "space": self.space,
            "scale": self.scale,
            "basis": self.basis,
            "collapsed": self.collapsed,
            "data": dict(self.data),
            "constraints": [_constraint_view(item) for item in self.constraints],
            "active_constraints": self.active_constraints(),
        }


@dataclass(slots=True)
class ReversibleTransform:
    decl: g.TransformDecl
    vm: Any

    def apply(
        self,
        manifold: ManifoldState,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> ManifoldState:
        execution = self.vm.execution
        previous = manifold.snapshot()
        execution.inverse_stack.append((self.decl.name, manifold.name, previous))
        context = execution.transform_context(self.decl, manifold, args, kwargs)
        for stmt in self.decl.body:
            execution.execute_transform_statement(stmt, context)
        return manifold

    def uncall(self, manifold: ManifoldState | None = None) -> ManifoldState | None:
        execution = self.vm.execution
        for index in range(len(execution.inverse_stack) - 1, -1, -1):
            name, manifold_name, snapshot = execution.inverse_stack[index]
            if name != self.decl.name:
                continue
            if manifold is not None and manifold.name != manifold_name:
                continue
            target = manifold or execution.manifolds.get(manifold_name)
            if target is not None:
                target.restore(snapshot)
            del execution.inverse_stack[index]
            return target
        return manifold


__all__ = ["CandidateView", "GaugeState", "ManifoldState", "ReversibleTransform"]
