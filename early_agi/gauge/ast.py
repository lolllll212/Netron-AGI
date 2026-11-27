"""Span-tagged Gauge AST nodes and deterministic serialization."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from .span import Span

Expr = Any
Statement = Any
Declaration = Any


@dataclass(frozen=True, slots=True)
class Program:
    declarations: list[Declaration]
    span: Span


@dataclass(frozen=True, slots=True)
class ScaleDecl:
    uv_scale: str
    ir_scale: str
    span: Span


@dataclass(frozen=True, slots=True)
class ManifoldDecl:
    name: str
    space: str
    constraints: list[Expr]
    span: Span


@dataclass(frozen=True, slots=True)
class GateDecl:
    name: str
    params: list[str]
    predicate: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class OpDecl:
    name: str
    params: list[str]
    deforms: list[str]
    commutes: list[str]
    after: list[str]
    span: Span


@dataclass(frozen=True, slots=True)
class TransformDecl:
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: list[Statement]
    reversible: bool
    span: Span


@dataclass(frozen=True, slots=True)
class GovernDecl:
    config: dict[str, Expr]
    body: list[Statement]
    span: Span
    plastic: bool = False


@dataclass(frozen=True, slots=True)
class EvolveStmt:
    manifold: str
    gates: list[str]
    span: Span


@dataclass(frozen=True, slots=True)
class TrapHandler:
    event: str
    binding: str | None
    body: list[Statement]
    span: Span


@dataclass(frozen=True, slots=True)
class SynthesizeStmt:
    op_name: str
    target_metric: str
    span: Span


@dataclass(frozen=True, slots=True)
class RecallStmt:
    condition: Expr
    span: Span
    kind: str = "sink"


@dataclass(frozen=True, slots=True)
class ProbeStmt:
    manifold: str
    ops: list[str]
    span: Span


@dataclass(frozen=True, slots=True)
class CollapseStmt:
    target: str | None
    manifold: str
    sink: str
    span: Span


@dataclass(frozen=True, slots=True)
class UncallStmt:
    transform: str
    args: list[Expr]
    span: Span


@dataclass(frozen=True, slots=True)
class AssignStmt:
    target: str
    value: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class ExprStmt:
    expr: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class UnaryExpr:
    op: str
    operand: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class BinaryExpr:
    op: str
    left: Expr
    right: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class MemberExpr:
    obj: Expr
    attr: str
    span: Span


@dataclass(frozen=True, slots=True)
class CallExpr:
    callee: Expr
    args: list[Expr] = field(default_factory=list)
    kwargs: dict[str, Expr] = field(default_factory=dict)
    span: Span = field(default_factory=Span)


@dataclass(frozen=True, slots=True)
class IdentExpr:
    name: str
    span: Span


@dataclass(frozen=True, slots=True)
class LiteralExpr:
    value: int | float | str | bool | None
    span: Span


@dataclass(frozen=True, slots=True)
class ListExpr:
    elements: list[Expr]
    span: Span


def node_to_dict(value: Any) -> Any:
    """Convert AST nodes to JSON-compatible dictionaries with explicit node tags."""
    if is_dataclass(value):
        out: dict[str, Any] = {"node": type(value).__name__}
        for item in fields(value):
            out[item.name] = node_to_dict(getattr(value, item.name))
        return out
    if isinstance(value, (list, tuple)):
        return [node_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): node_to_dict(item) for key, item in value.items()}
    return value


expr_to_dict = node_to_dict

__all__ = [
    "AssignStmt",
    "BinaryExpr",
    "CallExpr",
    "CollapseStmt",
    "Declaration",
    "EvolveStmt",
    "Expr",
    "ExprStmt",
    "GateDecl",
    "GovernDecl",
    "IdentExpr",
    "ListExpr",
    "LiteralExpr",
    "ManifoldDecl",
    "MemberExpr",
    "OpDecl",
    "ProbeStmt",
    "Program",
    "RecallStmt",
    "ScaleDecl",
    "Statement",
    "SynthesizeStmt",
    "TransformDecl",
    "TrapHandler",
    "UnaryExpr",
    "UncallStmt",
    "expr_to_dict",
    "node_to_dict",
]
