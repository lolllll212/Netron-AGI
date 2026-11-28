"""Gauge Step 2: static algebra, ordering, reversibility and no-leak checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any

from . import ast as g
from .parser import parse_program
from .span import GaugeSemanticError, Span

IRREVERSIBLE_CALLS = {
    "collapse",
    "recall_sink",
    "recall",
    "synthesize",
    "write_file",
    "run_shell",
    "commit",
    "delete",
    "overwrite",
    "drop",
    "discard",
}
KNOWN_EVENTS = {"volume_law", "drag", "frustration", "stall", "shift", "recall", "high_novelty"}
KNOWN_METRICS = {"eta", "ent", "chi", "loss", "frustration"}
KNOWN_GOVERN_OPTIONS = {
    "eta_min",
    "chi_max",
    "eta_high",
    "tau",
    "window",
    "stall_ticks",
    "gate_bits",
    "adaptation_rate",
}
READ_ONLY_FOOTPRINTS = frozenset({"none", "read", "readonly", "read_only"})


def _mutating_footprint(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in items if str(item).casefold() not in READ_ONLY_FOOTPRINTS)


@dataclass(frozen=True, slots=True)
class CheckIssue:
    level: str  # "error" | "warning"
    message: str
    span: Span

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, "span": self.span.as_dict()}


@dataclass(frozen=True, slots=True)
class CheckReport:
    issues: tuple[CheckIssue, ...] = ()

    @property
    def errors(self) -> tuple[CheckIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "error")

    @property
    def warnings(self) -> tuple[CheckIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            first = self.errors[0]
            detail = "; ".join(issue.message for issue in self.errors[:5])
            raise GaugeSemanticError(detail, first.span)


@dataclass(frozen=True, slots=True)
class CheckedProgram:
    program: g.Program
    report: CheckReport
    scales: tuple[str, ...]
    manifolds: dict[str, g.ManifoldDecl]
    gates: dict[str, g.GateDecl]
    ops: dict[str, g.OpDecl]
    transforms: dict[str, g.TransformDecl]
    governs: tuple[g.GovernDecl, ...]
    op_order: tuple[str, ...]


def expr_nodes(expr: Any) -> Iterable[Any]:
    if is_dataclass(expr):
        yield expr
        for item in fields(expr):
            if item.name == "span":
                continue
            yield from expr_nodes(getattr(expr, item.name))
    elif isinstance(expr, dict):
        for value in expr.values():
            yield from expr_nodes(value)
    elif isinstance(expr, (list, tuple)):
        for value in expr:
            yield from expr_nodes(value)


def walk_statements(stmts: Iterable[g.Statement]) -> Iterable[g.Statement]:
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, g.TrapHandler):
            yield from walk_statements(stmt.body)


def call_name(expr: Any) -> str | None:
    if isinstance(expr, g.IdentExpr):
        return expr.name
    if isinstance(expr, g.MemberExpr):
        return expr.attr
    return None


def literal_number(expr: Any) -> float | None:
    if (
        isinstance(expr, g.LiteralExpr)
        and isinstance(expr.value, (int, float))
        and not isinstance(expr.value, bool)
    ):
        return float(expr.value)
    return None


def scale_number(name: str) -> int | None:
    match = re.fullmatch(r"L(\d+)", name)
    return int(match.group(1)) if match else None


def _collect(program: g.Program) -> tuple[dict[str, list[Any]], list[CheckIssue]]:
    buckets: dict[str, list[Any]] = {
        "scale": [],
        "manifold": [],
        "gate": [],
        "op": [],
        "transform": [],
        "govern": [],
    }
    issues: list[CheckIssue] = []
    for decl in program.declarations:
        if isinstance(decl, g.ScaleDecl):
            buckets["scale"].append(decl)
        elif isinstance(decl, g.ManifoldDecl):
            buckets["manifold"].append(decl)
        elif isinstance(decl, g.GateDecl):
            buckets["gate"].append(decl)
        elif isinstance(decl, g.OpDecl):
            buckets["op"].append(decl)
        elif isinstance(decl, g.TransformDecl):
            buckets["transform"].append(decl)
        elif isinstance(decl, g.GovernDecl):
            buckets["govern"].append(decl)
        else:
            issues.append(CheckIssue("error", f"unknown declaration {type(decl).__name__}", decl.span))
    return buckets, issues


def _check_op_order(ops: dict[str, g.OpDecl], issues: list[CheckIssue]) -> tuple[str, ...]:
    for op in ops.values():
        for dependency in op.after:
            if dependency not in ops:
                issues.append(
                    CheckIssue(
                        "error",
                        f"operator {op.name!r} has after({dependency!r}) but that operator is not declared",
                        op.span,
                    )
                )
        for peer in op.commutes:
            if peer not in {"none", "all", "*"} and peer not in ops:
                issues.append(
                    CheckIssue(
                        "error",
                        f"operator {op.name!r} commutes with undeclared operator {peer!r}",
                        op.span,
                    )
                )
        overlap = sorted(set(op.after) & {peer for peer in op.commutes if peer != "none"})
        if overlap:
            issues.append(
                CheckIssue(
                    "warning",
                    f"operator {op.name!r} declares both after() and commutes() for {', '.join(overlap)}",
                    op.span,
                )
            )
        if not op.deforms:
            issues.append(
                CheckIssue("warning", f"operator {op.name!r} declares no deforms() footprint", op.span)
            )

    sorter = TopologicalSorter()
    for name, op in ops.items():
        sorter.add(name, *[dep for dep in op.after if dep in ops])
    try:
        return tuple(sorter.static_order())
    except CycleError:
        issues.append(CheckIssue("error", "extern op after() graph contains a cycle", Span()))
        return tuple(ops)


def _check_transform(
    decl: g.TransformDecl,
    manifolds: dict[str, g.ManifoldDecl],
    ops: dict[str, g.OpDecl],
    transforms: dict[str, g.TransformDecl],
    issues: list[CheckIssue],
) -> None:
    local_names = {name for name, _type in decl.params}
    for stmt in walk_statements(decl.body):
        if isinstance(
            stmt,
            (
                g.CollapseStmt,
                g.RecallStmt,
                g.SynthesizeStmt,
                g.EvolveStmt,
                g.TrapHandler,
                g.ProbeStmt,
                g.RaceStmt,
            ),
        ):
            issues.append(
                CheckIssue(
                    "error",
                    f"reversible transform {decl.name!r} contains irreversible statement {type(stmt).__name__}",
                    stmt.span,
                )
            )
            continue
        if isinstance(stmt, g.AssignStmt):
            issues.append(
                CheckIssue(
                    "error",
                    f"reversible transform {decl.name!r} cannot assign to {stmt.target!r}",
                    stmt.span,
                )
            )
        if isinstance(stmt, g.UncallStmt) and stmt.transform not in transforms:
            issues.append(
                CheckIssue(
                    "error",
                    f"uncall references unknown transform {stmt.transform!r}",
                    stmt.span,
                )
            )
        expressions: list[g.Expr] = []
        if isinstance(stmt, g.ExprStmt):
            expressions.append(stmt.expr)
        elif isinstance(stmt, g.AssignStmt):
            expressions.append(stmt.value)
        elif isinstance(stmt, g.UncallStmt):
            expressions.extend(stmt.args)
        for expr in expressions:
            for node in expr_nodes(expr):
                if isinstance(node, g.CallExpr):
                    name = call_name(node.callee)
                    if name in IRREVERSIBLE_CALLS or name in ops:
                        issues.append(
                            CheckIssue(
                                "error",
                                f"reversible transform {decl.name!r} calls irreversible operation {name!r}",
                                node.span,
                            )
                        )
                if isinstance(node, g.IdentExpr) and node.name in manifolds and node.name not in local_names:
                    issues.append(
                        CheckIssue(
                            "warning",
                            f"reversible transform {decl.name!r} references global manifold {node.name!r}",
                            node.span,
                        )
                    )


def _check_recall_condition(cond: g.Expr, scales: set[str], issues: list[CheckIssue]) -> None:
    for node in expr_nodes(cond):
        if isinstance(node, g.IdentExpr) and re.fullmatch(r"L\d+", node.name) and node.name not in scales:
            issues.append(
                CheckIssue("error", f"recall condition references undeclared scale {node.name!r}", node.span)
            )


def _check_analogy_condition(cond: g.Expr, issues: list[CheckIssue]) -> None:
    if not isinstance(cond, g.BinaryExpr) or cond.op not in {"==", ">=", ">", "<=", "<"}:
        issues.append(CheckIssue("error", "recall analogy needs a similarity comparison", cond.span))
        return
    threshold = literal_number(cond.right)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        issues.append(
            CheckIssue("error", "analogy similarity threshold must be a number in [0, 1]", cond.span)
        )
    if not any(isinstance(node, g.IdentExpr) and node.name == "similarity" for node in expr_nodes(cond.left)):
        issues.append(CheckIssue("warning", "analogy condition should reference 'similarity'", cond.span))


def _check_race(
    stmt: g.RaceStmt,
    *,
    manifolds: dict[str, g.ManifoldDecl],
    gates: dict[str, g.GateDecl],
    ops: dict[str, g.OpDecl],
    transforms: dict[str, g.TransformDecl],
    issues: list[CheckIssue],
) -> None:
    """Race branches must be reversible/read-only and scoped to declared names."""
    branches = [str(item) for item in stmt.branches]
    race_gates = [str(item) for item in stmt.gates]
    if not 2 <= len(branches) <= 4:
        issues.append(CheckIssue("error", "race requires 2 to 4 distinct branch names", stmt.span))
    if len(set(branches)) != len(branches):
        issues.append(CheckIssue("error", "race branch names must be distinct", stmt.span))
    if not race_gates:
        issues.append(CheckIssue("error", "race requires at least one gate under []", stmt.span))
    for gate_name in race_gates:
        if gate_name not in gates:
            issues.append(CheckIssue("error", f"race references unknown gate {gate_name!r}", stmt.span))
    for branch in branches:
        op = ops.get(branch)
        transform = transforms.get(branch)
        if op is None and transform is None:
            issues.append(
                CheckIssue(
                    "error",
                    f"race branch {branch!r} must name a reversible transform or read-only extern op",
                    stmt.span,
                )
            )
        elif op is not None and _mutating_footprint(op.deforms):
            issues.append(
                CheckIssue(
                    "error",
                    f"race branch {branch!r} is not reversible: extern op has a mutating footprint",
                    stmt.span,
                )
            )
        elif branch in manifolds:
            issues.append(
                CheckIssue(
                    "error",
                    f"race branch {branch!r} shadows global manifold {branch!r}",
                    stmt.span,
                )
            )
    reserved = set(branches) | set(manifolds) | set(gates) | set(ops) | set(transforms)
    if stmt.winner in reserved:
        issues.append(
            CheckIssue(
                "error",
                f"race winner {stmt.winner!r} must be a fresh scoped variable",
                stmt.span,
            )
        )


def _check_govern(
    decl: g.GovernDecl,
    manifolds: dict[str, g.ManifoldDecl],
    gates: dict[str, g.GateDecl],
    ops: dict[str, g.OpDecl],
    transforms: dict[str, g.TransformDecl],
    scales: set[str],
    issues: list[CheckIssue],
) -> set[str]:
    collapsed: set[str] = set()
    eta_min = literal_number(decl.config.get("eta_min"))
    chi_max = literal_number(decl.config.get("chi_max"))
    if eta_min is None or eta_min < 0:
        issues.append(CheckIssue("error", "govern requires numeric eta_min >= 0", decl.span))
    if chi_max is None or chi_max < 1 or not float(chi_max).is_integer():
        issues.append(CheckIssue("error", "govern requires integer chi_max >= 1", decl.span))
    for key, value in decl.config.items():
        if key not in KNOWN_GOVERN_OPTIONS:
            issues.append(CheckIssue("warning", f"unknown govern option {key!r}", decl.span))
        elif literal_number(value) is None and key in {
            "eta_high",
            "tau",
            "window",
            "stall_ticks",
            "gate_bits",
            "adaptation_rate",
        }:
            issues.append(CheckIssue("error", f"govern option {key!r} must be numeric", decl.span))
    if decl.plastic and "adaptation_rate" not in decl.config:
        issues.append(CheckIssue("warning", "plastic govern usually declares adaptation_rate", decl.span))

    for stmt in walk_statements(decl.body):
        if isinstance(stmt, g.EvolveStmt):
            if stmt.manifold not in manifolds:
                issues.append(
                    CheckIssue("error", f"evolve references unknown manifold {stmt.manifold!r}", stmt.span)
                )
            for gate_name in stmt.gates:
                if gate_name not in gates:
                    issues.append(
                        CheckIssue("error", f"evolve references unknown gate {gate_name!r}", stmt.span)
                    )
        elif isinstance(stmt, g.TrapHandler):
            if stmt.event not in KNOWN_EVENTS:
                issues.append(CheckIssue("error", f"unknown trap {stmt.event!r}", stmt.span))
        elif isinstance(stmt, g.SynthesizeStmt):
            if not stmt.op_name.isidentifier():
                issues.append(
                    CheckIssue(
                        "error", f"synthesized operator name {stmt.op_name!r} is not an identifier", stmt.span
                    )
                )
            if stmt.target_metric not in KNOWN_METRICS:
                issues.append(
                    CheckIssue("warning", f"unusual synthesis metric {stmt.target_metric!r}", stmt.span)
                )
        elif isinstance(stmt, g.RaceStmt):
            _check_race(
                stmt,
                manifolds=manifolds,
                gates=gates,
                ops=ops,
                transforms=transforms,
                issues=issues,
            )
        elif isinstance(stmt, g.RecallStmt):
            if stmt.kind == "analogy":
                _check_analogy_condition(stmt.condition, issues)
            else:
                _check_recall_condition(stmt.condition, scales, issues)
        elif isinstance(stmt, g.ProbeStmt):
            if stmt.manifold not in manifolds:
                issues.append(
                    CheckIssue("error", f"probe references unknown manifold {stmt.manifold!r}", stmt.span)
                )
            for op_name in stmt.ops:
                op = ops.get(op_name)
                if op is None:
                    issues.append(
                        CheckIssue("error", f"probe references unknown operator {op_name!r}", stmt.span)
                    )
                elif _mutating_footprint(op.deforms):
                    issues.append(
                        CheckIssue("error", f"probe operator {op_name!r} must be read-only", stmt.span)
                    )
        elif isinstance(stmt, g.CollapseStmt):
            if stmt.manifold not in manifolds:
                issues.append(
                    CheckIssue("error", f"collapse references unknown manifold {stmt.manifold!r}", stmt.span)
                )
            else:
                collapsed.add(stmt.manifold)
            if stmt.sink != "sink":
                issues.append(CheckIssue("error", "collapse must route to 'sink'", stmt.span))
        elif isinstance(stmt, g.UncallStmt):
            if stmt.transform not in transforms:
                issues.append(
                    CheckIssue("error", f"uncall references unknown transform {stmt.transform!r}", stmt.span)
                )
        elif isinstance(stmt, (g.AssignStmt, g.ExprStmt)):
            expr = stmt.value if isinstance(stmt, g.AssignStmt) else stmt.expr
            for node in expr_nodes(expr):
                if isinstance(node, g.CallExpr):
                    name = call_name(node.callee)
                    if name in transforms and isinstance(stmt, g.AssignStmt):
                        continue
                    if name in ops:
                        continue
                    if name == "collapse":
                        issues.append(
                            CheckIssue(
                                "error", "collapse must be written as 'collapse(...) -> sink'", node.span
                            )
                        )
                if isinstance(node, g.IdentExpr) and node.name in manifolds and isinstance(stmt, g.ExprStmt):
                    issues.append(
                        CheckIssue("warning", f"manifold {node.name!r} is used without collapse", node.span)
                    )
    return collapsed


def check_program(program: g.Program | str) -> CheckReport:
    if isinstance(program, str):
        program = parse_program(program)
    buckets, issues = _collect(program)

    scales: set[str] = set()
    for decl in buckets["scale"]:
        uv, ir = scale_number(decl.uv_scale), scale_number(decl.ir_scale)
        if uv is None or ir is None:
            issues.append(
                CheckIssue(
                    "error",
                    f"scale names must look like L0, L1, L2; got {decl.uv_scale!r} -> {decl.ir_scale!r}",
                    decl.span,
                )
            )
        elif uv >= ir:
            issues.append(
                CheckIssue(
                    "error", "scale declaration must move from a finer L-number to a coarser one", decl.span
                )
            )
        scales.update((decl.uv_scale, decl.ir_scale))

    def index(items: list[Any], kind: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in items:
            if item.name in out:
                issues.append(CheckIssue("error", f"duplicate {kind} declaration {item.name!r}", item.span))
            out[item.name] = item
        return out

    manifolds = index(buckets["manifold"], "manifold")
    gates = index(buckets["gate"], "gate")
    ops = index(buckets["op"], "extern op")
    transforms = index(buckets["transform"], "reversible transform")
    governs = tuple(buckets["govern"])

    for decl in buckets["manifold"]:
        for constraint in decl.constraints:
            if (
                isinstance(constraint, g.BinaryExpr)
                and constraint.op in {"<=", ">=", "<", ">"}
                and literal_number(constraint.right) is None
            ):
                issues.append(
                    CheckIssue(
                        "error", "manifold numeric bound must compare against a literal", constraint.span
                    )
                )
            for node in expr_nodes(constraint):
                if isinstance(node, g.CallExpr) and call_name(node.callee) in IRREVERSIBLE_CALLS:
                    issues.append(
                        CheckIssue(
                            "error", "manifold constraints cannot call irreversible operations", node.span
                        )
                    )

    for decl in buckets["gate"]:
        for node in expr_nodes(decl.predicate):
            if isinstance(node, g.CallExpr):
                name = call_name(node.callee)
                if name in IRREVERSIBLE_CALLS:
                    issues.append(
                        CheckIssue(
                            "error", f"gate {decl.name!r} calls irreversible operation {name!r}", node.span
                        )
                    )
            if isinstance(node, g.IdentExpr) and node.name in manifolds:
                issues.append(
                    CheckIssue(
                        "error", f"gate {decl.name!r} reads manifold {node.name!r} directly", node.span
                    )
                )

    _check_op_order(ops, issues)
    for transform in transforms.values():
        _check_transform(transform, manifolds, ops, transforms, issues)

    collapsed: set[str] = set()
    for govern in governs:
        collapsed |= _check_govern(govern, manifolds, gates, ops, transforms, scales, issues)
    if not governs:
        issues.append(CheckIssue("error", "program has no govern block", program.span))
    for name in manifolds:
        if name not in collapsed:
            issues.append(
                CheckIssue(
                    "error",
                    f"manifold {name!r} leaks uncollapsed; add collapse({name}) -> sink",
                    program.span,
                )
            )

    return CheckReport(tuple(issues))


def validate_program(program: g.Program | str) -> CheckedProgram:
    if isinstance(program, str):
        program = parse_program(program)
    report = check_program(program)
    report.raise_for_errors()
    buckets, _ = _collect(program)
    scales = tuple(
        dict.fromkeys(name for decl in buckets["scale"] for name in (decl.uv_scale, decl.ir_scale))
    )
    manifolds = {decl.name: decl for decl in buckets["manifold"]}
    gates = {decl.name: decl for decl in buckets["gate"]}
    ops = {decl.name: decl for decl in buckets["op"]}
    transforms = {decl.name: decl for decl in buckets["transform"]}
    sorter = TopologicalSorter()
    for name, op in ops.items():
        sorter.add(name, *[dep for dep in op.after if dep in ops])
    op_order = tuple(sorter.static_order())
    return CheckedProgram(
        program=program,
        report=report,
        scales=scales,
        manifolds=manifolds,
        gates=gates,
        ops=ops,
        transforms=transforms,
        governs=tuple(buckets["govern"]),
        op_order=op_order,
    )


__all__ = [
    "IRREVERSIBLE_CALLS",
    "CheckIssue",
    "CheckReport",
    "CheckedProgram",
    "call_name",
    "check_program",
    "expr_nodes",
    "literal_number",
    "scale_number",
    "validate_program",
    "walk_statements",
]
