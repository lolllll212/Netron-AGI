"""Gauge Step 3: lower a checked AST into VM instructions."""

from __future__ import annotations

from . import ast as g
from .checker import CheckedProgram, call_name, validate_program
from .ir import Instruction, ProgramIR
from .span import Span


def _event_opcode(event: str) -> str:
    return {
        "volume_law": "TRAP_VOLUME_LAW",
        "drag": "TRAP_DRAG",
        "frustration": "TRAP_FRUSTRATION",
        "stall": "TRAP_STALL",
        "shift": "TRAP_SHIFT",
        "recall": "TRAP_RECALL",
    }.get(event, "TRAP_GENERIC")


class GaugeCompiler:
    def __init__(self, checked: CheckedProgram) -> None:
        self.checked = checked

    def compile(self) -> ProgramIR:
        setup: list[Instruction] = []
        body: list[Instruction] = []
        for decl in self.checked.program.declarations:
            if isinstance(decl, g.ScaleDecl):
                setup.append(
                    Instruction(
                        "SCALE_INIT",
                        {"uv_scale": decl.uv_scale, "ir_scale": decl.ir_scale},
                        span=decl.span,
                    )
                )
            elif isinstance(decl, g.ManifoldDecl):
                setup.append(
                    Instruction(
                        "MANIFOLD_INIT",
                        {
                            "name": decl.name,
                            "space": decl.space,
                            "constraints": list(decl.constraints),
                        },
                        span=decl.span,
                    )
                )
            elif isinstance(decl, g.GateDecl):
                setup.append(
                    Instruction(
                        "GATE_DEF",
                        {"name": decl.name, "params": list(decl.params), "predicate": decl.predicate},
                        span=decl.span,
                    )
                )
            elif isinstance(decl, g.OpDecl):
                setup.append(
                    Instruction(
                        "OP_DEF",
                        {
                            "name": decl.name,
                            "params": list(decl.params),
                            "deforms": list(decl.deforms),
                            "commutes": list(decl.commutes),
                            "after": list(decl.after),
                        },
                        span=decl.span,
                    )
                )
            elif isinstance(decl, g.TransformDecl):
                setup.append(
                    Instruction(
                        "TRANSFORM_REV",
                        {
                            "name": decl.name,
                            "params": list(decl.params),
                            "return_type": decl.return_type,
                            "body": list(decl.body),
                        },
                        span=decl.span,
                    )
                )
            elif isinstance(decl, g.GovernDecl):
                setup.append(Instruction("GOVERN_BEGIN", {"config": dict(decl.config)}, span=decl.span))
                body.extend(self._compile_block(decl.body))
        body.append(Instruction("HALT", span=self.checked.program.span))
        metadata = {
            "scales": list(self.checked.scales),
            "manifolds": sorted(self.checked.manifolds),
            "gates": sorted(self.checked.gates),
            "ops": sorted(self.checked.ops),
            "transforms": sorted(self.checked.transforms),
            "op_order": list(self.checked.op_order),
            "warnings": [issue.as_dict() for issue in self.checked.report.warnings],
        }
        return ProgramIR(tuple(setup), tuple(body), metadata, self.checked.program.span)

    def _compile_block(self, stmts: list[g.Statement]) -> list[Instruction]:
        out: list[Instruction] = []
        for stmt in stmts:
            out.extend(self._compile_stmt(stmt))
        return out

    def _compile_stmt(self, stmt: g.Statement) -> list[Instruction]:
        if isinstance(stmt, g.EvolveStmt):
            instructions: list[Instruction] = []
            for op_name in self.checked.op_order:
                op = self.checked.ops[op_name]
                instructions.append(
                    Instruction(
                        "OP_APPLY",
                        {"name": op.name, "params": list(op.params)},
                        span=stmt.span,
                    )
                )
            instructions.extend(
                [
                    Instruction("ENT_MEASURE", {"manifold": stmt.manifold}, span=stmt.span),
                    Instruction("GATE_PROJECT", {"gates": list(stmt.gates)}, span=stmt.span),
                    Instruction("GOVERN_CHECK", {}, span=stmt.span),
                    Instruction(
                        "EVOLVE", {"manifold": stmt.manifold, "gates": list(stmt.gates)}, span=stmt.span
                    ),
                ]
            )
            return instructions
        if isinstance(stmt, g.TrapHandler):
            return [
                Instruction(
                    _event_opcode(stmt.event),
                    {"event": stmt.event, "binding": stmt.binding},
                    body=tuple(self._compile_block(stmt.body)),
                    span=stmt.span,
                )
            ]
        if isinstance(stmt, g.SynthesizeStmt):
            return [
                Instruction(
                    "SYNTHESIZE_OP",
                    {"op_name": stmt.op_name, "metric": stmt.target_metric},
                    span=stmt.span,
                )
            ]
        if isinstance(stmt, g.RecallStmt):
            return [Instruction("RECALL_SINK", {"condition": stmt.condition}, span=stmt.span)]
        if isinstance(stmt, g.CollapseStmt):
            return [
                Instruction(
                    "COLLAPSE_SINK",
                    {"target": stmt.target, "manifold": stmt.manifold, "sink": stmt.sink},
                    span=stmt.span,
                )
            ]
        if isinstance(stmt, g.UncallStmt):
            return [
                Instruction(
                    "UNCALL_TRANSFORM",
                    {"transform": stmt.transform, "args": list(stmt.args)},
                    span=stmt.span,
                )
            ]
        if isinstance(stmt, g.AssignStmt):
            expr = stmt.value
            if isinstance(expr, g.CallExpr) and call_name(expr.callee) in self.checked.transforms:
                return [
                    Instruction(
                        "TRANSFORM_APPLY",
                        {
                            "target": stmt.target,
                            "transform": call_name(expr.callee),
                            "args": list(expr.args),
                            "kwargs": dict(expr.kwargs),
                        },
                        span=stmt.span,
                    )
                ]
            return [Instruction("ASSIGN", {"target": stmt.target, "value": expr}, span=stmt.span)]
        if isinstance(stmt, g.ExprStmt):
            expr = stmt.expr
            name = call_name(expr.callee) if isinstance(expr, g.CallExpr) else None
            if name in self.checked.ops:
                return [Instruction("OP_APPLY", {"name": name, "expression": expr}, span=stmt.span)]
            if name in self.checked.transforms:
                return [
                    Instruction(
                        "TRANSFORM_APPLY",
                        {
                            "target": None,
                            "transform": name,
                            "args": list(expr.args),
                            "kwargs": dict(expr.kwargs),
                        },
                        span=stmt.span,
                    )
                ]
            return [Instruction("EXPR", {"value": expr}, span=stmt.span)]
        return [Instruction("NOOP", {"statement": type(stmt).__name__}, span=getattr(stmt, "span", Span()))]


def compile_program(program: g.Program | str | CheckedProgram) -> ProgramIR:
    checked = program if isinstance(program, CheckedProgram) else validate_program(program)
    return GaugeCompiler(checked).compile()


compile_gauge = compile_program

__all__ = ["GaugeCompiler", "compile_gauge", "compile_program"]
