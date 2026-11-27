"""Trivial assignment, expression, and no-op opcode handlers."""

from __future__ import annotations

from typing import Any

from early_agi.gauge.expressions import evaluate
from early_agi.gauge.ir import Instruction

__all__ = ["_ExpressionOps"]


class _ExpressionOps:
    """Expression opcode handlers."""

    def _op_assign(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        value = evaluate(args["value"], self.execution.evaluation_context())
        self.execution.variables[str(args["target"])] = value
        return {"target": args["target"], "value": str(value)}

    def _op_expr(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"value": str(evaluate(args["value"], self.execution.evaluation_context()))}

    def _op_noop(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"noop": args}
