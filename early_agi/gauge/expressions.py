"""Safe, tri-state evaluator for Gauge expressions. No Python eval()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import ast as g
from .checker import call_name
from .span import Span


class ExpressionError(RuntimeError):
    def __init__(self, message: str, span: Span | None = None) -> None:
        super().__init__(f"{message} at {span}" if span is not None else message)
        self.message = message
        self.span = span


@dataclass(slots=True)
class EvaluationContext:
    variables: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    resolver: Any | None = None

    def lookup(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        if name in self.state:
            return self.state[name]
        if len(name) > 1 and name[0] == "L" and name[1:].isdigit():
            return int(name[1:])
        if self.resolver is not None and hasattr(self.resolver, "lookup"):
            return self.resolver.lookup(name)
        raise ExpressionError(f"unknown identifier {name!r}")

    def call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        fn = self.functions.get(name)
        if fn is None and self.resolver is not None and hasattr(self.resolver, "call"):
            return self.resolver.call(name, args, kwargs)
        if fn is None:
            raise ExpressionError(f"unknown function {name!r}")
        return fn(*args, **kwargs)


def get_member(obj: Any, attr: str, span: Span | None = None) -> Any:
    if isinstance(obj, dict) and attr in obj:
        return obj[attr]
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if attr in {"length", "len"} and isinstance(obj, (list, tuple, str, bytes, dict)):
        return len(obj)
    raise ExpressionError(f"object has no member {attr!r}", span)


def _tri_not(value: Any) -> Any:
    if value is None:
        return None
    return not bool(value)


def _tri_and(left: Any, right: Any) -> Any:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return bool(left) and bool(right)


def _tri_or(left: Any, right: Any) -> Any:
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return bool(left) or bool(right)


def _compare(op: str, left: Any, right: Any, span: Span) -> Any:
    if left is None or right is None:
        return None
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "in":
            return left in right
    except TypeError as exc:
        raise ExpressionError(f"invalid comparison {op}: {exc}", span) from exc
    raise ExpressionError(f"unknown comparison {op!r}", span)


def _arithmetic(op: str, left: Any, right: Any, span: Span) -> Any:
    if left is None or right is None:
        return None
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
    except (TypeError, ZeroDivisionError) as exc:
        raise ExpressionError(f"invalid arithmetic {op}: {exc}", span) from exc
    raise ExpressionError(f"unknown operator {op!r}", span)


def evaluate(expr: Any, context: EvaluationContext) -> Any:
    if isinstance(expr, g.LiteralExpr):
        return expr.value
    if isinstance(expr, g.ListExpr):
        return [evaluate(item, context) for item in expr.elements]
    if isinstance(expr, g.IdentExpr):
        return context.lookup(expr.name)
    if isinstance(expr, g.MemberExpr):
        obj = evaluate(expr.obj, context)
        return get_member(obj, expr.attr, expr.span)
    if isinstance(expr, g.UnaryExpr):
        operand = evaluate(expr.operand, context)
        if expr.op == "not":
            return _tri_not(operand)
        if expr.op == "-":
            return None if operand is None else -operand
        raise ExpressionError(f"unknown unary operator {expr.op!r}", expr.span)
    if isinstance(expr, g.BinaryExpr):
        if expr.op == "and":
            left = evaluate(expr.left, context)
            if left is False:
                return False
            return _tri_and(left, evaluate(expr.right, context))
        if expr.op == "or":
            left = evaluate(expr.left, context)
            if left is True:
                return True
            return _tri_or(left, evaluate(expr.right, context))
        left = evaluate(expr.left, context)
        right = evaluate(expr.right, context)
        if expr.op in {"==", "!=", "<", "<=", ">", ">=", "in"}:
            return _compare(expr.op, left, right, expr.span)
        return _arithmetic(expr.op, left, right, expr.span)
    if isinstance(expr, g.CallExpr):
        name = call_name(expr.callee)
        if name is None:
            raise ExpressionError("call target must be an identifier or member", expr.span)
        args = [evaluate(item, context) for item in expr.args]
        kwargs = {key: evaluate(item, context) for key, item in expr.kwargs.items()}
        return context.call(name, args, kwargs)
    raise ExpressionError(f"cannot evaluate {type(expr).__name__}", getattr(expr, "span", None))


evaluate_expr = evaluate

__all__ = ["EvaluationContext", "ExpressionError", "evaluate", "evaluate_expr", "get_member"]
