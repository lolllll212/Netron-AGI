"""Compact failure-to-constraint distillation for gate results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["distilled_constraint"]


def distilled_constraint(name: str, result: Mapping[str, Any]) -> str | None:
    """Turn a binary gate failure into one minimal, actionable assertion.

    Only the failing assertion or SMT counterexample is retained. Raw stack
    traces and tool diagnostics stay out of prompt/state history.
    """
    if result.get("passed") is not False and result.get("verdict") != "fail":
        return None
    constraint = str(result.get("constraint") or "").strip()
    if constraint:
        return constraint
    assertion = str(result.get("assertion") or "").strip()
    if assertion:
        return assertion if assertion.startswith("assert ") else f"assert {assertion}"
    model = result.get("model") or result.get("counterexample")
    if model:
        rendered = (
            str(model)
            if not isinstance(model, Mapping)
            else ", ".join(f"{key} == {value!r}" for key, value in sorted(model.items()))
        )
        return f"counterexample: {rendered[:240]}"
    failures = result.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) and failures:
        first = failures[0]
        if isinstance(first, Mapping):
            assertion = str(first.get("assertion") or first.get("message") or "").strip()
            if assertion:
                return assertion if assertion.startswith("assert ") else f"assert {assertion}"
    return f"assert not {name}_failed"
