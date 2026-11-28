"""Constraint distillation and opcode-footprint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from early_agi.core.adaptation.constraints import distilled_constraint
from early_agi.operators.dark_state import GateResult

__all__ = [
    "_READ_ONLY_FOOTPRINTS",
    "_append_distilled_constraints",
    "_constraint_view",
    "_mutating_footprint",
]


_READ_ONLY_FOOTPRINTS = frozenset({"none", "read", "readonly", "read_only"})


def _mutating_footprint(items: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (items or ()) if str(item).casefold() not in _READ_ONLY_FOOTPRINTS)


def _append_distilled_constraints(execution: Any, results: Mapping[str, GateResult]) -> list[str]:
    """Put one minimal constraint per failed gate on every active manifold."""
    distilled: list[str] = []
    for name, result in results.items():
        if result.passed is not False:
            continue
        constraint = result.constraint or distilled_constraint(name, result.as_dict())
        if not constraint:
            continue
        distilled.append(constraint)
        for manifold in execution.manifolds.values():
            existing = {_constraint_view(item) for item in manifold.constraints}
            if constraint not in existing:
                manifold.constraints.append(constraint)
    execution.distilled_constraints = list(dict.fromkeys([*execution.distilled_constraints, *distilled]))
    return distilled


def _constraint_view(item: Any) -> str:
    return item if isinstance(item, str) else str(item)[:240]
