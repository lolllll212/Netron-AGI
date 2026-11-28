"""Shift artifacts, safe expression validation, and tick result records."""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from early_agi.core.entropy_sensor import EntanglementReport
from early_agi.core.governor import Directive

__all__ = [
    "SHIFT_KINDS",
    "KernelDirective",
    "ShiftArtifact",
    "TickResult",
    "validate_shift_artifact",
]


SHIFT_KINDS = ("decomposition", "invariant", "new_operator")


KernelDirective = Directive


@dataclass(frozen=True)
class ShiftArtifact:
    """A reframing product from the deep tier. Not a patch, by construction."""

    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    tokens: int = 0
    latency_ms: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ShiftArtifact:
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            payload = {
                key: value
                for key, value in data.items()
                if key not in ("kind", "rationale", "tokens", "latency_ms")
            }
        return cls(
            kind=str(data.get("kind", "")).strip(),
            payload=dict(payload),
            rationale=str(data.get("rationale", ""))[:2000],
            tokens=int(data.get("tokens", 0) or 0),
            latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
        )

    def validate(
        self,
        *,
        state: Mapping[str, Any] | None = None,
        entanglement: EntanglementReport | None = None,
    ) -> tuple[bool, str]:
        return validate_shift_artifact(self.kind, self.payload, state=state, entanglement=entanglement)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": dict(self.payload),
            "rationale": self.rationale,
            "tokens": self.tokens,
            "latency_ms": round(self.latency_ms, 3),
        }


def _safe_member(obj: Any, attr: str) -> Any:
    if attr.startswith("__"):
        raise ValueError("dunder access is not allowed in an invariant")
    if isinstance(obj, Mapping) and attr in obj:
        return obj[attr]
    if isinstance(obj, object) and hasattr(obj, attr):
        return getattr(obj, attr)
    raise ValueError(f"unknown member {attr!r}")


def _safe_eval(node: ast.AST, names: Mapping[str, Any]) -> Any:
    """Evaluate a deliberately tiny expression grammar without Python eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id, None)
    if isinstance(node, ast.Attribute):
        return _safe_member(_safe_eval(node.value, names), node.attr)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_safe_eval(item, names) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _safe_eval(key, names): _safe_eval(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
        operand = _safe_eval(node.operand, names)
        if operand is None:
            return None
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        return -operand if isinstance(node.op, ast.USub) else +operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
        left, right = _safe_eval(node.left, names), _safe_eval(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left % right
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_safe_eval(value, names) for value in node.values]
        if isinstance(node.op, ast.And):
            return False if False in values else None if None in values else all(values)
        return True if True in values else None if None in values else any(values)
    if isinstance(node, ast.Call):
        target = node.func
        if not isinstance(target, ast.Name) or target.id not in _SAFE_CALLABLES:
            raise ValueError("only pure helper calls are allowed")
        args = [_safe_eval(arg, names) for arg in node.args]
        if node.keywords:
            raise ValueError("keyword calls are not allowed")
        if any(arg is None for arg in args):
            return None
        return _SAFE_CALLABLES[target.id](*args)
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, names)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _safe_eval(comparator, names)
            if left is None or right is None:
                return None
            if isinstance(op, ast.Eq):
                value = left == right
            elif isinstance(op, ast.NotEq):
                value = left != right
            elif isinstance(op, ast.Lt):
                value = left < right
            elif isinstance(op, ast.LtE):
                value = left <= right
            elif isinstance(op, ast.Gt):
                value = left > right
            elif isinstance(op, ast.GtE):
                value = left >= right
            elif isinstance(op, ast.In):
                value = left in right
            elif isinstance(op, ast.NotIn):
                value = left not in right
            else:  # pragma: no cover - future AST comparison operators
                raise ValueError(f"unsupported comparison {type(op).__name__}")
            if not value:
                return False
            left = right
        return True
    raise ValueError(f"unsafe expression node {type(node).__name__}")


_SAFE_CALLABLES: dict[str, Any] = {
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "all": all,
    "any": any,
    "sorted": sorted,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "mul": lambda left, right: left * right,
}


def _validate_invariant(check: str, state: Mapping[str, Any] | None) -> tuple[bool, str]:
    try:
        tree = ast.parse(check, mode="eval")
    except SyntaxError as exc:
        return False, f"invariant does not compile: {exc.msg}"
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.Subscript)) for node in ast.walk(tree)
    ):
        return False, "invariant uses unsafe syntax"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not isinstance(target, ast.Name) or target.id not in _SAFE_CALLABLES:
            return False, "invariant may only call pure builtin helpers"
        if node.keywords:
            return False, "invariant helper calls cannot use keyword arguments"
    names: dict[str, Any] = {
        "True": True,
        "False": False,
        "None": None,
        **dict(state or {}),
        "state": dict(state or {}),
    }
    try:
        verdict = _safe_eval(tree, names)
    except (ValueError, TypeError, AttributeError, ZeroDivisionError) as exc:
        return False, f"invariant is not safely evaluable: {exc}"
    if verdict is False:
        return False, "invariant is false for this candidate state"
    if verdict is None:
        return True, "invariant references unavailable values; structural check only"
    return bool(verdict), ""


def _validate_new_operator(name: str, source: str, state: Mapping[str, Any] | None) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"new_operator source does not parse: {exc.msg}"
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if name not in defined | assigned:
        return False, f"source does not define {name!r}"

    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.Await, ast.Yield)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return False, f"new_operator source contains forbidden {type(node).__name__}"
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id in {"eval", "exec", "compile", "open"}
        ):
            return False, f"new_operator source references unsafe name {node.id!r}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "new_operator source cannot access interpreter internals"
        if isinstance(node, ast.Call):
            target = node.func
            if not isinstance(target, ast.Name) or target.id not in {
                *_SAFE_CALLABLES,
                "print",
            }:
                return False, "new_operator source may only call pure helpers"
    if state is None:
        return True, ""

    namespace: dict[str, Any] = {"__builtins__": {**_SAFE_CALLABLES, "range": range}}
    try:
        exec(compile(tree, "<new_operator>", "exec"), namespace, namespace)
    except Exception as exc:
        return False, f"new_operator source failed in its restricted namespace: {type(exc).__name__}: {exc}"
    fn = namespace.get(name)
    if not callable(fn):
        return False, f"{name!r} is not callable after validation"
    payload = copy.deepcopy(dict(state))
    before = copy.deepcopy(payload)
    try:
        fn(payload)
    except TypeError:
        try:
            fn()
        except Exception as exc:
            return False, f"new_operator is not executable: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"new_operator is not executable: {type(exc).__name__}: {exc}"
    if payload != before:
        return False, "new_operator mutated its candidate state"
    return True, ""


def validate_shift_artifact(
    kind: str,
    payload: Mapping[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
    entanglement: EntanglementReport | None = None,
) -> tuple[bool, str]:
    """Validate structure and, when a state is available, semantic effect."""
    if kind not in SHIFT_KINDS:
        return False, f"kind must be one of {', '.join(SHIFT_KINDS)}; got {kind!r}"

    if kind == "decomposition":
        steps = payload.get("steps")
        if not isinstance(steps, (list, tuple)) or len(steps) < 2:
            return False, "decomposition needs at least two 'steps'"
        if not all(isinstance(step, str) and step.strip() for step in steps):
            return False, "every decomposition step must be a non-empty string"
        if len({step.strip().casefold() for step in steps}) != len(steps):
            return False, "decomposition steps must be distinct"
        if entanglement is not None and entanglement.is_volume_law and len(steps) < 2:
            return False, "volume-law coupling needs a genuine split"
        return True, ""

    if kind == "invariant":
        check = payload.get("check") or payload.get("assertion")
        if not isinstance(check, str) or not check.strip():
            return False, "invariant needs a non-empty 'check' expression"
        return _validate_invariant(check, state)

    name = payload.get("name")
    source = payload.get("source")
    if not isinstance(name, str) or not name.isidentifier():
        return False, "new_operator needs an identifier 'name'"
    if not isinstance(source, str) or not source.strip():
        return False, "new_operator needs 'source'"
    return _validate_new_operator(name, source, state)


@dataclass(frozen=True)
class TickResult:
    hash: str
    directive: str
    governor_directive: str
    eta_cog: float
    verified_drop_bits: float
    spectral_entropy_bits: float
    truncation_error: float
    effective_rank: int
    volume_law: bool
    gates: Mapping[str, str]
    writes_locked: bool
    shift_held: bool
    shift_kind: str = ""
    shift_error: str = ""
    synthesis_due: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash[:10],
            "directive": self.directive,
            "governor_directive": self.governor_directive,
            "eta_cog": round(self.eta_cog, 3),
            "verified_drop_bits": round(self.verified_drop_bits, 3),
            "spectral_entropy_bits": round(self.spectral_entropy_bits, 3),
            "truncation_error": round(self.truncation_error, 3),
            "effective_rank": self.effective_rank,
            "volume_law": self.volume_law,
            "gates": dict(self.gates),
            "writes_locked": self.writes_locked,
            "shift_held": self.shift_held,
            "shift_kind": self.shift_kind,
            "shift_error": self.shift_error,
            "synthesis_due": self.synthesis_due,
        }
