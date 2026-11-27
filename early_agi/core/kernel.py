"""The early-AGI kernel: one tick, adaptive directives, no escape hatches.

A tick is deterministic and reproducible: export pruned branches, measure
entanglement, run the dark-state gates, ask the governor for a directive,
archive the state. Nothing in here calls a model; the router feeds it. Plasticity
is opt-in and can request a read-only ``EPISTEMIC_PROBE`` before the first
mutation.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backends.base import RelaxationBackend
from ..backends.classical_svd import ClassicalSVDBackend
from ..operators.dark_state import DarkStateGate, GateResult
from ..operators.registry import OperatorRegistry
from .cold_sink import ColdSink
from .entropy_sensor import EntanglementReport, matrix_fingerprint
from .governor import CognitiveStepCost, Directive, ThermodynamicGovernor

__all__ = [
    "SHIFT_KINDS",
    "Directive",
    "EarlyAGIKernel",
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


class EarlyAGIKernel:
    def __init__(
        self,
        sink: ColdSink,
        backend: RelaxationBackend | None = None,
        governor: ThermodynamicGovernor | None = None,
        gates: Sequence[DarkStateGate] = (),
        registry: OperatorRegistry | None = None,
        *,
        chi_max: int = 4,
        tau: float = 0.15,
        scale: int = 0,
        lock_writes_on_shift: bool = True,
        plasticity: bool = False,
    ) -> None:
        self.sink = sink
        self.backend = backend or ClassicalSVDBackend()
        self.governor = governor or ThermodynamicGovernor()
        self.gates = list(gates)
        self.registry = registry
        self.chi_max = chi_max
        self.tau = tau
        self.scale = scale
        self.lock_writes_on_shift = lock_writes_on_shift
        self.plasticity = bool(plasticity)
        if self.plasticity:
            self.governor.enable_plasticity(base_tau=self.tau, base_chi_max=self.chi_max)

        self.last_hash: str | None = None
        self.satisfied_gates: set[str] = set()
        self.history: list[TickResult] = []
        self.coordinate_shifts = 0
        self.pruned_branches = 0
        self._writes_locked = False
        self._lock_reason = ""
        self._last_eta: float | None = None

    @property
    def writes_locked(self) -> bool:
        return self._writes_locked

    @property
    def lock_reason(self) -> str:
        return self._lock_reason

    @property
    def ticks(self) -> int:
        return len(self.history)

    def _engage_lock(self, reason: str) -> None:
        self._writes_locked = True
        self._lock_reason = reason
        if self.registry is not None:
            self.registry.isolate_writes(reason=reason)
            if reason.startswith("EPISTEMIC_PROBE"):
                self.registry.set_directive(Directive.EPISTEMIC_PROBE.value)

    def _release_lock(self) -> None:
        self._writes_locked = False
        self._lock_reason = ""
        if self.registry is not None:
            self.registry.release_writes()
            self.registry.clear_directive()

    def engage_probe_lock(self) -> None:
        """Enter the read-only probe directive before any mutating operation."""
        self._engage_lock("EPISTEMIC_PROBE (read-only measurements only)")

    def release_probe_lock(self) -> None:
        """Unlock writes after a caller has completed read-only epistemic probes."""
        if self._lock_reason.startswith("EPISTEMIC_PROBE"):
            self._release_lock()

    def evaluate_gates(self, candidate_state: Mapping[str, Any]) -> tuple[dict[str, GateResult], float]:
        """Run gates once without mutating kernel state; the tick commits them."""
        return self._gate_delta(candidate_state, commit=False)

    def apply_gate_results(self, results: Mapping[str, GateResult]) -> None:
        """Commit a gate transition after the workspace transaction survives."""
        for result in results.values():
            if result.passed is True:
                self.satisfied_gates.add(result.name)
            elif result.passed is False:
                self.satisfied_gates.discard(result.name)

    def reconcile_gates(self, candidate_state: Mapping[str, Any]) -> dict[str, GateResult]:
        """Re-align gate bookkeeping with a restored workspace without paying bits."""
        results: dict[str, GateResult] = {}
        for gate in self.gates:
            result = gate.evaluate(candidate_state)
            results[gate.name] = result
            if result.passed is True:
                self.satisfied_gates.add(gate.name)
            elif result.passed is False:
                self.satisfied_gates.discard(gate.name)
        return results

    def checkpoint(self) -> dict[str, Any]:
        """Capture mutable kernel/governor state for a failed-turn rollback."""
        return {
            "last_hash": self.last_hash,
            "satisfied_gates": set(self.satisfied_gates),
            "history": list(self.history),
            "coordinate_shifts": self.coordinate_shifts,
            "pruned_branches": self.pruned_branches,
            "writes_locked": self._writes_locked,
            "lock_reason": self._lock_reason,
            "last_eta": self._last_eta,
            "governor": {
                "history": list(self.governor.history),
                "drops": list(self.governor.drops),
                "directives": list(self.governor.directives),
                "shift_streak": self.governor.shift_streak,
                "recall_streak": self.governor.recall_streak,
            },
            "plasticity": self.governor.plasticity_snapshot(),
            "chi_max": self.chi_max,
            "tau": self.tau,
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.last_hash = checkpoint.get("last_hash")
        self.satisfied_gates = set(checkpoint.get("satisfied_gates", ()))
        self.history = list(checkpoint.get("history", ()))
        self.coordinate_shifts = int(checkpoint.get("coordinate_shifts", 0))
        self.pruned_branches = int(checkpoint.get("pruned_branches", 0))
        self._writes_locked = bool(checkpoint.get("writes_locked", False))
        self._lock_reason = str(checkpoint.get("lock_reason", ""))
        self._last_eta = checkpoint.get("last_eta")
        gov = checkpoint.get("governor", {})
        self.governor.history = list(gov.get("history", ()))
        self.governor.drops = list(gov.get("drops", ()))
        self.governor.directives = list(gov.get("directives", ()))
        self.governor.shift_streak = int(gov.get("shift_streak", 0))
        self.governor.recall_streak = int(gov.get("recall_streak", 0))
        plasticity = checkpoint.get("plasticity")
        if isinstance(plasticity, Mapping):
            self.governor.restore_plasticity(plasticity)
        self.chi_max = int(checkpoint.get("chi_max", self.chi_max))
        self.tau = float(checkpoint.get("tau", self.tau))
        if self.registry is not None:
            if self._writes_locked:
                self.registry.isolate_writes(reason=self._lock_reason)
                if self._lock_reason.startswith("EPISTEMIC_PROBE"):
                    self.registry.set_directive(Directive.EPISTEMIC_PROBE.value)
            else:
                self.registry.release_writes()
                self.registry.clear_directive()

    def tick(
        self,
        candidate_state: Mapping[str, Any],
        coupling_matrix: Any,
        cost: CognitiveStepCost | None = None,
        pruned_tails: Sequence[Mapping[str, Any]] | None = None,
        shift_artifact: ShiftArtifact | Mapping[str, Any] | None = None,
        label: str = "",
        *,
        gate_results: Mapping[str, GateResult] | None = None,
        verified_drop: float | None = None,
        entanglement: EntanglementReport | None = None,
        eta_cog: float | None = None,
        governor_directive: Directive | None = None,
        commit_gate_state: bool = True,
        predicted_drop: float | None = None,
        analogy_similarity: float = 0.0,
        risk_level: float = 0.5,
        probe_completed: bool = False,
    ) -> TickResult:
        cost = cost or CognitiveStepCost()
        matrix = np.asarray(coupling_matrix, dtype=np.float64)
        if (eta_cog is None) != (governor_directive is None):
            raise ValueError("eta_cog and governor_directive must be supplied together")

        for tail in pruned_tails or ():
            self.sink.put(
                "pruned_branch",
                dict(tail),
                scale=self.scale,
                parent_hash=self.last_hash,
                eta_cog=self._last_eta,
            )
            self.pruned_branches += 1

        entanglement_measured = entanglement is None
        if entanglement is None:
            entanglement = self.backend.measure_entanglement(matrix, self.chi_max, self.tau)
        if gate_results is None or verified_drop is None:
            gate_results, verified_drop = self._gate_delta(candidate_state, commit=commit_gate_state)
        else:
            gate_results = dict(gate_results)
            if commit_gate_state:
                self.apply_gate_results(gate_results)

        thresholds = self.governor.modulate(
            analogy_similarity,
            risk_level,
            predicted_drop=predicted_drop,
            verified_drop=verified_drop,
        )
        if probe_completed:
            self.governor.clear_probe()
            thresholds = self.governor.last_thresholds or thresholds
            # A preflight probe already ran; the kernel must not re-lock writes
            # on the same directive when Gauge advances to EVOLVE/CRYSTALLIZE.
            if governor_directive is Directive.EPISTEMIC_PROBE:
                governor_directive = Directive.CRYSTALLIZE
        if self.governor.plasticity_enabled:
            self.tau = thresholds.tau
            self.chi_max = thresholds.chi_max
            if entanglement_measured:
                entanglement = self.backend.measure_entanglement(matrix, self.chi_max, self.tau)
        if eta_cog is None or governor_directive is None:
            eta_cog, governor_directive = self.governor.evaluate(verified_drop, cost, entanglement)

        artifact = self._coerce_artifact(shift_artifact)
        artifact_ok, artifact_error = (
            artifact.validate(state=candidate_state, entanglement=entanglement)
            if artifact is not None
            else (False, "")
        )
        directive = governor_directive
        shift_held = False

        if self._writes_locked:
            if artifact is not None and artifact_ok:
                self._release_lock()
            else:
                directive = Directive.SHIFT_COORDINATE
                shift_held = True
        elif governor_directive is Directive.SHIFT_COORDINATE and self.lock_writes_on_shift:
            self._engage_lock(f"SHIFT_COORDINATE (eta={eta_cog:.2f})")
        elif governor_directive is Directive.EPISTEMIC_PROBE and self.lock_writes_on_shift:
            self._engage_lock("EPISTEMIC_PROBE (read-only measurements only)")

        payload = {
            "label": label,
            "state": dict(candidate_state),
            "directive": directive.value,
            "governor_directive": governor_directive.value,
            "eta_cog": round(eta_cog, 6),
            "verified_drop_bits": round(verified_drop, 6),
            "entanglement": entanglement.as_dict(),
            "coupling_fingerprint": matrix_fingerprint(matrix),
            "coupling_shape": list(matrix.shape),
            "gates": {name: result.verdict for name, result in gate_results.items()},
            "cost": cost.as_dict(),
            "shift_held": shift_held,
            "writes_locked": self._writes_locked,
            "scale": self.scale,
            "plasticity": {
                "enabled": self.governor.plasticity_enabled,
                "thresholds": thresholds.as_dict(),
                "analogy_similarity": round(float(analogy_similarity), 4),
                "risk_level": round(float(risk_level), 4),
                "surprise_ema": round(self.governor.regulator.surprise_ema, 6),
            },
        }
        self.last_hash = self.sink.put(
            "uv_state", payload, scale=self.scale, parent_hash=self.last_hash, eta_cog=eta_cog
        )

        shift_kind = ""
        if artifact is not None:
            shift_kind = artifact.kind
            record = {
                "artifact": artifact.as_dict(),
                "accepted": artifact_ok,
                "error": artifact_error,
                "eta_cog": round(eta_cog, 6),
                "parent_uv": self.last_hash,
            }
            if artifact_ok:
                self.sink.put("shift_artifact", record, scale=self.scale + 1, parent_hash=self.last_hash)
                self.coordinate_shifts += 1
            else:
                self.sink.put("shift_rejected", record, scale=self.scale, parent_hash=self.last_hash)

        result = TickResult(
            hash=self.last_hash,
            directive=directive.value,
            governor_directive=governor_directive.value,
            eta_cog=eta_cog,
            verified_drop_bits=verified_drop,
            spectral_entropy_bits=entanglement.spectral_entropy_bits,
            truncation_error=entanglement.truncation_error,
            effective_rank=entanglement.effective_rank,
            volume_law=entanglement.is_volume_law,
            gates={name: verdict.verdict for name, verdict in gate_results.items()},
            writes_locked=self._writes_locked,
            shift_held=shift_held,
            shift_kind=shift_kind,
            shift_error=artifact_error,
            synthesis_due=self.governor.synthesis_due,
        )
        self.history.append(result)
        self._last_eta = eta_cog
        return result

    @staticmethod
    def _coerce_artifact(
        raw: ShiftArtifact | Mapping[str, Any] | None,
    ) -> ShiftArtifact | None:
        if raw is None:
            return None
        if isinstance(raw, ShiftArtifact):
            return raw
        if isinstance(raw, Mapping):
            return ShiftArtifact.from_mapping(raw)
        raise TypeError(f"shift_artifact must be a ShiftArtifact or mapping, got {type(raw).__name__}")

    def _gate_delta(
        self, candidate_state: Mapping[str, Any], *, commit: bool = True
    ) -> tuple[dict[str, GateResult], float]:
        results: dict[str, GateResult] = {}
        newly_verified = 0.0
        broken = 0.0
        for gate in self.gates:
            result = gate.evaluate(candidate_state)
            results[gate.name] = result
            if result.passed is True and gate.name not in self.satisfied_gates:
                newly_verified += gate.bits
                if commit:
                    self.satisfied_gates.add(gate.name)
            elif result.passed is False and gate.name in self.satisfied_gates:
                broken += gate.bits
                if commit:
                    self.satisfied_gates.discard(gate.name)
        return results, newly_verified - broken

    def snapshot(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "scale": self.scale,
            "last_hash": self.last_hash[:10] if self.last_hash else None,
            "writes_locked": self._writes_locked,
            "lock_reason": self._lock_reason,
            "satisfied_gates": sorted(self.satisfied_gates),
            "coordinate_shifts": self.coordinate_shifts,
            "pruned_branches": self.pruned_branches,
            "governor": self.governor.snapshot(),
            "plasticity": {
                "enabled": self.governor.plasticity_enabled,
                "chi_max": self.chi_max,
                "tau": self.tau,
                "thresholds": (
                    self.governor.last_thresholds.as_dict() if self.governor.last_thresholds else None
                ),
            },
            "sink": self.sink.stats(),
        }
