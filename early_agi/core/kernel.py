"""The early-AGI kernel: one tick, three directives, no escape hatches.

A tick is deterministic and reproducible: export pruned branches, measure
entanglement, run the dark-state gates, ask the governor for a directive,
archive the state. Nothing in here calls a model; the router feeds it.

Two invariants are enforced here rather than asked for in a prompt:

1. Verified entropy only. A gate awards its bits the first time it passes and
   charges them back if a later edit breaks it. Unverifiable gates are neutral.
2. Coordinate shifts physically block edits. When SHIFT_COORDINATE fires, writes
   are isolated until a valid reframing artefact (decomposition / invariant /
   new operator) arrives. Sending another patch holds the lock instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..backends.base import RelaxationBackend
from ..backends.classical_svd import ClassicalSVDBackend
from ..operators.dark_state import DarkStateGate, GateResult
from ..operators.registry import OperatorRegistry
from .cold_sink import ColdSink
from .entropy_sensor import matrix_fingerprint
from .governor import CognitiveStepCost, Directive, ThermodynamicGovernor

__all__ = [
    "EarlyAGIKernel",
    "SHIFT_KINDS",
    "ShiftArtifact",
    "TickResult",
    "validate_shift_artifact",
]

SHIFT_KINDS = ("decomposition", "invariant", "new_operator")


# ---------------------------------------------------------------------------
# Coordinate-shift artefacts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShiftArtifact:
    """A reframing product from the deep tier. Not a patch, by construction."""

    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ShiftArtifact":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            payload = {key: value for key, value in data.items() if key not in ("kind", "rationale")}
        return cls(kind=str(data.get("kind", "")).strip(), payload=dict(payload), rationale=str(data.get("rationale", ""))[:2000])

    def validate(self) -> tuple[bool, str]:
        return validate_shift_artifact(self.kind, self.payload)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload), "rationale": self.rationale}


def validate_shift_artifact(kind: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Structural check; keeps model output inside the loop's contract."""
    if kind not in SHIFT_KINDS:
        return False, f"kind must be one of {', '.join(SHIFT_KINDS)}; got {kind!r}"

    if kind == "decomposition":
        steps = payload.get("steps")
        if not isinstance(steps, (list, tuple)) or len(steps) < 2:
            return False, "decomposition needs at least two 'steps'"
        if not all(isinstance(step, str) and step.strip() for step in steps):
            return False, "every decomposition step must be a non-empty string"
        return True, ""

    if kind == "invariant":
        check = payload.get("check") or payload.get("assertion")
        if not isinstance(check, str) or not check.strip():
            return False, "invariant needs a non-empty 'check' expression"
        try:
            compile(check, "<invariant>", "eval")
        except SyntaxError as exc:
            return False, f"invariant does not compile: {exc.msg}"
        return True, ""

    name = payload.get("name")
    source = payload.get("source")
    if not isinstance(name, str) or not name.isidentifier():
        return False, "new_operator needs an identifier 'name'"
    if not isinstance(source, str) or not source.strip():
        return False, "new_operator needs 'source'"
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
    return True, ""


# ---------------------------------------------------------------------------
# Tick result
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
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

        self.last_hash: str | None = None
        self.satisfied_gates: set[str] = set()
        self.history: list[TickResult] = []
        self.coordinate_shifts = 0
        self.pruned_branches = 0
        self._writes_locked = False
        self._lock_reason = ""
        self._last_eta: float | None = None

    # -- state --------------------------------------------------------------
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

    def _release_lock(self) -> None:
        self._writes_locked = False
        self._lock_reason = ""
        if self.registry is not None:
            self.registry.release_writes()

    # -- the tick -----------------------------------------------------------
    def tick(
        self,
        candidate_state: Mapping[str, Any],
        coupling_matrix: Any,
        cost: CognitiveStepCost | None = None,
        pruned_tails: Sequence[Mapping[str, Any]] | None = None,
        shift_artifact: ShiftArtifact | Mapping[str, Any] | None = None,
        label: str = "",
    ) -> TickResult:
        cost = cost or CognitiveStepCost()
        matrix = np.asarray(coupling_matrix, dtype=np.float64)

        # 1. Export pruned branches before anything can depend on them.
        for tail in pruned_tails or ():
            self.sink.put(
                "pruned_branch",
                dict(tail),
                scale=self.scale,
                parent_hash=self.last_hash,
                eta_cog=self._last_eta,
            )
            self.pruned_branches += 1

        # 2. Measure entanglement of the active problem graph.
        entanglement = self.backend.measure_entanglement(matrix, self.chi_max, self.tau)

        # 3. Dark-state gates -> verified (not claimed) entropy collapse.
        gate_results, verified_drop = self._gate_delta(candidate_state)

        # 4. Governor decides.
        eta_cog, governor_directive = self.governor.evaluate(verified_drop, cost, entanglement)

        # 5. Trap 2: the fast executor cannot patch its way out of a shift.
        artifact = self._coerce_artifact(shift_artifact)
        artifact_ok, artifact_error = artifact.validate() if artifact is not None else (False, "")
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

        # 6. Archive the UV state (and the reframing, at a coarser scale).
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

    # -- internals ----------------------------------------------------------
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

    def _gate_delta(self, candidate_state: Mapping[str, Any]) -> tuple[dict[str, GateResult], float]:
        results: dict[str, GateResult] = {}
        newly_verified = 0.0
        broken = 0.0
        for gate in self.gates:
            result = gate.evaluate(candidate_state)
            results[gate.name] = result
            if result.passed is True:
                if gate.name not in self.satisfied_gates:
                    newly_verified += gate.bits
                    self.satisfied_gates.add(gate.name)
            elif result.passed is False:
                if gate.name in self.satisfied_gates:
                    broken += gate.bits
                    self.satisfied_gates.discard(gate.name)
            # None: unverifiable, so neither awarded nor charged
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
            "sink": self.sink.stats(),
        }
