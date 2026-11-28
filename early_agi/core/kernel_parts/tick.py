"""Deterministic kernel tick loop, gate reconciliation, and snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from early_agi.backends.base import RelaxationBackend
from early_agi.backends.classical_svd import ClassicalSVDBackend
from early_agi.core.cold_sink import ColdSink
from early_agi.core.entropy_sensor import EntanglementReport, matrix_fingerprint
from early_agi.core.governor import CognitiveStepCost, Directive, ThermodynamicGovernor
from early_agi.operators.dark_state import DarkStateGate, GateResult
from early_agi.operators.registry import OperatorRegistry

from .artifacts import ShiftArtifact, TickResult

__all__ = ["_KernelBase"]


class _KernelBase:
    """Tick-loop methods inherited by the public EarlyAGIKernel."""

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
