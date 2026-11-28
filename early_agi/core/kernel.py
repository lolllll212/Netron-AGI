"""The early-AGI kernel: one tick, adaptive directives, no escape hatches.

A tick is deterministic and reproducible: export pruned branches, measure
entanglement, run the dark-state gates, ask the governor for a directive,
archive the state. Nothing in here calls a model; the router feeds it. Plasticity
is opt-in and can request a read-only ``EPISTEMIC_PROBE`` before the first
mutation. This module is the public compatibility façade over
``core.kernel_parts``; all public names and class identities are preserved.
"""

from __future__ import annotations

from .governor import Directive
from .kernel_parts import artifacts as _artifacts
from .kernel_parts import tick as _tick
from .kernel_parts.artifacts import (
    SHIFT_KINDS,
    ShiftArtifact,
    TickResult,
    _safe_eval,
    _safe_member,
    _validate_invariant,
    _validate_new_operator,
    validate_shift_artifact,
)
from .kernel_parts.tick import _KernelBase

KernelDirective = Directive

__all__ = [
    "SHIFT_KINDS",
    "Directive",
    "EarlyAGIKernel",
    "KernelDirective",
    "ShiftArtifact",
    "TickResult",
    "validate_shift_artifact",
]

for _public_item in (
    ShiftArtifact,
    TickResult,
    validate_shift_artifact,
    _safe_member,
    _safe_eval,
    _validate_invariant,
    _validate_new_operator,
):
    _public_item.__module__ = __name__

# Compatibility namespace from the former kernel monolith.
_SAFE_CALLABLES = _artifacts._SAFE_CALLABLES
ast = _artifacts.ast
copy = _artifacts.copy
Mapping = _tick.Mapping
Sequence = _tick.Sequence
Any = _tick.Any
np = _tick.np
RelaxationBackend = _tick.RelaxationBackend
ClassicalSVDBackend = _tick.ClassicalSVDBackend
DarkStateGate = _tick.DarkStateGate
GateResult = _tick.GateResult
OperatorRegistry = _tick.OperatorRegistry
ColdSink = _tick.ColdSink
EntanglementReport = _tick.EntanglementReport
matrix_fingerprint = _tick.matrix_fingerprint
CognitiveStepCost = _tick.CognitiveStepCost
ThermodynamicGovernor = _tick.ThermodynamicGovernor


class EarlyAGIKernel(_KernelBase):
    """Deterministic, backwards-compatible composition of the kernel parts."""
