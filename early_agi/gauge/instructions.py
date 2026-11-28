"""Opcode handlers for the Gauge VM.

``vm.py`` is only a dispatcher. This compatibility façade owns the composed
``GaugeOpcodeRunner`` identity and re-exports the historical module namespace;
setup, governance, plasticity, and expression handlers live in
``gauge.instruction_parts``. ``state.py`` owns mutable views, ``builtins.py``
owns expression vocabulary, and ``traps.py`` owns trap recovery.
"""

from __future__ import annotations

from .instruction_parts import expressions as _expressions
from .instruction_parts import governance as _governance
from .instruction_parts import helpers as _helpers
from .instruction_parts import plasticity as _plasticity
from .instruction_parts import setup as _setup
from .instruction_parts.expressions import _ExpressionOps
from .instruction_parts.governance import _GovernanceOps
from .instruction_parts.plasticity import _PlasticityOps
from .instruction_parts.race import _RaceOps
from .instruction_parts.setup import _SetupOps

__all__ = ["GaugeOpcodeRunner"]

# Compatibility namespace from the former opcode monolith.
pyast = _setup.pyast
Mapping = _setup.Mapping
Any = _setup.Any
np = _setup.np
distilled_constraint = _helpers.distilled_constraint
Constraint = _setup.Constraint
RepoScanner = _setup.RepoScanner
EntanglementReport = _governance.EntanglementReport
matrix_fingerprint = _governance.matrix_fingerprint
Directive = _governance.Directive
GateResult = _governance.GateResult
OperatorSpec = _setup.OperatorSpec
g = _setup.g
GaugeGate = _setup.GaugeGate
call_name = _governance.call_name
GaugeRuntimeError = _plasticity.GaugeRuntimeError
evaluate = _expressions.evaluate
Instruction = _setup.Instruction
ManifoldState = _setup.ManifoldState
ReversibleTransform = _setup.ReversibleTransform
_READ_ONLY_FOOTPRINTS = _helpers._READ_ONLY_FOOTPRINTS
_mutating_footprint = _helpers._mutating_footprint
_append_distilled_constraints = _helpers._append_distilled_constraints
_constraint_view = _helpers._constraint_view


class GaugeOpcodeRunner(_SetupOps, _GovernanceOps, _PlasticityOps, _RaceOps, _ExpressionOps):
    """Execute one bytecode instruction against the VM's focused modules."""
