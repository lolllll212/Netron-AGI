"""Gauge VM: parse/check/compile orchestration and bytecode dispatch.

This module is deliberately thin. It owns the instruction loop and the public
API; opcode semantics live in ``instructions.py``, workspace/manifold views in
``state.py``, expression built-ins in ``builtins.py``, and trap evaluation plus
inverse-stack recovery in ``traps.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..backends import get_backend
from ..config import Settings
from ..core.cold_sink import ColdSink
from ..core.governor import Directive
from ..core.kernel import EarlyAGIKernel
from ..memory.mera_graph import MeraGraph
from ..memory.reflex import ReflexLibrary
from ..operators.registry import OperatorRegistry
from . import ast as g
from .builtins import GaugeGate
from .checker import CheckedProgram, validate_program
from .compiler import compile_program
from .errors import GaugeRuntimeError
from .instructions import GaugeOpcodeRunner
from .ir import Instruction, ProgramIR
from .results import GaugeRunResult
from .state import GaugeExecutionState
from .synthesizer import Synthesizer
from .traps import TrapRunner

__all__ = [
    "GaugeGate",
    "GaugeRunResult",
    "GaugeRuntimeError",
    "GaugeVM",
    "run_gauge",
    "run_program",
]

_EXECUTION_FIELDS = frozenset(
    {
        "applied_op_chain",
        "applied_ops",
        "collapses",
        "cost",
        "coupling_fingerprint",
        "coupling_matrix",
        "directive",
        "distilled_constraints",
        "drag",
        "entanglement",
        "events",
        "gate_defs",
        "gate_results",
        "gate_state_committed",
        "govern_config",
        "halted",
        "inverse_stack",
        "kernel_checkpoint",
        "manifolds",
        "op_after",
        "op_commutes",
        "ops",
        "program_name",
        "recalled",
        "run_index",
        "scales",
        "state",
        "status",
        "synthesis_events",
        "tick",
        "trace",
        "transforms",
        "traps",
        "variables",
        "verified_drop",
        "violations",
        "workspace",
        "eta",
        "analogy",
        "analogy_similarity",
        "predicted_drop",
        "probe_results",
        "novelty_score",
        "plasticity",
    }
)

_RESULT_FIELDS = frozenset(
    {
        "gates",
        "governor",
        "mera",
        "reflexes",
        "synthesis",
        "plasticity",
    }
)

_RESULT_LIST_BINDINGS = {
    "collapses": "collapses",
    "recalled": "recalled",
    "synthesis_events": "synthesis",
    "trace": "trace",
    "traps": "traps",
    "violations": "violations",
}

_DELEGATED_METHODS = frozenset(
    {
        "apply_transform",
        "evaluation_context",
        "execute_transform_statement",
        "plain_state",
        "restore_inverse_stack",
        "scale_number",
        "scan_workspace",
        "transform_context",
        "view",
    }
)

_PROBE_NOVELTY_THRESHOLD = 0.65


def _walk_instructions(instruction: Instruction) -> list[Instruction]:
    """Flatten nested trap bodies so preflight can find a PROBE plan."""
    found = [instruction]
    for child in instruction.body:
        found.extend(_walk_instructions(child))
    return found


class GaugeVM:
    """Dispatch one checked Gauge program against a repository workspace."""

    def __init__(
        self,
        root: str | Path,
        *,
        settings: Settings | None = None,
        sink: ColdSink | None = None,
        kernel: EarlyAGIKernel | None = None,
        registry: OperatorRegistry | None = None,
        backend: Any | None = None,
        router: Any | None = None,
        synthesizer: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.settings = settings or Settings.from_env()
        validate = getattr(self.settings, "validate", None)
        if callable(validate):
            validate()
        self.sink = sink or (kernel.sink if kernel is not None else ColdSink(self.settings.sink_path))
        self.backend = backend or (
            kernel.backend if kernel is not None else get_backend(self.settings.backend)
        )
        if backend is not None:
            self.requested_backend = str(getattr(backend, "name", type(backend).__name__))
        elif kernel is not None:
            self.requested_backend = str(getattr(self.backend, "name", type(self.backend).__name__))
        else:
            self.requested_backend = self.settings.backend
        self.registry = registry or (kernel.registry if kernel is not None else OperatorRegistry())
        self.kernel = kernel or EarlyAGIKernel(
            self.sink,
            backend=self.backend,
            registry=self.registry,
            chi_max=self.settings.chi_max,
            tau=self.settings.tau,
        )
        self.kernel.registry = self.registry
        self.graph = MeraGraph(self.sink, self.settings.mera_path, backend=self.backend)
        self.reflex_library = ReflexLibrary(self.graph)
        self.base_gates = list(self.kernel.gates)
        self.synthesizer = self._configure_synthesizer(router, synthesizer)
        self.execution = GaugeExecutionState(self)
        self.execution.run_index = 0  # the first explicit run becomes index 1
        self.traps_runner = TrapRunner(self)
        self.opcodes = GaugeOpcodeRunner(self)
        self.checked: CheckedProgram | None = None
        self.ir: ProgramIR | None = None
        self.program: g.Program | None = None
        self.result = GaugeRunResult("running", "<gauge>", str(self.root))
        self._bind_result_lists()

    def _configure_synthesizer(self, router: Any | None, synthesizer: Any | None) -> Any:
        bindings = {
            "sink": self.sink,
            "registry": self.registry,
            "gates": self.base_gates,
            "reflex_library": self.reflex_library,
            "eta_min": self.kernel.governor.eta_min,
        }
        if synthesizer is None:
            return Synthesizer(
                router,
                sink=self.sink,
                registry=self.registry,
                gates=self.base_gates,
                reflex_library=self.reflex_library,
                eta_min=self.kernel.governor.eta_min,
            )
        if hasattr(synthesizer, "bind"):
            synthesizer.bind(**bindings)
            return synthesizer
        for name, value in bindings.items():
            if getattr(synthesizer, name, None) is None:
                try:
                    setattr(synthesizer, name, value)
                except (AttributeError, TypeError):
                    continue
        return synthesizer

    # -- compatibility delegation ------------------------------------------
    def __getattr__(self, name: str) -> Any:
        namespace = self.__dict__
        execution = namespace.get("execution")
        result = namespace.get("result")
        if execution is not None and name in _EXECUTION_FIELDS:
            return getattr(execution, name)
        if result is not None and name in _RESULT_FIELDS:
            return getattr(result, name)
        if execution is not None and name in _DELEGATED_METHODS:
            return getattr(execution, name)
        if name == "_plain_state" and execution is not None:
            return execution._plain_state
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        namespace = self.__dict__
        execution = namespace.get("execution")
        result = namespace.get("result")
        if execution is not None and name in _EXECUTION_FIELDS:
            setattr(execution, name, value)
            if result is not None and name in _RESULT_LIST_BINDINGS:
                setattr(result, _RESULT_LIST_BINDINGS[name], value)
            return
        if result is not None and name in _RESULT_FIELDS:
            setattr(result, name, value)
            return
        object.__setattr__(self, name, value)

    def _bind_result_lists(self) -> None:
        execution = self.execution
        self.result.backend = str(self.requested_backend)
        self.result.backend_impl = str(getattr(self.backend, "name", type(self.backend).__name__))
        self.result.run_index = execution.run_index
        self.result.trace = execution.trace
        self.result.violations = execution.violations
        self.result.traps = execution.traps
        self.result.synthesis = execution.synthesis_events
        self.result.collapses = execution.collapses
        self.result.recalled = execution.recalled
        self.result.gates = {}
        self.result.entanglement = {}
        self.result.governor = {}
        self.result.manifolds = {}
        self.result.ir = {}
        self.result.tick = {}
        self.result.mera = {}
        self.result.reflexes = {}
        self.result.plasticity = dict(execution.plasticity)

    # -- public -------------------------------------------------------------
    def run(
        self,
        program: str | g.Program | CheckedProgram,
        *,
        name: str = "<gauge>",
    ) -> GaugeRunResult:
        """Parse/check/compile a program, dispatch its IR, and return a report."""
        checked = program if isinstance(program, CheckedProgram) else validate_program(program)
        ir = compile_program(checked)
        self._reset(name)
        self.checked = checked
        self.ir = ir
        self.program = checked.program
        self.execution.scan_workspace()
        self.execution.kernel_checkpoint = self.kernel.checkpoint()
        self.kernel.gates = list(self.base_gates)
        self.result.ir = ir.as_dict()
        self.result.run_index = self.execution.run_index
        if hasattr(self.synthesizer, "eta_min"):
            self.synthesizer.eta_min = self.kernel.governor.eta_min

        try:
            instructions = ir.instructions()
            setup_count = len(ir.setup)
            for index, instruction in enumerate(instructions):
                if self.execution.halted:
                    break
                self.execute(instruction)
                # Setup registered the extern operators and plastic thresholds;
                # emit EPISTEMIC_PROBE before the first body OP_APPLY can write.
                if index + 1 == setup_count:
                    self._preflight_probe(ir)
            if not self.execution.collapses:
                self.violate("program halted with an uncollapsed manifold leak into the sink")
            if not self.execution.gate_state_committed and self.execution.gate_results:
                self.kernel.apply_gate_results(self.execution.gate_results)
                self.execution.gate_state_committed = True
            self.execution.status = "halted"
            self.result.reflexes = self._compile_hot_chains()
            self.result.plasticity = dict(self.execution.plasticity)
        except Exception as exc:
            self.execution.status = "error"
            self.violate(f"{type(exc).__name__}: {exc}")
            try:
                self.traps_runner.recover()
            except Exception as recover_exc:
                self.violate(f"rollback failed: {type(recover_exc).__name__}: {recover_exc}")
        finally:
            self.result.status = self.execution.status
            self.result.gates = {name: result.verdict for name, result in self.execution.gate_results.items()}
            self.result.entanglement = (
                self.execution.entanglement.as_dict() if self.execution.entanglement is not None else {}
            )
            self.result.governor = {
                "eta_cog": round(self.execution.eta, 6),
                "directive": self.execution.directive.value,
                "eta_min": self.kernel.governor.eta_min,
                "chi_max": self.kernel.chi_max,
                "synthesis_due": self.kernel.governor.synthesis_due,
            }
            self.result.manifolds = {
                name: manifold.as_dict() for name, manifold in self.execution.manifolds.items()
            }
            self.result.tick = self.execution.tick.as_dict() if self.execution.tick is not None else {}
            self.result.trace = self.execution.trace
            self.result.violations = self.execution.violations
            self.result.traps = self.execution.traps
            self.result.synthesis = self.execution.synthesis_events
            self.result.collapses = self.execution.collapses
            self.result.recalled = self.execution.recalled
            self.result.plasticity = dict(self.execution.plasticity)
        return self.result

    def execute(self, instruction: Instruction) -> None:
        """Dispatch exactly one instruction to the focused opcode runner."""
        self.opcodes.execute(instruction)

    @staticmethod
    def _probe_plan(ir: ProgramIR) -> Instruction | None:
        for instruction in ir.instructions():
            for nested in _walk_instructions(instruction):
                if nested.op == "PROBE":
                    return nested
        return None

    def _preflight_probe(self, ir: ProgramIR) -> None:
        """Emit EPISTEMIC_PROBE ahead of CRYSTALLIZE and every mutating op."""
        if not self.kernel.plasticity or self.execution.probe_results:
            return
        thresholds = self.kernel.governor.last_thresholds
        if thresholds is None or thresholds.novelty_score <= _PROBE_NOVELTY_THRESHOLD:
            return
        plan = self._probe_plan(ir)
        if plan is None:
            return
        ops = [str(item) for item in plan.args.get("ops", ())]
        if not ops:
            return
        self.execution.directive = Directive.EPISTEMIC_PROBE
        self.kernel.engage_probe_lock()
        self.emit(
            "EPISTEMIC_PROBE",
            {
                "phase": "preflight",
                "novelty_score": thresholds.novelty_score,
                "operators": ops,
                "writes_locked": True,
            },
        )
        args = {**plan.args, "preflight": True}
        self.execute(Instruction("PROBE", args, (), plan.span))
        self.execution.plasticity["preflight_probe"] = {
            "directive": Directive.EPISTEMIC_PROBE.value,
            "operators": ops,
            "results": len(self.execution.probe_results),
            "novelty_score": thresholds.novelty_score,
        }
        self.result.plasticity = dict(self.execution.plasticity)

    # -- thin helpers -------------------------------------------------------
    def _reset(self, name: str) -> None:
        self.checked = None
        self.ir = None
        self.program = None
        self.execution.reset(name)
        self.result = GaugeRunResult("running", name, str(self.root))
        self._bind_result_lists()

    def _compile_hot_chains(self) -> dict[str, Any]:
        compile_chains = getattr(self.synthesizer, "compile_hot_chains", None)
        if not callable(compile_chains):
            return {
                "fused": 0,
                "operators": [],
                "signatures": 0,
                "skipped": ["no chain compiler"],
            }
        try:
            return compile_chains(min_support=2)
        except Exception as exc:
            return {
                "fused": 0,
                "operators": [],
                "signatures": 0,
                "skipped": [f"chain compiler failed: {type(exc).__name__}: {exc}"],
            }

    def _scan_workspace(self) -> Any:
        return self.execution.scan_workspace()

    def _functions(self) -> dict[str, Any]:
        return self.execution.builtins.functions()

    def _scale_number(self, value: Any) -> int:
        return self.execution.scale_number(value)

    def trap_due(self, event: str) -> bool:
        return self.traps_runner.is_due(event)

    def emit(self, op: str, detail: Any) -> None:
        payload = {key: value for key, value in dict(detail).items() if key != "op"}
        self.trace.append({"op": op, **payload})
        self.result.trace = self.trace

    def violate(self, message: str) -> None:
        self.violations.append(message)
        self.result.violations = self.violations


def run_program(
    program: str | g.Program | CheckedProgram,
    root: str | Path = ".",
    *,
    name: str = "<gauge>",
    **kwargs: Any,
) -> GaugeRunResult:
    """Convenience wrapper: compile and execute a Gauge program."""
    return GaugeVM(root, **kwargs).run(program, name=name)


def run_gauge(
    source: str,
    root: str | Path = ".",
    **kwargs: Any,
) -> GaugeRunResult:
    """Alias with source-text semantics for CLI and notebook use."""
    return run_program(source, root, **kwargs)


def _json_safe(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
