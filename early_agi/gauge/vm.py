"""Gauge Step 3/4: the hybrid thermodynamic VM.

The VM is the sovereign control layer. It lowers a checked ``.gauge`` program
into bytecode and executes that bytecode against the real early-AGI subsystems:

* SQLite :class:`~early_agi.core.cold_sink.ColdSink` for collapse and recall;
* :class:`~early_agi.backends.base.RelaxationBackend` for entanglement;
* :class:`~early_agi.core.governor.ThermodynamicGovernor` for ``eta_min`` and drag;
* :class:`~early_agi.operators.registry.OperatorRegistry` for extern ops;
* safe tri-state Gauge predicates for dark-state gates;
* :class:`~early_agi.memory.mera_graph.MeraGraph` for L-scale consolidation;
* a deterministic synthesizer that must show a positive efficiency gain.

No Python ``eval`` is used for Gauge expressions. The LLM, when supplied, is
called only from a ``synthesize op`` trap and never becomes the outer loop.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.classical_svd import ClassicalSVDBackend
from ..config import Settings
from ..core.cold_sink import ColdSink
from ..core.coupling import Constraint, RepoScanner
from ..core.entropy_sensor import EntanglementReport, matrix_fingerprint
from ..core.governor import CognitiveStepCost, Directive
from ..core.kernel import EarlyAGIKernel, ShiftArtifact, TickResult
from ..memory.mera_graph import MeraGraph
from ..operators.dark_state import GateResult, run_pytest
from ..operators.registry import OperatorRegistry, OperatorSpec
from . import ast as g
from .checker import IRREVERSIBLE_CALLS, CheckedProgram, call_name, validate_program
from .compiler import compile_program
from .expressions import EvaluationContext, ExpressionError, evaluate
from .ir import Instruction, ProgramIR
from .runtime import GaugeState, ManifoldState, ReversibleTransform
from .synthesizer import SynthesisResult, Synthesizer

__all__ = [
    "GaugeGate",
    "GaugeRunResult",
    "GaugeRuntimeError",
    "GaugeVM",
    "run_gauge",
    "run_program",
]

_SYMBOLIC_NAMES = frozenset(
    {
        "identity",
        "dependency_cut",
        "module_cut",
        "cut_graph",
        "coordinate_shift",
        "sink",
        "none",
        "all",
        "true",
        "false",
    }
)


class GaugeRuntimeError(RuntimeError):
    """A checked program failed a runtime invariant."""


class _Resolver:
    """Resolve symbolic enum-like identifiers without exposing Python names."""

    def lookup(self, name: str) -> Any:
        if name in _SYMBOLIC_NAMES:
            return None if name == "none" else name
        raise ExpressionError(f"unknown symbolic identifier {name!r}")

    def call(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        raise ExpressionError(f"unknown function {name!r}")


@dataclass
class GaugeGate:
    """Bridge a Gauge predicate into the kernel's tri-state gate protocol."""

    name: str
    predicate: Any
    vm: GaugeVM
    bits: float = 1.0

    def evaluate(self, state: Mapping[str, Any]) -> GateResult:
        try:
            value = evaluate(self.predicate, self.vm.evaluation_context(state))
        except Exception as exc:
            return GateResult(self.name, None, self.bits, f"{type(exc).__name__}: {exc}")
        if value is None:
            return GateResult(self.name, None, self.bits, "predicate could not be verified")
        return GateResult(self.name, bool(value), self.bits, "")


@dataclass
class GaugeRunResult:
    status: str
    program: str
    root: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, str] = field(default_factory=dict)
    entanglement: dict[str, Any] = field(default_factory=dict)
    governor: dict[str, Any] = field(default_factory=dict)
    manifolds: dict[str, Any] = field(default_factory=dict)
    collapses: dict[str, str] = field(default_factory=dict)
    synthesis: list[dict[str, Any]] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    recalled: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    ir: dict[str, Any] = field(default_factory=dict)
    tick: dict[str, Any] = field(default_factory=dict)
    mera: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "halted" and not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "program": self.program,
            "root": self.root,
            "trace": self.trace,
            "gates": self.gates,
            "entanglement": self.entanglement,
            "governor": self.governor,
            "manifolds": self.manifolds,
            "collapses": self.collapses,
            "synthesis": self.synthesis,
            "traps": self.traps,
            "recalled": self.recalled,
            "violations": self.violations,
            "ir": self.ir,
            "tick": self.tick,
            "mera": self.mera,
        }


class GaugeVM:
    """Execute one checked Gauge program against a repository workspace."""

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
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.settings = settings or Settings.from_env()
        self.sink = sink or (kernel.sink if kernel is not None else ColdSink(self.settings.sink_path))
        self.backend = backend or (kernel.backend if kernel is not None else ClassicalSVDBackend())
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
        self.synthesizer = synthesizer or Synthesizer(router)
        self.base_gates = list(self.kernel.gates)
        self._reset("<gauge>")

    # -- public -------------------------------------------------------------
    def run(
        self,
        program: str | g.Program | CheckedProgram,
        *,
        name: str = "<gauge>",
    ) -> GaugeRunResult:
        """Parse/check/compile a program, execute its IR, and return a report."""
        checked = program if isinstance(program, CheckedProgram) else validate_program(program)
        ir = compile_program(checked)
        self._reset(name)
        self.checked = checked
        self.ir = ir
        self.program = checked.program
        self._scan_workspace()
        self.kernel_checkpoint = self.kernel.checkpoint()
        self.kernel.gates = list(self.base_gates)
        self.result.ir = ir.as_dict()

        try:
            for instruction in ir.instructions():
                if self.halted:
                    break
                self.execute(instruction)
            if not self.collapses:
                self.violate("program halted with an uncollapsed manifold leak into the sink")
            if not self.gate_state_committed and self.gate_results:
                self.kernel.apply_gate_results(self.gate_results)
                self.gate_state_committed = True
            self.status = "halted"
        except Exception as exc:
            self.status = "error"
            self.violate(f"{type(exc).__name__}: {exc}")
            if not self.gate_state_committed and self.gate_results:
                self.kernel.reconcile_gates(self._plain_state())
        finally:
            self.result.status = self.status
            self.result.gates = {name: result.verdict for name, result in self.gate_results.items()}
            self.result.entanglement = self.entanglement.as_dict() if self.entanglement is not None else {}
            self.result.governor = {
                "eta_cog": round(self.eta, 6),
                "directive": self.directive.value,
                "eta_min": self.kernel.governor.eta_min,
                "chi_max": self.kernel.chi_max,
                "synthesis_due": self.kernel.governor.synthesis_due,
            }
            self.result.manifolds = {name: manifold.as_dict() for name, manifold in self.manifolds.items()}
            self.result.tick = self.tick.as_dict() if self.tick is not None else {}
        return self.result

    def execute(self, instruction: Instruction) -> None:
        """Execute one instruction, including a trap body when its event is due."""
        op = instruction.op
        if op == "HALT":
            self.halted = True
            self.emit("HALT", {})
            return
        if op.startswith("TRAP_"):
            event = str(instruction.args.get("event", op.removeprefix("TRAP_").casefold()))
            if not self.trap_due(event):
                self.emit(op, {"event": event, "skipped": True})
                return
            self.traps.append(event)
            self.emit(op, {"event": event, "skipped": False})
            for child in instruction.body:
                self.execute(child)
            return

        handler = getattr(self, f"_op_{op.casefold()}", None)
        if handler is None:
            self.violate(f"no VM handler for opcode {op!r}")
            self.emit(op, {"unhandled": True})
            return
        detail = handler(instruction.args, instruction)
        self.emit(op, detail if isinstance(detail, dict) else {"value": str(detail)})

    # -- contexts -----------------------------------------------------------
    def evaluation_context(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> EvaluationContext:
        plain = self._plain_state()
        if state:
            plain.update(dict(state))
        primary = next(iter(self.manifolds.values()), None)
        context_variables: dict[str, Any] = {
            "p": self.state.candidate,
            "m": primary,
            "repo": {"root": str(self.root), "ast": {"modules": list(self.state.files)}},
            "scale": self.state.scale,
            "ent": self.entanglement.spectral_entropy_bits if self.entanglement else 0.0,
            "eta": self.eta,
            "chi": self.kernel.chi_max,
            "sink": {"records": self.sink.count()},
            **self.variables,
        }
        for name, manifold in self.manifolds.items():
            context_variables[name] = manifold
        if variables:
            context_variables.update(dict(variables))
        return EvaluationContext(
            variables=context_variables,
            state=plain,
            functions=self._functions(),
            resolver=_Resolver(),
        )

    def transform_context(
        self,
        decl: g.TransformDecl,
        manifold: ManifoldState,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> EvaluationContext:
        context = self.evaluation_context(
            variables={"p": manifold, "args": list(args), "kwargs": dict(kwargs)}
        )
        for index, (param_name, _param_type) in enumerate(decl.params):
            context.variables[param_name] = args[index] if index < len(args) else manifold
        context.variables.update(kwargs)

        def reencode(target: Any, basis: str = "identity") -> Any:
            if isinstance(target, ManifoldState):
                target.basis = str(basis)
                target.data["basis"] = str(basis)
                target.data["reencodes"] = int(target.data.get("reencodes", 0)) + 1
            return target

        context.functions.update(
            {
                "reencode": reencode,
                "rotate_basis": reencode,
                "relabel": reencode,
            }
        )
        return context

    def execute_transform_statement(self, stmt: g.Statement, context: EvaluationContext) -> None:
        """Run a reversible transform body; destructive operations are refused."""
        if isinstance(stmt, (g.CollapseStmt, g.RecallStmt, g.SynthesizeStmt, g.TrapHandler, g.EvolveStmt)):
            raise GaugeRuntimeError(f"{type(stmt).__name__} is irreversible inside a transform")
        if isinstance(stmt, g.AssignStmt):
            value = evaluate(stmt.value, context)
            context.variables[stmt.target] = value
            if isinstance(value, ManifoldState):
                self.variables[stmt.target] = value
                self.manifolds[value.name] = value
            return
        if isinstance(stmt, g.ExprStmt):
            expr = stmt.expr
            if isinstance(expr, g.CallExpr):
                name = call_name(expr.callee)
                if name in IRREVERSIBLE_CALLS:
                    raise GaugeRuntimeError(f"irreversible call {name!r} inside a transform")
            evaluate(expr, context)
            return
        if isinstance(stmt, g.UncallStmt):
            transform = self.transforms.get(stmt.transform)
            if transform is not None:
                transform.uncall()
            return
        raise GaugeRuntimeError(f"unsupported transform statement {type(stmt).__name__}")

    # -- opcode handlers ----------------------------------------------------
    def _op_scale_init(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        uv = self._scale_number(args.get("uv_scale", "L0"))
        ir = self._scale_number(args.get("ir_scale", "L2"))
        self.scales = (uv, ir)
        self.state.scale = uv
        self.kernel.scale = uv
        return {"uv_scale": uv, "ir_scale": ir}

    def _op_manifold_init(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        manifold = ManifoldState(
            name=name,
            space=str(args.get("space", "PatchSpace")),
            constraints=list(args.get("constraints", ())),
            scale=self.state.scale,
        )
        self.manifolds[name] = manifold
        self.variables[name] = manifold
        return {"manifold": name, "space": manifold.space}

    def _op_gate_def(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        gate = GaugeGate(name, args.get("predicate"), self, bits=1.0 + 0.5 * len(args.get("params", ())))
        self.gate_defs[name] = gate
        self.kernel.gates = [*self.base_gates, *self.gate_defs.values()]
        return {"gate": name}

    def _op_op_def(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        deforms = tuple(str(item) for item in args.get("deforms", ()))
        after = tuple(str(item) for item in args.get("after", ()))
        commutes = tuple(str(item) for item in args.get("commutes", ()))
        self.op_after[name] = after
        self.op_commutes[name] = commutes

        def operator_fn(**kwargs: Any) -> dict[str, Any]:
            return {
                "operator": name,
                "deforms": list(deforms),
                "scale": self.state.scale,
                "state_keys": sorted(kwargs),
            }

        spec = OperatorSpec(
            name=name,
            fn=operator_fn,
            kind="WRITE" if deforms else "READ",
            writes=frozenset(deforms),
            reads=frozenset(deforms),
            cost=1.0 + 0.25 * len(after),
            description=f"Gauge extern op {name}",
        )
        self.registry.replace(spec)
        self.ops[name] = spec
        return {"op": name, "after": list(after), "deforms": list(deforms)}

    def _op_transform_rev(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        decl = g.TransformDecl(
            name=str(args["name"]),
            params=list(args.get("params", ())),
            return_type=str(args.get("return_type", "Manifold")),
            body=list(args.get("body", ())),
            reversible=bool(args.get("reversible", True)),
            span=_instruction.span,
        )
        self.transforms[decl.name] = ReversibleTransform(decl, self)
        return {"transform": decl.name, "reversible": decl.reversible}

    def _op_govern_begin(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        context = self.evaluation_context()
        config: dict[str, Any] = {}
        for key, expr in dict(args.get("config", {})).items():
            config[key] = evaluate(expr, context)
        self.govern_config.update(config)
        if "eta_min" in config:
            self.kernel.governor.eta_min = float(config["eta_min"])
        if "chi_max" in config:
            self.kernel.chi_max = max(1, int(config["chi_max"]))
        if "tau" in config:
            self.kernel.tau = float(config["tau"])
        return {"config": config}

    def _op_op_apply(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args.get("name") or call_name(args.get("expression")) or "")
        after = self.op_after.get(name, ())
        missing = [item for item in after if item not in self.applied_ops]
        if missing:
            raise GaugeRuntimeError(f"operator {name!r} requires after({', '.join(missing)})")
        spec = self.registry.operator(name)
        result = self.registry.apply(name, state=self._plain_state())
        if isinstance(result, Mapping) and "tokens" in result:
            self.cost.discarded_tokens += int(result.get("tokens", 0) or 0)
        self.cost.tool_cost += spec.cost
        self.applied_ops.add(name)
        return {"op": name, "result": result if isinstance(result, Mapping) else str(result)}

    def _op_ent_measure(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        scanner = RepoScanner(self.root, max_files=self.settings.max_files)
        graph = scanner.scan()
        constraints = [Constraint(id=f"manifold:{name}", paths=(".",)) for name in self.manifolds] or [
            Constraint(id="gauge:workspace", paths=(".",))
        ]
        coupling = graph.coupling(constraints)
        self.coupling_matrix = np.asarray(coupling.matrix, dtype=np.float64)
        self.entanglement = self.backend.measure_entanglement(
            self.coupling_matrix, chi_max=self.kernel.chi_max, tau=self.kernel.tau
        )
        self.coupling_fingerprint = matrix_fingerprint(self.coupling_matrix)
        return {
            "manifold": args.get("manifold"),
            "shape": list(self.coupling_matrix.shape),
            **self.entanglement.as_dict(),
        }

    def _op_gate_project(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        names = [str(item) for item in args.get("gates", ())]
        state = self._plain_state()
        results: dict[str, GateResult] = {}
        for name in names:
            gate = self.gate_defs.get(name)
            if gate is None:
                results[name] = GateResult(name, None, 1.0, "undeclared gate")
            else:
                results[name] = gate.evaluate(state)
        self.gate_results = results
        self.verified_drop = sum(
            (result.bits for result in results.values() if result.passed is True), 0.0
        ) - sum((result.bits for result in results.values() if result.passed is False), 0.0)
        self.state.has_regressions = any(result.passed is False for result in results.values())
        return {name: result.verdict for name, result in results.items()}

    def _op_govern_check(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        report = self.entanglement or EntanglementReport(0.0, 0.0, 0, False)
        self.eta, self.directive = self.kernel.governor.evaluate(self.verified_drop, self.cost, report)
        rolling = self.kernel.governor.rolling_eta
        self.drag = bool(
            self.eta <= self.kernel.governor.eta_min
            or (rolling is not None and rolling <= self.kernel.governor.eta_min)
        )
        self.events.clear()
        if report.is_volume_law:
            self.events.update({"volume_law", "shift"})
        if self.drag:
            self.events.update({"drag", "stall"})
        if self.directive is Directive.RECALL_SINK:
            self.events.update({"frustration", "recall"})
        return {
            "eta": round(self.eta, 6),
            "directive": self.directive.value,
            "events": sorted(self.events),
        }

    def _op_evolve(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        report = self.entanglement or EntanglementReport(0.0, 0.0, 0, False)
        self.tick = self.kernel.tick(
            self._plain_state(),
            self.coupling_matrix,
            self.cost,
            gate_results=self.gate_results,
            verified_drop=self.verified_drop,
            entanglement=report,
            eta_cog=self.eta,
            governor_directive=self.directive,
            commit_gate_state=False,
            label=f"gauge:evolve:{args.get('manifold', 'manifold')}",
        )
        return {"tick": self.tick.hash[:10], "directive": self.tick.directive}

    def _op_transform_apply(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["transform"])
        transform = self.transforms.get(name)
        if transform is None:
            raise GaugeRuntimeError(f"unknown reversible transform {name!r}")
        context = self.evaluation_context()
        evaluated_args = [evaluate(item, context) for item in args.get("args", ())]
        kwargs = {key: evaluate(item, context) for key, item in dict(args.get("kwargs", {})).items()}
        manifold = next(
            (item for item in evaluated_args if isinstance(item, ManifoldState)),
            None,
        )
        if manifold is None:
            source = args.get("target")
            candidate = self.variables.get(str(source)) if source else None
            manifold = (
                candidate
                if isinstance(candidate, ManifoldState)
                else next(iter(self.manifolds.values()), None)
            )
        if manifold is None:
            raise GaugeRuntimeError(f"transform {name!r} has no manifold to deform")
        transform.apply(manifold, evaluated_args, kwargs)
        target = args.get("target")
        if target:
            self.variables[str(target)] = manifold
        return {
            "transform": name,
            "manifold": manifold.name,
            "basis": manifold.basis,
            "inverse_depth": len(self.inverse_stack),
        }

    def _op_uncall_transform(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["transform"])
        transform = self.transforms.get(name)
        context = self.evaluation_context()
        evaluated_args = [evaluate(item, context) for item in args.get("args", ())]
        manifold = next((item for item in evaluated_args if isinstance(item, ManifoldState)), None)
        restored = transform.uncall(manifold) if transform is not None else None
        return {"transform": name, "restored": restored.name if restored else ""}

    def _op_synthesize_op(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args.get("op_name", "synthesized_op"))
        metric = str(args.get("metric", "eta"))
        state = self._plain_state()
        result = self.synthesizer.synthesize(name, metric, state)
        accepted, detail, improvement = self._validate_synthesis(result, state, metric)
        event = result.as_dict()
        event.update({"accepted": accepted, "detail": detail, "efficiency_improvement": improvement})
        self.synthesis_events.append(event)
        self.result.synthesis = list(self.synthesis_events)
        self.sink.put(
            "gauge_synthesis" if accepted else "gauge_synthesis_rejected",
            {
                "name": result.name,
                "metric": metric,
                "accepted": accepted,
                "detail": detail,
                "origin": result.origin,
            },
            scale=self.scales[1] if self.scales else self.kernel.scale,
            parent_hash=self.tick.hash if self.tick else self.kernel.last_hash,
            eta_cog=self.eta,
        )
        return {"op": result.name, "accepted": accepted, "improvement": improvement, "detail": detail}

    def _op_recall_sink(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        condition = args.get("condition")
        verdict = evaluate(condition, self.evaluation_context()) if condition is not None else True
        if verdict is not True:
            return {"recalled": 0, "condition": verdict}
        records = self.sink.records(
            kinds=("gauge_collapse", "pruned_branch", "uv_state"),
            scale=self.state.scale,
            limit=8,
            oldest_first=False,
        )
        self.recalled = [record.as_dict() for record in records]
        self.result.recalled = list(self.recalled)
        for name, manifold_name, snapshot in reversed(self.inverse_stack):
            _ = name
            target = self.manifolds.get(manifold_name)
            if target is not None:
                target.restore(snapshot)
        self.inverse_stack.clear()
        self.kernel.restore_checkpoint(self.kernel_checkpoint)
        self.kernel.reconcile_gates(self._plain_state())
        self.gate_state_committed = True
        return {"recalled": len(self.recalled), "scale": self.state.scale}

    def _op_collapse_sink(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["manifold"])
        manifold = self.manifolds.get(name)
        if manifold is None:
            raise GaugeRuntimeError(f"cannot collapse unknown manifold {name!r}")
        if manifold.collapsed:
            raise GaugeRuntimeError(f"manifold {name!r} was already collapsed")
        leaks = sorted(
            other.name for other in self.manifolds.values() if other.name != name and not other.collapsed
        )
        if leaks:
            raise GaugeRuntimeError(f"unarchived manifold leak into sink: {', '.join(leaks)}")
        uv_scale, ir_scale = self.scales or (self.kernel.scale, self.kernel.scale + 1)
        payload = {
            "manifold": manifold.as_dict(),
            "gates": {key: value.verdict for key, value in self.gate_results.items()},
            "entanglement": self.entanglement.as_dict() if self.entanglement else {},
            "eta": round(self.eta, 6),
            "directive": self.directive.value,
            "uv_scale": uv_scale,
            "ir_scale": ir_scale,
            "inverse_depth": len(self.inverse_stack),
            "coupling_fingerprint": self.coupling_fingerprint,
        }
        digest = self.sink.put(
            "gauge_collapse",
            payload,
            scale=ir_scale,
            parent_hash=self.tick.hash if self.tick else self.kernel.last_hash,
            eta_cog=self.eta,
        )
        manifold.collapsed = True
        manifold.data["sink"] = digest
        self.collapses[name] = digest
        self.result.collapses = dict(self.collapses)
        if args.get("target"):
            self.variables[str(args["target"])] = digest
        report = self.graph.coarse_grain(
            scale=uv_scale,
            chi_max=self.kernel.chi_max,
            tau=self.kernel.tau,
            kinds=("uv_state", "gauge_collapse"),
        )
        self.result.mera = report.as_dict()
        return {"manifold": name, "sink": digest[:10], "scale": ir_scale, "mera": report.clusters}

    def _op_assign(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        value = evaluate(args["value"], self.evaluation_context())
        self.variables[str(args["target"])] = value
        return {"target": args["target"], "value": str(value)}

    def _op_expr(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"value": str(evaluate(args["value"], self.evaluation_context()))}

    def _op_noop(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"noop": args}

    # -- internals ----------------------------------------------------------
    def _reset(self, name: str) -> None:
        self.program: g.Program | None = None
        self.checked: CheckedProgram | None = None
        self.ir: ProgramIR | None = None
        self.result = GaugeRunResult(status="running", program=name, root=str(self.root))
        self.trace: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self.result.trace = self.trace
        self.result.violations = self.violations
        self.state = GaugeState(self.root)
        self.manifolds: dict[str, ManifoldState] = {}
        self.transforms: dict[str, ReversibleTransform] = {}
        self.gate_defs: dict[str, GaugeGate] = {}
        self.ops: dict[str, OperatorSpec] = {}
        self.op_after: dict[str, tuple[str, ...]] = {}
        self.op_commutes: dict[str, tuple[str, ...]] = {}
        self.applied_ops: set[str] = set()
        self.inverse_stack: list[tuple[str, str, dict[str, Any]]] = []
        self.variables: dict[str, Any] = {}
        self.scales: tuple[int, int] | None = None
        self.govern_config: dict[str, Any] = {}
        self.gate_results: dict[str, GateResult] = {}
        self.verified_drop = 0.0
        self.entanglement: EntanglementReport | None = None
        self.coupling_matrix = np.zeros((0, 0), dtype=np.float64)
        self.coupling_fingerprint = ""
        self.eta = 0.0
        self.directive = Directive.CRYSTALLIZE
        self.drag = False
        self.events: set[str] = set()
        self.traps: list[str] = []
        self.result.traps = self.traps
        self.synthesis_events: list[dict[str, Any]] = []
        self.result.synthesis = self.synthesis_events
        self.collapses: dict[str, str] = {}
        self.result.collapses = self.collapses
        self.recalled: list[dict[str, Any]] = []
        self.result.recalled = self.recalled
        self.tick: TickResult | None = None
        self.cost = CognitiveStepCost()
        self.status = "running"
        self.halted = False
        self.gate_state_committed = False
        self.kernel_checkpoint = self.kernel.checkpoint()

    def _scan_workspace(self) -> None:
        scanner = RepoScanner(self.root, max_files=self.settings.max_files)
        graph = scanner.scan()
        files = [info.path for info in graph.modules.values()]
        sources: dict[str, str] = {}
        for relative in files:
            try:
                sources[relative] = (self.root / relative).read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
        self.state.files = files
        self.state.sources = sources
        self.state.targets = [
            path for path in files if Path(path).name.startswith("test_") or path.endswith("_test.py")
        ]
        self.state.task = f"gauge:{self.result.program}"

    def _plain_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "program": self.result.program,
            "repo_root": str(self.root),
            "paths": list(self.state.files),
            "files": list(self.state.files),
            "sources": dict(self.state.sources),
            "targets": tuple(self.state.targets),
            "task": self.state.task,
            "scale": self.state.scale,
            "chi": self.kernel.chi_max,
            "eta": self.eta,
            "ent": self.entanglement.spectral_entropy_bits if self.entanglement else 0.0,
            "truncation": self.entanglement.truncation_error if self.entanglement else 0.0,
            "volume_law": self.entanglement.is_volume_law if self.entanglement else False,
            "has_regressions": self.state.has_regressions,
            "coupling_shape": list(self.coupling_matrix.shape),
            "coupling_fingerprint": self.coupling_fingerprint,
            "gates": {name: result.verdict for name, result in self.gate_results.items()},
            "repo": {"root": str(self.root), "ast": {"modules": list(self.state.files)}},
            "p": {
                "files": list(self.state.files),
                "targets": list(self.state.targets),
                "has_regressions": self.state.has_regressions,
            },
        }
        for name, manifold in self.manifolds.items():
            state[name] = manifold.as_dict()
        return state

    def _functions(self) -> dict[str, Any]:
        def ast_check(files: Any = None) -> bool | None:
            candidates = (
                list(files)
                if isinstance(files, Sequence) and not isinstance(files, str)
                else list(self.state.files)
            )
            checked = 0
            for relative in candidates:
                relative = str(relative)
                if not relative.endswith(".py"):
                    continue
                source = self.state.sources.get(relative)
                if source is None:
                    try:
                        source = (self.root / relative).read_bytes().decode("utf-8", errors="replace")
                    except OSError:
                        return None
                checked += 1
                try:
                    compile(source, relative, "exec")
                except SyntaxError:
                    return False
            return True if checked else None

        def pytest_run(targets: Any = None) -> bool | None:
            candidates = [
                str(item)
                for item in (targets if targets is not None else self.state.targets)
                if str(item).endswith(".py")
                and ("test" in Path(str(item)).name or "tests" in Path(str(item)).parts)
            ]
            if not candidates:
                return None
            result = run_pytest(self.root, candidates, timeout=min(self.settings.timeout_s, 60.0))
            return result["passed"]

        return {
            "ast_check": ast_check,
            "syntax_check": ast_check,
            "pytest_run": pytest_run,
            "tests_pass": pytest_run,
            "entropy": lambda *_: self.entanglement.spectral_entropy_bits if self.entanglement else None,
            "eta": lambda *_: self.eta,
            "chi": lambda *_: self.kernel.chi_max,
            "truncation": lambda *_: self.entanglement.truncation_error if self.entanglement else None,
            "rank": lambda *_: self.entanglement.effective_rank if self.entanglement else None,
            "volume_law": lambda *_: self.entanglement.is_volume_law if self.entanglement else None,
            "regressions": lambda *_: self.state.has_regressions,
            "len": len,
            "abs": abs,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "dict": dict,
            "list": list,
        }

    def _validate_synthesis(
        self, result: SynthesisResult, state: Mapping[str, Any], metric: str
    ) -> tuple[bool, str, float]:
        artifact: ShiftArtifact = result.artifact
        ok, detail = artifact.validate(state=state, entanglement=self.entanglement)
        if not result.accepted or not ok:
            return False, detail or result.detail, 0.0
        source = str(artifact.payload.get("source", result.source))
        namespace: dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
            }
        }
        try:
            exec(compile(source, f"<gauge:{result.name}>", "exec"), namespace, namespace)
            fn = namespace[str(artifact.payload.get("name", result.name))]
            before = copy.deepcopy(dict(state))
            try:
                candidate = fn(copy.deepcopy(before), metric=metric)
            except TypeError:
                try:
                    candidate = fn(copy.deepcopy(before))
                except TypeError:
                    candidate = fn()
            if before != state:
                return False, "synthesized operator mutated its input", 0.0
            if not isinstance(candidate, Mapping):
                return False, "synthesized operator did not return a candidate state", 0.0
            improvement = float(candidate.get("gauge_metric_gain", 0.0) or 0.0)
        except Exception as exc:
            return False, f"synthesized operator is not executable: {type(exc).__name__}: {exc}", 0.0
        if improvement <= 0.0:
            return False, "no positive efficiency improvement", improvement

        def registered_fn(**kwargs: Any) -> dict[str, Any]:
            payload = copy.deepcopy(dict(kwargs.get("state", {})))
            try:
                produced = fn(payload, metric=kwargs.get("metric", metric))
            except TypeError:
                produced = fn(payload)
            return dict(produced) if isinstance(produced, Mapping) else payload

        self.registry.replace(
            OperatorSpec(
                name=result.name,
                fn=registered_fn,
                kind="READ",
                reads=frozenset({"coordinate/*"}),
                cost=0.0,
                description=f"synthesized Gauge operator improving {metric}",
            )
        )
        self.sink.put(
            "gauge_synthesis",
            {
                "name": result.name,
                "metric": metric,
                "improvement": improvement,
                "artifact": artifact.as_dict(),
            },
            scale=self.scales[1] if self.scales else self.kernel.scale,
            parent_hash=self.tick.hash if self.tick else self.kernel.last_hash,
            eta_cog=self.eta,
        )
        self.cost.discarded_tokens += artifact.tokens
        self.cost.latency_ms += artifact.latency_ms
        return True, "", improvement

    def trap_due(self, event: str) -> bool:
        aliases = {
            "volume_law": {"volume_law", "frustration"},
            "drag": {"drag", "stall"},
            "frustration": {"frustration", "recall"},
            "stall": {"stall", "drag"},
            "shift": {"shift", "volume_law"},
            "recall": {"recall", "frustration"},
        }
        return bool(self.events & aliases.get(event, {event}))

    def emit(self, op: str, detail: Mapping[str, Any]) -> None:
        payload = {key: value for key, value in dict(detail).items() if key != "op"}
        self.trace.append({"op": op, **payload})
        self.result.trace = self.trace

    def violate(self, message: str) -> None:
        self.violations.append(message)
        self.result.violations = self.violations

    @staticmethod
    def _scale_number(value: Any) -> int:
        if isinstance(value, int):
            return value
        text = str(value)
        if len(text) > 1 and text[0] in {"L", "l"} and text[1:].isdigit():
            return int(text[1:])
        return int(text)


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
