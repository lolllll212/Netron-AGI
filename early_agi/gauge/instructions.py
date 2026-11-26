"""Opcode handlers for the Gauge VM.

``vm.py`` is only a dispatcher. This module owns the instruction semantics while
``state.py`` owns mutable views, ``builtins.py`` owns expression vocabulary, and
``traps.py`` owns trap evaluation and inverse-stack recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.coupling import Constraint, RepoScanner
from ..core.entropy_sensor import EntanglementReport, matrix_fingerprint
from ..core.governor import Directive
from ..operators.dark_state import GateResult
from ..operators.registry import OperatorSpec
from . import ast as g
from .builtins import GaugeGate
from .checker import call_name
from .errors import GaugeRuntimeError
from .expressions import evaluate
from .ir import Instruction
from .runtime import ManifoldState, ReversibleTransform

__all__ = ["GaugeOpcodeRunner"]


class GaugeOpcodeRunner:
    """Execute one bytecode instruction against the VM's focused modules."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    @property
    def execution(self) -> Any:
        return self.vm.execution

    @property
    def traps(self) -> Any:
        return self.vm.traps_runner

    def execute(self, instruction: Instruction) -> None:
        op = instruction.op
        if op.startswith("TRAP_"):
            self.traps.execute(instruction, self.vm.execute)
            return
        handler = getattr(self, f"_op_{op.casefold()}", None)
        if handler is None:
            self.vm.violate(f"no VM handler for opcode {op!r}")
            self.vm.emit(op, {"unhandled": True})
            return
        detail = handler(instruction.args, instruction)
        self.vm.emit(op, detail if isinstance(detail, Mapping) else {"value": str(detail)})

    # -- setup --------------------------------------------------------------
    def _op_halt(self, _args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        self.execution.halted = True
        return {}

    def _op_scale_init(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        uv = self.execution.scale_number(args.get("uv_scale", "L0"))
        ir = self.execution.scale_number(args.get("ir_scale", "L2"))
        self.execution.scales = (uv, ir)
        self.execution.state.scale = uv
        self.vm.kernel.scale = uv
        self.execution.view()
        return {"uv_scale": uv, "ir_scale": ir}

    def _op_manifold_init(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        manifold = ManifoldState(
            name=name,
            space=str(args.get("space", "PatchSpace")),
            constraints=list(args.get("constraints", ())),
            scale=self.execution.state.scale,
        )
        self.execution.manifolds[name] = manifold
        self.execution.variables[name] = manifold
        return {"manifold": name, "space": manifold.space}

    def _op_gate_def(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        gate = GaugeGate(
            name,
            args.get("predicate"),
            self.vm,
            bits=1.0 + 0.5 * len(args.get("params", ())),
        )
        self.execution.gate_defs[name] = gate
        self.vm.kernel.gates = [*self.vm.base_gates, *self.execution.gate_defs.values()]
        return {"gate": name}

    def _op_op_def(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["name"])
        deforms = tuple(str(item) for item in args.get("deforms", ()))
        after = tuple(str(item) for item in args.get("after", ()))
        commutes = tuple(str(item) for item in args.get("commutes", ()))
        self.execution.op_after[name] = after
        self.execution.op_commutes[name] = commutes

        def operator_fn(**kwargs: Any) -> dict[str, Any]:
            return {
                "operator": name,
                "deforms": list(deforms),
                "scale": self.execution.state.scale,
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
        self.vm.registry.replace(spec)
        self.execution.ops[name] = spec
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
        self.execution.transforms[decl.name] = ReversibleTransform(decl, self.vm)
        return {"transform": decl.name, "reversible": decl.reversible}

    def _op_govern_begin(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        context = self.execution.evaluation_context()
        config: dict[str, Any] = {}
        for key, expr in dict(args.get("config", {})).items():
            config[key] = evaluate(expr, context)
        self.execution.govern_config.update(config)
        if "eta_min" in config:
            self.vm.kernel.governor.eta_min = float(config["eta_min"])
        if "chi_max" in config:
            self.vm.kernel.chi_max = max(1, int(config["chi_max"]))
        if "tau" in config:
            self.vm.kernel.tau = float(config["tau"])
        if hasattr(self.vm.synthesizer, "eta_min"):
            self.vm.synthesizer.eta_min = self.vm.kernel.governor.eta_min
        return {"config": config}

    # -- measurement and governance ----------------------------------------
    def _op_op_apply(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args.get("name") or call_name(args.get("expression")) or "")
        after = self.execution.op_after.get(name, ())
        missing = [item for item in after if item not in self.execution.applied_ops]
        if missing:
            raise GaugeRuntimeError(f"operator {name!r} requires after({', '.join(missing)})")
        spec = self.vm.registry.operator(name)
        result = self.vm.registry.apply(name, state=self.execution.plain_state())
        if isinstance(result, Mapping) and "tokens" in result:
            self.execution.cost.discarded_tokens += int(result.get("tokens", 0) or 0)
        self.execution.cost.tool_cost += spec.cost
        self.execution.applied_ops.add(name)
        self.execution.applied_op_chain.append(name)
        return {"op": name, "result": result if isinstance(result, Mapping) else str(result)}

    def _op_ent_measure(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        scanner = RepoScanner(self.vm.root, max_files=self.vm.settings.max_files)
        graph = scanner.scan()
        constraints = [
            Constraint(id=f"manifold:{name}", paths=(".",)) for name in self.execution.manifolds
        ] or [Constraint(id="gauge:workspace", paths=(".",))]
        coupling = graph.coupling(constraints)
        self.execution.coupling_matrix = np.asarray(coupling.matrix, dtype=np.float64)
        self.execution.entanglement = self.vm.backend.measure_entanglement(
            self.execution.coupling_matrix,
            chi_max=self.vm.kernel.chi_max,
            tau=self.vm.kernel.tau,
        )
        self.execution.coupling_fingerprint = matrix_fingerprint(self.execution.coupling_matrix)
        return {
            "manifold": args.get("manifold"),
            "shape": list(self.execution.coupling_matrix.shape),
            **self.execution.entanglement.as_dict(),
        }

    def _op_gate_project(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        names = [str(item) for item in args.get("gates", ())]
        state = self.execution.plain_state()
        results: dict[str, GateResult] = {}
        for name in names:
            gate = self.execution.gate_defs.get(name)
            results[name] = (
                GateResult(name, None, 1.0, "undeclared gate") if gate is None else gate.evaluate(state)
            )
        self.execution.gate_results = results
        self.execution.verified_drop = sum(
            (result.bits for result in results.values() if result.passed is True), 0.0
        ) - sum((result.bits for result in results.values() if result.passed is False), 0.0)
        self.execution.state.has_regressions = any(result.passed is False for result in results.values())
        return {name: result.verdict for name, result in results.items()}

    def _op_govern_check(self, _args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        report = self.execution.entanglement or EntanglementReport(0.0, 0.0, 0, False)
        self.execution.eta, self.execution.directive = self.vm.kernel.governor.evaluate(
            self.execution.verified_drop, self.execution.cost, report
        )
        rolling = self.vm.kernel.governor.rolling_eta
        self.execution.drag = bool(
            self.execution.eta <= self.vm.kernel.governor.eta_min
            or (rolling is not None and rolling <= self.vm.kernel.governor.eta_min)
        )
        self.execution.events.clear()
        if report.is_volume_law:
            self.execution.events.update({"volume_law", "shift"})
        if self.execution.drag:
            self.execution.events.update({"drag", "stall"})
        if self.execution.directive is Directive.RECALL_SINK:
            self.execution.events.update({"frustration", "recall"})
        return {
            "eta": round(self.execution.eta, 6),
            "directive": self.execution.directive.value,
            "events": sorted(self.execution.events),
        }

    def _op_evolve(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        report = self.execution.entanglement or EntanglementReport(0.0, 0.0, 0, False)
        self.execution.tick = self.vm.kernel.tick(
            self.execution.plain_state(),
            self.execution.coupling_matrix,
            self.execution.cost,
            gate_results=self.execution.gate_results,
            verified_drop=self.execution.verified_drop,
            entanglement=report,
            eta_cog=self.execution.eta,
            governor_directive=self.execution.directive,
            commit_gate_state=False,
            label=f"gauge:evolve:{args.get('manifold', 'manifold')}:{self.execution.run_index}",
        )
        chain = tuple(self.execution.applied_op_chain) or tuple(sorted(self.execution.applied_ops))
        gate_verdicts = {name: result.verdict for name, result in self.execution.gate_results.items()}
        chain_ok = not self.execution.state.has_regressions and all(
            result.passed is not False for result in self.execution.gate_results.values()
        )
        observe = getattr(self.vm.synthesizer, "observe_operator_chain", None)
        if callable(observe):
            observe(
                chain,
                ok=chain_ok,
                eta=self.execution.eta,
                tick_hash=self.execution.tick.hash,
                gates=gate_verdicts,
                scale=self.execution.state.scale,
                task=self.execution.state.task,
                run_index=self.execution.run_index,
            )
        self.execution.applied_op_chain.clear()
        compile_chains = getattr(self.vm.synthesizer, "compile_hot_chains", None)
        self.vm.result.reflexes = (
            compile_chains(min_support=2)
            if callable(compile_chains)
            else {"fused": 0, "operators": [], "signatures": 0, "skipped": ["no chain compiler"]}
        )
        return {
            "tick": self.execution.tick.hash[:10],
            "directive": self.execution.tick.directive,
            "chain": list(chain),
            "reflexes": self.vm.result.reflexes.get("fused", 0),
        }

    # -- transforms, synthesis, recall, and collapse -----------------------
    def _op_transform_apply(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return self.execution.apply_transform(args)

    def _op_uncall_transform(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return self.traps.uncall_transform(args)

    def _op_synthesize_op(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args.get("op_name", "synthesized_op"))
        metric = str(args.get("metric", "eta"))
        if hasattr(self.vm.synthesizer, "eta_min"):
            self.vm.synthesizer.eta_min = self.vm.kernel.governor.eta_min
        state = self.execution.plain_state()
        try:
            result = self.vm.synthesizer.synthesize(
                name,
                metric,
                state,
                entanglement=self.execution.entanglement,
            )
        except TypeError as exc:
            if "entanglement" not in str(exc):
                raise
            result = self.vm.synthesizer.synthesize(name, metric, state)
        replay = getattr(result, "replay", {}) or {}
        event = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        detail = str(getattr(result, "detail", "") or replay.get("detail", ""))
        accepted = bool(event.get("accepted", getattr(result, "accepted", False)))
        event["detail"] = detail
        event.setdefault("accepted", accepted)
        event.setdefault("replay", dict(replay))
        event.setdefault("registered", bool(getattr(result, "registered", False)))
        event.setdefault("promoted", bool(getattr(result, "promoted", accepted)))
        self.execution.synthesis_events.append(event)
        self.vm.result.synthesis = self.execution.synthesis_events
        self.execution.cost.discarded_tokens += result.artifact.tokens
        self.execution.cost.latency_ms += result.artifact.latency_ms
        return {
            "op": result.name,
            "accepted": accepted,
            "registered": event["registered"],
            "improvement": float(getattr(result, "efficiency_improvement", 0.0) or 0.0),
            "replay_cases": replay.get("cases", 0),
            "strictly_improved": replay.get("strictly_improved", False),
            "gates_green": replay.get("gates_green", False),
            "detail": detail,
        }

    def _op_recall_sink(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return self.traps.recall_sink(args.get("condition"))

    def _op_collapse_sink(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        name = str(args["manifold"])
        manifold = self.execution.manifolds.get(name)
        if manifold is None:
            raise GaugeRuntimeError(f"cannot collapse unknown manifold {name!r}")
        if manifold.collapsed:
            raise GaugeRuntimeError(f"manifold {name!r} was already collapsed")
        leaks = sorted(
            other.name
            for other in self.execution.manifolds.values()
            if other.name != name and not other.collapsed
        )
        if leaks:
            raise GaugeRuntimeError(f"unarchived manifold leak into sink: {', '.join(leaks)}")
        uv_scale, ir_scale = self.execution.scales or (
            self.vm.kernel.scale,
            self.vm.kernel.scale + 1,
        )
        payload = {
            "manifold": manifold.as_dict(),
            "workspace": self.execution.view().as_dict(),
            "gates": {key: value.verdict for key, value in self.execution.gate_results.items()},
            "entanglement": (self.execution.entanglement.as_dict() if self.execution.entanglement else {}),
            "eta": round(self.execution.eta, 6),
            "directive": self.execution.directive.value,
            "uv_scale": uv_scale,
            "ir_scale": ir_scale,
            "run_index": self.execution.run_index,
            "inverse_depth": len(self.execution.inverse_stack),
            "coupling_fingerprint": self.execution.coupling_fingerprint,
        }
        digest = self.vm.sink.put(
            "gauge_collapse",
            payload,
            scale=ir_scale,
            parent_hash=self.execution.tick.hash if self.execution.tick else self.vm.kernel.last_hash,
            eta_cog=self.execution.eta,
        )
        manifold.collapsed = True
        manifold.data["sink"] = digest
        self.execution.collapses[name] = digest
        self.vm.result.collapses = self.execution.collapses
        if args.get("target"):
            self.execution.variables[str(args["target"])] = digest
        report = self.vm.graph.coarse_grain(
            scale=uv_scale,
            chi_max=self.vm.kernel.chi_max,
            tau=self.vm.kernel.tau,
            kinds=("uv_state", "gauge_collapse"),
        )
        self.vm.result.mera = report.as_dict()
        return {
            "manifold": name,
            "sink": digest[:10],
            "scale": ir_scale,
            "mera": report.clusters,
        }

    # -- trivial expressions ------------------------------------------------
    def _op_assign(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        value = evaluate(args["value"], self.execution.evaluation_context())
        self.execution.variables[str(args["target"])] = value
        return {"target": args["target"], "value": str(value)}

    def _op_expr(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"value": str(evaluate(args["value"], self.execution.evaluation_context()))}

    def _op_noop(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        return {"noop": args}
