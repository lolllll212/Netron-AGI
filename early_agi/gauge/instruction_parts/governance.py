"""Measurement, gate projection, governance, evolve, and synthesis opcodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from early_agi.core.entropy_sensor import EntanglementReport, matrix_fingerprint
from early_agi.core.governor import Directive
from early_agi.gauge.checker import call_name
from early_agi.gauge.errors import GaugeRuntimeError
from early_agi.gauge.ir import Instruction
from early_agi.operators.dark_state import GateResult

from .helpers import _append_distilled_constraints

__all__ = ["_GovernanceOps"]


class _GovernanceOps:
    """Measurement, governance, and synthesis opcode handlers."""

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
        self._measure_workspace()
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
        distilled = _append_distilled_constraints(self.execution, results)
        self.execution.plasticity["distilled_constraints"] = distilled
        self.vm.result.plasticity = dict(self.execution.plasticity)
        self.vm.result.manifolds = {
            name: manifold.as_dict() for name, manifold in self.execution.manifolds.items()
        }
        return {name: result.verdict for name, result in results.items()} | (
            {"distilled_constraints": distilled} if distilled else {}
        )

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
        thresholds = self.vm.kernel.governor.last_thresholds
        if thresholds is not None:
            self.execution.novelty_score = thresholds.novelty_score
            if thresholds.novelty_score > 0.65:
                self.execution.events.add("high_novelty")
        if report.is_volume_law:
            self.execution.events.update({"volume_law", "shift"})
        if self.execution.drag:
            self.execution.events.update({"drag", "stall"})
        if self.execution.directive is Directive.RECALL_SINK:
            self.execution.events.update({"frustration", "recall"})
        if self.execution.plasticity.get("preflight_probe") and self.execution.probe_results:
            # The preflight directive is part of the trace, and the nested trap
            # still records the declared read-only probe without repeating it.
            self.execution.directive = Directive.EPISTEMIC_PROBE
            self.execution.events.add("high_novelty")
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
            predicted_drop=self.execution.predicted_drop,
            analogy_similarity=self.execution.analogy_similarity,
            risk_level=0.5,
            probe_completed=bool(self.execution.probe_results),
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
