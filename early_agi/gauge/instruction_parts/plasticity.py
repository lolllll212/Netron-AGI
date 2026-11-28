"""Read-only probe, analogy recall, and collapse-to-sink opcodes."""

from __future__ import annotations

from typing import Any

from early_agi.gauge.errors import GaugeRuntimeError
from early_agi.gauge.ir import Instruction

__all__ = ["_PlasticityOps"]


class _PlasticityOps:
    """Plasticity/MERA/sink opcode handlers."""

    @staticmethod
    def _probe_call_kwargs(name: str, execution: Any) -> dict[str, Any]:
        """Baseline tests and SMT checks are read-only but have fixed signatures."""
        if name == "run_pytest":
            return {"paths": (), "timeout": 30.0}
        if name == "check_z3":
            return {"smt2": "(check-sat)", "timeout_ms": 1000}
        return {"state": execution.plain_state()}

    def _op_probe(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        """Run read-only extern operators before any mutating Gauge op."""
        manifold = str(args.get("manifold", ""))
        if manifold not in self.execution.manifolds:
            raise GaugeRuntimeError(f"cannot probe unknown manifold {manifold!r}")
        ops = [str(item) for item in args.get("ops", ())]
        preflight = bool(args.get("preflight", False))
        if self.execution.probe_results and not preflight:
            return {
                "manifold": manifold,
                "operators": ops,
                "results": len(self.execution.probe_results),
                "already_probed": True,
            }
        calls = [(name, self._probe_call_kwargs(name, self.execution)) for name in ops]
        probe = self.vm.registry.epistemic_probe(calls) if calls else {"operators": [], "results": []}
        self.execution.probe_results = list(probe["results"])
        self.vm.kernel.governor.clear_probe()
        self.vm.kernel.release_probe_lock()
        thresholds = self.vm.kernel.governor.modulate(max(0.5, self.execution.analogy_similarity), 0.5)
        self.execution.novelty_score = thresholds.novelty_score
        if not preflight:
            self.execution.events.discard("high_novelty")
        self.execution.plasticity.update(
            {
                "thresholds": thresholds.as_dict(),
                "novelty_score": thresholds.novelty_score,
                "probe": probe,
                "distilled_constraints": list(self.execution.distilled_constraints),
            }
        )
        self.vm.result.plasticity = dict(self.execution.plasticity)
        return {
            "manifold": manifold,
            "operators": ops,
            "results": len(probe["results"]),
            "probe": probe,
            "preflight": preflight,
        }

    def _op_recall_analogy(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        """Recall a structural strategy from MERA L1/L2 memory."""
        min_similarity = float(args.get("min_similarity", 0.72))
        matrix = self._measure_workspace() if self.execution.coupling_matrix.size == 0 else None
        matrix = matrix if matrix is not None else self.execution.coupling_matrix
        num_modules, num_imports, num_functions, num_gates = self._motif_counts(self.execution)
        signature = self.vm.graph.encode_signature(
            matrix,
            num_modules=num_modules,
            num_imports=num_imports,
            num_functions=num_functions,
            num_gates=num_gates,
        )
        match = self.vm.graph.best_analogy(signature, min_similarity)
        self.execution.analogy = match.as_dict() if match is not None else None
        self.execution.analogy_similarity = match.similarity if match is not None else 0.0
        if match is not None:
            self.execution.predicted_drop = float(match.metadata.get("eta_mean", 0.0) or 0.0)
        probe_familiarity = 0.5 if self.execution.probe_results else 0.0
        familiarity = max(self.execution.analogy_similarity, probe_familiarity)
        thresholds = self.vm.kernel.governor.modulate(familiarity, 0.5)
        self.execution.novelty_score = thresholds.novelty_score
        self.execution.plasticity.update(
            {
                "thresholds": thresholds.as_dict(),
                "novelty_score": thresholds.novelty_score,
                "analogy": self.execution.analogy,
            }
        )
        self.vm.result.plasticity = dict(self.execution.plasticity)
        return {
            "match": self.execution.analogy is not None,
            "similarity": self.execution.analogy_similarity,
            "strategy_hint": match.strategy_hint if match is not None else "",
            "min_similarity": min_similarity,
        }

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
