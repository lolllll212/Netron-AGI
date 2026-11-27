"""VM dispatch, workspace measurement, and setup/govern-begin opcodes."""

from __future__ import annotations

import ast as pyast
from collections.abc import Mapping
from typing import Any

import numpy as np

from early_agi.core.coupling import Constraint, RepoScanner
from early_agi.core.entropy_sensor import matrix_fingerprint
from early_agi.gauge import ast as g
from early_agi.gauge.builtins import GaugeGate
from early_agi.gauge.expressions import evaluate
from early_agi.gauge.ir import Instruction
from early_agi.gauge.runtime import ManifoldState, ReversibleTransform
from early_agi.operators.registry import OperatorSpec

from .helpers import _mutating_footprint

__all__ = ["_SetupOps"]


class _SetupOps:
    """Dispatcher and setup/govern-begin opcode handlers."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    @property
    def execution(self) -> Any:
        return self.vm.execution

    @property
    def traps(self) -> Any:
        return self.vm.traps_runner

    def _measure_workspace(self) -> np.ndarray:
        """Refresh the coupling matrix without advancing the governor."""
        scanner = RepoScanner(self.vm.root, max_files=self.vm.settings.max_files)
        graph = scanner.scan()
        constraints = [
            Constraint(id=f"manifold:{name}", paths=(".",)) for name in self.execution.manifolds
        ] or [Constraint(id="gauge:workspace", paths=(".",))]
        coupling = graph.coupling(constraints)
        self.execution.coupling_matrix = np.asarray(coupling.matrix, dtype=np.float64)
        self.execution.coupling_fingerprint = matrix_fingerprint(self.execution.coupling_matrix)
        return self.execution.coupling_matrix

    @staticmethod
    def _motif_counts(execution: Any) -> tuple[int, int, int, int]:
        files = list(execution.state.files)
        sources = dict(execution.state.sources)
        imports = 0
        functions = 0
        for text in sources.values():
            try:
                tree = pyast.parse(str(text))
            except SyntaxError:
                imports += sum(
                    1 for line in str(text).splitlines() if line.strip().startswith(("import ", "from "))
                )
                functions += sum(
                    1 for line in str(text).splitlines() if line.strip().startswith(("def ", "async def "))
                )
                continue
            for node in pyast.walk(tree):
                if isinstance(node, (pyast.Import, pyast.ImportFrom)):
                    imports += 1
                elif isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef)):
                    functions += 1
        gates = len(execution.gate_results) or len(execution.gate_defs)
        return max(1, len(files)), imports, functions, gates

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
        deforms = _mutating_footprint(args.get("deforms", ()))
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

    def _op_plastic_govern_begin(self, args: dict[str, Any], instruction: Instruction) -> dict[str, Any]:
        config = self._op_govern_begin(args, instruction)["config"]
        kernel = self.vm.kernel
        kernel.plasticity = True
        kernel.governor.enable_plasticity(
            base_eta_min=float(config.get("eta_min", kernel.governor.eta_min)),
            base_eta_high=float(config.get("eta_high", kernel.governor.eta_high)),
            base_tau=float(config.get("tau", kernel.tau)),
            base_chi_max=int(config.get("chi_max", kernel.chi_max)),
            ema_alpha=float(config.get("adaptation_rate", 0.30)),
        )
        kernel.governor.eta_min = float(config.get("eta_min", kernel.governor.eta_min))
        kernel.governor.eta_high = float(config.get("eta_high", kernel.governor.eta_high))
        kernel.chi_max = max(1, int(config.get("chi_max", kernel.chi_max)))
        kernel.tau = float(config.get("tau", kernel.tau))
        thresholds = kernel.governor.modulate(0.0, 0.5)
        self.execution.novelty_score = thresholds.novelty_score
        previous_drops = kernel.governor.drops
        self.execution.predicted_drop = float(previous_drops[-1]) if previous_drops else 0.0
        info = {
            "enabled": True,
            "thresholds": thresholds.as_dict(),
            "novelty_score": thresholds.novelty_score,
            "predicted_drop": self.execution.predicted_drop,
            "surprise_ema": round(kernel.governor.regulator.surprise_ema, 6),
            "analogy": None,
            "probe": None,
        }
        self.execution.plasticity = dict(info)
        if thresholds.novelty_score > 0.65:
            self.execution.events.add("high_novelty")
        self.vm.result.plasticity = dict(info)
        return {"config": config, "plasticity": dict(info)}
