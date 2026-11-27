"""Workspace and manifold execution state for the Gauge VM.

This module owns the mutable views that a VM instruction can touch. Keeping them
outside the dispatcher makes partial-tick rollback, ``uncall``, and workspace
snapshots easier to audit in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.coupling import RepoScanner
from ..core.entropy_sensor import EntanglementReport
from ..core.governor import CognitiveStepCost, Directive
from ..core.kernel import TickResult
from ..operators.dark_state import GateResult
from ..operators.registry import OperatorSpec
from . import ast as g
from .builtins import GaugeBuiltins, GaugeGate, SymbolResolver
from .checker import IRREVERSIBLE_CALLS, call_name
from .errors import GaugeRuntimeError
from .expressions import EvaluationContext, evaluate
from .runtime import GaugeState, ManifoldState, ReversibleTransform

__all__ = ["GaugeExecutionState", "WorkspaceView"]


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    """Immutable, JSON-friendly view of the workspace at the current scale."""

    root: Path
    files: tuple[str, ...]
    sources: dict[str, str]
    targets: tuple[str, ...]
    task: str
    scale: int
    run_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "files": list(self.files),
            "targets": list(self.targets),
            "task": self.task,
            "scale": self.scale,
            "run_index": self.run_index,
            "source_count": len(self.sources),
        }


class GaugeExecutionState:
    """All per-run VM state, isolated from bytecode dispatch."""

    def __init__(self, vm: Any, builtins: GaugeBuiltins | None = None) -> None:
        self.vm = vm
        self.builtins = builtins or GaugeBuiltins(vm)
        self.resolver = SymbolResolver()
        self.run_index = 0
        self.reset("<gauge>")

    # -- lifecycle ----------------------------------------------------------
    def reset(self, name: str) -> None:
        self.run_index += 1
        self.program_name = name
        self.state = GaugeState(self.vm.root)
        self.workspace: WorkspaceView | None = None
        self.manifolds: dict[str, ManifoldState] = {}
        self.transforms: dict[str, ReversibleTransform] = {}
        self.variables: dict[str, Any] = {}
        self.inverse_stack: list[tuple[str, str, dict[str, Any]]] = []
        self.gate_defs: dict[str, GaugeGate] = {}
        self.ops: dict[str, OperatorSpec] = {}
        self.op_after: dict[str, tuple[str, ...]] = {}
        self.op_commutes: dict[str, tuple[str, ...]] = {}
        self.applied_ops: set[str] = set()
        self.applied_op_chain: list[str] = []
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
        self.tick: TickResult | None = None
        self.cost = CognitiveStepCost()
        self.collapses: dict[str, str] = {}
        self.recalled: list[dict[str, Any]] = []
        self.synthesis_events: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self.traps: list[str] = []
        self.kernel_checkpoint: dict[str, Any] = {}
        self.gate_state_committed = False
        self.status = "running"
        self.halted = False
        self.analogy: dict[str, Any] | None = None
        self.analogy_similarity = 0.0
        self.predicted_drop = 0.0
        self.probe_results: list[dict[str, Any]] = []
        self.novelty_score = 0.0
        self.distilled_constraints: list[str] = []
        self.plasticity: dict[str, Any] = {}

    def scan_workspace(self) -> WorkspaceView:
        scanner = RepoScanner(self.vm.root, max_files=self.vm.settings.max_files)
        graph = scanner.scan()
        files = tuple(info.path for info in graph.modules.values())
        sources: dict[str, str] = {}
        for relative in files:
            try:
                sources[relative] = (self.vm.root / relative).read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
        targets = tuple(
            path for path in files if Path(path).name.startswith("test_") or path.endswith("_test.py")
        )
        self.state.files = list(files)
        self.state.sources = sources
        self.state.targets = list(targets)
        self.state.task = f"gauge:{self.program_name}"
        self.workspace = self.view()
        return self.workspace

    def view(self) -> WorkspaceView:
        """Return a live scale-aware view without exposing Python objects."""
        view = WorkspaceView(
            root=self.vm.root,
            files=tuple(self.state.files),
            sources=dict(self.state.sources),
            targets=tuple(self.state.targets),
            task=self.state.task,
            scale=self.state.scale,
            run_index=self.run_index,
        )
        self.workspace = view
        return view

    # -- serialisable views -------------------------------------------------
    def plain_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "program": self.program_name,
            "run_index": self.run_index,
            "repo_root": str(self.vm.root),
            "paths": list(self.state.files),
            "files": list(self.state.files),
            "sources": dict(self.state.sources),
            "targets": tuple(self.state.targets),
            "task": self.state.task,
            "scale": self.state.scale,
            "chi": self.vm.kernel.chi_max,
            "eta": self.eta,
            "ent": self.entanglement.spectral_entropy_bits if self.entanglement else 0.0,
            "truncation": self.entanglement.truncation_error if self.entanglement else 0.0,
            "volume_law": self.entanglement.is_volume_law if self.entanglement else False,
            "has_regressions": self.state.has_regressions,
            "coupling_shape": list(self.coupling_matrix.shape),
            "coupling_fingerprint": self.coupling_fingerprint,
            "gates": {name: result.verdict for name, result in self.gate_results.items()},
            "drag": self.drag,
            "directive": self.directive.value,
            "events": sorted(self.events),
            "tick_hash": self.tick.hash if self.tick else None,
            "repo": {"root": str(self.vm.root), "ast": {"modules": list(self.state.files)}},
            "p": {
                "files": list(self.state.files),
                "targets": list(self.state.targets),
                "has_regressions": self.state.has_regressions,
            },
            "analogy": self.analogy,
            "analogy_similarity": self.analogy_similarity,
            "predicted_drop": self.predicted_drop,
            "novelty_score": self.novelty_score,
            "probe_results": list(self.probe_results),
            "distilled_constraints": list(self.distilled_constraints),
            "plasticity": dict(self.plasticity),
        }
        workspace = self.view()
        state["workspace"] = workspace.as_dict()
        for name, manifold in self.manifolds.items():
            state[name] = manifold.as_dict()
        return state

    # Compatibility for embedders that used the pre-decomposition private name.
    def _plain_state(self) -> dict[str, Any]:
        return self.plain_state()

    def evaluation_context(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> EvaluationContext:
        plain = self.plain_state()
        if state:
            plain.update(dict(state))
        primary = next(iter(self.manifolds.values()), None)
        context_variables: dict[str, Any] = {
            "p": self.state.candidate,
            "m": primary,
            "repo": {"root": str(self.vm.root), "ast": {"modules": list(self.state.files)}},
            "scale": self.state.scale,
            "run_index": self.run_index,
            "ent": self.entanglement.spectral_entropy_bits if self.entanglement else 0.0,
            "eta": self.eta,
            "chi": self.vm.kernel.chi_max,
            "sink": {"records": self.vm.sink.count()},
            "similarity": self.analogy_similarity,
            "novelty": self.novelty_score,
            "score": self.novelty_score,
            **self.variables,
        }
        for name, manifold in self.manifolds.items():
            context_variables[name] = manifold
        if variables:
            context_variables.update(dict(variables))
        return EvaluationContext(
            variables=context_variables,
            state=plain,
            functions=self.builtins.functions(),
            resolver=self.resolver,
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
        context.functions.update(self.builtins.transform_functions(manifold))
        return context

    # -- reversible transforms ---------------------------------------------
    def apply_transform(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args["transform"])
        transform = self.transforms.get(name)
        if transform is None:
            raise GaugeRuntimeError(f"unknown reversible transform {name!r}")
        context = self.evaluation_context()
        evaluated_args = [evaluate(item, context) for item in args.get("args", ())]
        kwargs = {key: evaluate(item, context) for key, item in dict(args.get("kwargs", {})).items()}
        manifold = next((item for item in evaluated_args if isinstance(item, ManifoldState)), None)
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

    def execute_transform_statement(self, stmt: g.Statement, context: EvaluationContext) -> None:
        """Run a reversible transform body; destructive operations are refused."""
        if isinstance(
            stmt,
            (
                g.CollapseStmt,
                g.RecallStmt,
                g.SynthesizeStmt,
                g.TrapHandler,
                g.EvolveStmt,
                g.ProbeStmt,
            ),
        ):
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

    def restore_inverse_stack(self) -> int:
        """Restore every invertible frame in LIFO order; return frame count."""
        count = len(self.inverse_stack)
        for _name, manifold_name, snapshot in reversed(self.inverse_stack):
            target = self.manifolds.get(manifold_name)
            if target is not None:
                target.restore(snapshot)
        self.inverse_stack.clear()
        return count

    @staticmethod
    def scale_number(value: Any) -> int:
        if isinstance(value, int):
            return value
        text = str(value)
        if len(text) > 1 and text[0] in {"L", "l"} and text[1:].isdigit():
            return int(text[1:])
        return int(text)
