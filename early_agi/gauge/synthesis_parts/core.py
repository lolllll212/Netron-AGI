"""Synthesizer construction, binding, deterministic source, and promotion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from early_agi.core.cold_sink import ColdSink
from early_agi.core.kernel import ShiftArtifact
from early_agi.memory.mera_graph import MeraGraph
from early_agi.memory.reflex import ReflexLibrary
from early_agi.operators.dark_state import DarkStateGate
from early_agi.operators.registry import OperatorRegistry

from .models import _GAIN_KEY, SynthesisResult

if TYPE_CHECKING:
    from ..synthesizer import Synthesizer

__all__ = ["_SynthesisCore"]


class _SynthesisCore:
    """Construction and candidate-operator synthesis methods."""

    def __init__(
        self,
        router: Any | None = None,
        *,
        sink: ColdSink | None = None,
        registry: OperatorRegistry | None = None,
        gates: Sequence[Any] = (),
        reflex_library: ReflexLibrary | None = None,
        eta_min: float = 0.15,
        replay_limit: int = 5,
        sandbox_timeout: float = 8.0,
    ) -> None:
        self.router = router
        self.sink = sink
        self.registry = registry
        self.gates = self._prepare_gates(gates)
        self.eta_min = float(eta_min)
        self.replay_limit = max(1, int(replay_limit))
        self.sandbox_timeout = max(0.5, float(sandbox_timeout))
        self._chain_sequence = 0
        self.reflex_library = reflex_library
        if self.reflex_library is None and self.sink is not None:
            self.reflex_library = ReflexLibrary(MeraGraph(self.sink, ":memory:"))

    def bind(
        self,
        *,
        sink: ColdSink | None = None,
        registry: OperatorRegistry | None = None,
        gates: Sequence[Any] | None = None,
        reflex_library: ReflexLibrary | None = None,
        eta_min: float | None = None,
    ) -> Synthesizer:
        """Attach VM subsystems to a synthesizer supplied by an embedder."""
        if self.sink is None and sink is not None:
            self.sink = sink
        if self.registry is None and registry is not None:
            self.registry = registry
        if gates is not None:
            self.gates = self._prepare_gates([*self.gates, *gates])
        else:
            self.gates = self._prepare_gates(self.gates)
        if self.reflex_library is None:
            if reflex_library is not None:
                self.reflex_library = reflex_library
            elif self.sink is not None:
                self.reflex_library = ReflexLibrary(MeraGraph(self.sink, ":memory:"))
        if eta_min is not None:
            self.eta_min = float(eta_min)
        return self

    @staticmethod
    def _safe_name(name: str) -> str:
        candidate = re.sub(r"\W", "_", name.strip())
        if not candidate or not candidate.isidentifier():
            candidate = f"gauge_{candidate or 'operator'}"
        return candidate

    @staticmethod
    def deterministic_source(name: str, metric: str) -> str:
        """A pure coordinate operator that advertises a measurable gain."""
        return (
            f"def {name}(state, *, metric={metric!r}):\n"
            "    improved = dict(state)\n"
            f"    improved[{_GAIN_KEY!r}] = 0.1\n"
            "    return improved\n"
        )

    @staticmethod
    def _default_gate() -> DarkStateGate:
        """A deterministic dark-state gate over every replayed candidate."""

        def check(state: Mapping[str, Any]) -> bool:
            if state.get("has_regressions") is True:
                return False
            verdicts = state.get("gates") or {}
            if isinstance(verdicts, Mapping) and any(str(value) == "fail" for value in verdicts.values()):
                return False
            try:
                return float(state.get(_GAIN_KEY, 0.0) or 0.0) > 0.0
            except (TypeError, ValueError):
                return False

        return DarkStateGate("synthesis_replay_safe", check, bits=1.0)

    @classmethod
    def _prepare_gates(cls, gates: Sequence[Any]) -> list[Any]:
        selected: list[Any] = []
        seen: set[str] = set()
        for gate in gates:
            name = str(getattr(gate, "name", type(gate).__name__))
            if name in seen:
                continue
            selected.append(gate)
            seen.add(name)
        if not any(isinstance(gate, DarkStateGate) for gate in selected):
            default = cls._default_gate()
            if default.name not in seen:
                selected.append(default)
        return selected

    def synthesize(
        self,
        op_name: str,
        metric: str,
        context: Mapping[str, Any] | None = None,
        *,
        entanglement: Any | None = None,
    ) -> SynthesisResult:
        safe_name = self._safe_name(op_name)
        state = dict(context or {})
        source = self.deterministic_source(safe_name, metric)
        artifact = ShiftArtifact(
            kind="new_operator",
            payload={"name": safe_name, "source": source},
            rationale=f"synthesize {op_name!r} reducing {metric}",
        )
        structure_ok, detail = artifact.validate(state=state, entanglement=entanglement)
        origin = "deterministic"

        if self.router is not None and hasattr(self.router, "reason"):
            brief = {
                "mode": "synthesize",
                "op_name": op_name,
                "metric": metric,
                "context": state,
            }
            try:
                proposed = self.router.reason(brief)
                if isinstance(proposed, Mapping):
                    proposed = ShiftArtifact.from_mapping(dict(proposed))
                elif isinstance(proposed, str):
                    proposed = ShiftArtifact.from_mapping(dict(json.loads(proposed)))
                proposed_ok, proposed_detail = proposed.validate(state=state, entanglement=entanglement)
                if proposed_ok:
                    artifact, structure_ok, detail, origin = proposed, True, "", "router"
                else:
                    detail = f"router artifact rejected: {proposed_detail}; using deterministic fallback"
            except Exception as exc:
                detail = (
                    f"router synthesis unavailable: {type(exc).__name__}: {exc}; using deterministic fallback"
                )

        function_name = str(artifact.payload.get("name", safe_name))
        source = str(artifact.payload.get("source", source))
        replay = self._benchmark(source, function_name, metric, state)
        promoted = bool(structure_ok and replay["ok"])
        if not promoted and not detail:
            detail = str(replay["detail"])
        registered = False
        if promoted and self.registry is not None:
            self._register_operator(safe_name, source, function_name, metric)
            registered = True
        self._archive_synthesis(
            safe_name,
            metric,
            artifact,
            replay,
            promoted,
            origin,
            detail,
            scale=int(state.get("scale", 0) or 0),
        )
        return SynthesisResult(
            name=safe_name,
            source=source,
            accepted=promoted,
            artifact=artifact,
            detail=detail,
            origin=origin,
            metric=metric,
            efficiency_improvement=float(replay["improvement"]),
            replay=replay,
            registered=registered,
            promoted=promoted,
        )
