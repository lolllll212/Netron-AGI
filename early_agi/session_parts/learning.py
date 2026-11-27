"""Reflex compilation gate for learned session procedures."""

from __future__ import annotations

from typing import Any

from early_agi.gauge.synthesizer import Synthesizer
from early_agi.memory.reflex import ReflexCompileReport, ReflexLibrary

__all__ = ["_LearningMethods"]


class _LearningMethods:
    """Learning/consolidation methods used by AgentSession."""

    def compile_reflexes(self, *, min_support: int = 2) -> ReflexCompileReport:
        """Phase-3 consolidation, gated by a local sandbox replay."""
        if self.reflexes is None:
            self.reflexes = ReflexLibrary(self.mera)
        synthesizer = Synthesizer(sink=self.kernel.sink, gates=(), replay_limit=3)

        def replay_gate() -> dict[str, Any]:
            name = "reflex_promotion"
            return synthesizer._benchmark(
                synthesizer.deterministic_source(name, "eta"),
                name,
                "eta",
                {"eta": 0.0, "task": "session:reflex-promotion", "has_regressions": False},
            )

        return self.reflexes.compile(min_support=min_support, replay_gate=replay_gate)
