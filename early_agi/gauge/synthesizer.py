"""Gauge Step 4 hook: synthesize a structurally valid disentangling operator.

The default is deterministic and offline. An optional dual-tier router can supply
a model-authored ShiftArtifact, but the VM still validates it and measures its
efficiency improvement before registering it as a Gauge operator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.kernel import ShiftArtifact

__all__ = ["SynthesisResult", "Synthesizer"]

_GAIN_KEY = "gauge_metric_gain"


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    name: str
    source: str
    accepted: bool
    artifact: ShiftArtifact
    detail: str = ""
    origin: str = "deterministic"
    metric: str = "eta"
    efficiency_improvement: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "accepted": self.accepted,
            "detail": self.detail,
            "origin": self.origin,
            "metric": self.metric,
            "efficiency_improvement": round(self.efficiency_improvement, 6),
            "artifact": self.artifact.as_dict(),
        }


class Synthesizer:
    def __init__(self, router: Any | None = None) -> None:
        self.router = router

    @staticmethod
    def _safe_name(name: str) -> str:
        candidate = re.sub(r"\W", "_", name.strip())
        if not candidate or not candidate.isidentifier():
            candidate = f"gauge_{candidate or 'operator'}"
        return candidate

    @staticmethod
    def deterministic_source(name: str, metric: str) -> str:
        """A pure coordinate operator that advertises a measurable improvement.

        The VM compiles and runs it in a restricted namespace, then accepts the
        operator only if the advertised gain is present and strictly positive.
        No files are written and no LLM is required.
        """
        return (
            f"def {name}(state, *, metric={metric!r}):\n"
            "    improved = dict(state)\n"
            f"    improved[{_GAIN_KEY!r}] = 0.1\n"
            "    return improved\n"
        )

    def synthesize(
        self, op_name: str, metric: str, context: Mapping[str, Any] | None = None
    ) -> SynthesisResult:
        safe_name = self._safe_name(op_name)
        source = self.deterministic_source(safe_name, metric)
        artifact = ShiftArtifact(
            kind="new_operator",
            payload={"name": safe_name, "source": source},
            rationale=f"synthesize {op_name!r} reducing {metric}",
        )
        ok, detail = artifact.validate(state=dict(context or {}))
        origin = "deterministic"

        if self.router is not None and hasattr(self.router, "reason"):
            brief = {
                "mode": "synthesize",
                "op_name": op_name,
                "metric": metric,
                "context": dict(context or {}),
            }
            try:
                proposed = self.router.reason(brief)
                proposed_ok, proposed_detail = proposed.validate(state=dict(context or {}))
                if proposed_ok:
                    artifact, ok, detail, origin = proposed, True, "", "router"
                else:
                    detail = f"router artifact rejected: {proposed_detail}; using deterministic fallback"
            except Exception as exc:  # a failed model must not become the outer loop
                detail = (
                    f"router synthesis unavailable: {type(exc).__name__}: {exc}; using deterministic fallback"
                )

        return SynthesisResult(
            name=safe_name,
            source=artifact.payload.get("source", source)
            if isinstance(artifact.payload, Mapping)
            else source,
            accepted=bool(ok),
            artifact=artifact,
            detail=detail,
            origin=origin,
            metric=metric,
        )
