"""ColdSink archiving and transactional failure outcome construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from early_agi.core.coupling import CouplingMatrix
from early_agi.core.governor import CognitiveStepCost, Directive
from early_agi.core.kernel import TickResult
from early_agi.llm.router import TurnProposal
from early_agi.memory.reflex import content_sha

from .models import TurnOutcome

__all__ = ["_ArchiveMethods"]


class _ArchiveMethods:
    """Evidence archiving and failure-outcome helpers."""

    def _archive_edits(
        self,
        proposal: TurnProposal,
        snapshots: Mapping[str, bytes | None],
        tick: TickResult,
        *,
        source: str,
    ) -> None:
        for edit in proposal.edits:
            if edit.path not in snapshots:
                continue
            self.kernel.sink.put(
                "applied_edit",
                {
                    "path": edit.path,
                    "content": edit.content,
                    "before_sha": content_sha(snapshots[edit.path]),
                    "after_sha": content_sha(edit.content),
                    "source": source,
                    # The sink is content-addressed and idempotent, so the tick is
                    # part of the identity: the same fix on the same pre-state in a
                    # different turn is a second observation, not a duplicate.
                    "tick": tick.hash,
                },
                scale=self.kernel.scale,
                parent_hash=tick.hash,
                eta_cog=tick.eta_cog,
            )

    def _archive_failure(self, task: str, proposal: TurnProposal, error: str) -> None:
        try:
            self.kernel.sink.put(
                "write_failure",
                {
                    "task": task,
                    "paths": list(proposal.paths),
                    "error": error[:1000],
                },
                scale=self.kernel.scale,
                parent_hash=self.kernel.last_hash,
                eta_cog=self.kernel.governor.last_eta,
            )
        except Exception:
            # The workspace/kernel rollback is already authoritative. A sink
            # outage must not turn a contained write failure into a crash.
            return

    def _archive_snapshots(self, snapshots: Mapping[str, bytes | None]) -> None:
        for path, content in snapshots.items():
            self.kernel.sink.put(
                "file_snapshot",
                {"path": path, "content": content},
                scale=self.kernel.scale,
                parent_hash=self.kernel.last_hash,
                eta_cog=self.kernel.governor.last_eta,
            )

    def _failure_outcome(
        self,
        coupling: CouplingMatrix,
        proposal: TurnProposal,
        snapshots: Mapping[str, bytes | None],
        error: str,
        *,
        task: str = "",
        user_overrode: bool = False,
        plasticity_info: Mapping[str, Any] | None = None,
    ) -> TurnOutcome:
        report = coupling.report(self.kernel.chi_max, self.kernel.tau)
        rejected = (*proposal.rejected, {"reason": "transactional write failed", "detail": error[:500]})
        outcome = TurnOutcome(
            turn=len(self.history) + 1,
            status="write_failed",
            directive=Directive.RECALL_SINK.value,
            eta_cog=0.0,
            verified_drop_bits=0.0,
            spectral_entropy_bits=report.spectral_entropy_bits,
            truncation_error=report.truncation_error,
            volume_law=report.is_volume_law,
            applied=(),
            reverted=tuple(snapshots),
            rationale=f"{proposal.rationale} [rolled back: {error[:200]}]",
            hash="",
            writes_locked=self.kernel.writes_locked,
            rejected=rejected,
            cost=CognitiveStepCost(
                discarded_tokens=proposal.tokens,
                tool_cost=float(len(proposal.edits)),
                pruned_branches=len(rejected),
                latency_ms=proposal.latency_ms,
            ).as_dict(),
            user_profile=(plasticity_info or {}).get("user_profile"),
            analogy=(plasticity_info or {}).get("analogy"),
            plasticity=(plasticity_info or {}).get("plasticity"),
            probe=(plasticity_info or {}).get("probe"),
        )
        outcome = self._finalize_outcome(outcome, task=task, user_overrode=user_overrode)
        self.history.append(outcome)
        return outcome
