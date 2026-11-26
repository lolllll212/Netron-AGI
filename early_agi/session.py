"""AgentSession: the Phase-1 live loop over a real repository.

Per turn:

1. scan the repo, build the bipartite incidence matrix, measure entanglement;
2. ask the fast tier for edits (or the deep tier for a reframing, if writes are
   isolated by a previous SHIFT_COORDINATE);
3. snapshot the touched files into the cold sink, apply edits;
4. run the dark-state gates on the *resulting workspace* and tick the kernel;
5. on RECALL_SINK, restore the snapshot — exact undo, not "try again".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.coupling import Constraint, CouplingMatrix, RepoGraph, RepoScanner
from .core.governor import CognitiveStepCost, Directive
from .core.kernel import EarlyAGIKernel, TickResult
from .llm.router import Edit, Router, TurnProposal
from .memory.mera_graph import MeraGraph
from .memory.reflex import CompiledReflex, ReflexCompileReport, ReflexLibrary, content_sha

__all__ = ["AgentSession", "TurnOutcome"]


@dataclass(frozen=True)
class TurnOutcome:
    turn: int
    status: str
    directive: str
    eta_cog: float
    verified_drop_bits: float
    spectral_entropy_bits: float
    truncation_error: float
    volume_law: bool
    applied: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    rationale: str = ""
    hash: str = ""
    writes_locked: bool = False
    reflex: str = ""
    rejected: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "status": self.status,
            "directive": self.directive,
            "eta_cog": round(self.eta_cog, 3),
            "verified_drop_bits": round(self.verified_drop_bits, 3),
            "spectral_entropy_bits": round(self.spectral_entropy_bits, 3),
            "truncation_error": round(self.truncation_error, 3),
            "volume_law": self.volume_law,
            "applied": list(self.applied),
            "reverted": list(self.reverted),
            "rationale": self.rationale,
            "hash": self.hash[:10],
            "writes_locked": self.writes_locked,
            "reflex": self.reflex,
            "rejected": list(self.rejected),
        }


class AgentSession:
    def __init__(
        self,
        root: str | Path,
        kernel: EarlyAGIKernel,
        router: Router,
        *,
        scanner: RepoScanner | None = None,
        constraints: Sequence[Constraint] = (),
        apply_edits: bool = True,
        revert_on_recall: bool = True,
        reflexes: ReflexLibrary | None = None,
        max_brief_files: int = 40,
        max_brief_source_chars: int = 4000,
    ) -> None:
        self.root = Path(root).resolve()
        self.kernel = kernel
        self.router = router
        self.scanner = scanner or RepoScanner(self.root)
        self.constraints = tuple(constraints)
        self.apply_edits = apply_edits
        self.revert_on_recall = revert_on_recall
        self.reflexes = reflexes
        self.max_brief_files = max_brief_files
        self.max_brief_source_chars = max_brief_source_chars
        self.history: list[TurnOutcome] = []

    # -- public -------------------------------------------------------------
    def turn(self, task: str) -> TurnOutcome:
        graph = self.scanner.scan()
        coupling = graph.coupling(self.constraints)
        brief = self._brief(task, graph, coupling)

        proposal: TurnProposal | None = None
        artifact = None
        reflex: CompiledReflex | None = None
        applied: tuple[str, ...] = ()
        reverted: tuple[str, ...] = ()
        snapshots: dict[str, str | None] = {}
        status = "no_change"

        if self.kernel.writes_locked:
            artifact = self.router.reason({**brief, "mode": "reason"})
            status = "reframed"
        else:
            if self.reflexes is not None:
                reflex = self.reflexes.match(self.root)
            if reflex is not None:
                # A reflex replays a cached *procedure*, not a cached decision:
                # its edits still face the gates in the tick below.
                proposal = TurnProposal(
                    edits=tuple(Edit(edit.path, edit.content) for edit in reflex.edits),
                    rationale=f"reflex {reflex.name}: replayed {len(reflex.edits)} learned edit(s), 0 tokens",
                )
            else:
                proposal = self.router.execute({**brief, "mode": "execute"})
            if proposal.edits:
                snapshots = self._snapshot(proposal.paths)
                if self.apply_edits:
                    applied = self._write(proposal.edits)
                    status = "edited"
                else:
                    status = "proposed"

        state = {"repo_root": str(self.root), "paths": self._scope(graph)}
        cost = CognitiveStepCost(
            discarded_tokens=proposal.tokens if proposal else 0,
            tool_cost=float(len(applied)),
            pruned_branches=len(proposal.rejected) if proposal else 0,
        )
        pruned_tails = list(proposal.rejected) if proposal else []

        tick = self.kernel.tick(
            state,
            coupling.matrix,
            cost,
            pruned_tails=pruned_tails,
            shift_artifact=artifact,
            label=status,
        )

        if reflex is not None:
            # A reflex must reproduce the verified effect it was compiled from.
            passed_now = {name for name, verdict in tick.gates.items() if verdict == "pass"}
            satisfied = set(reflex.expected_gates).issubset(passed_now)
            if not satisfied and self.revert_on_recall and applied:
                self._restore(snapshots)
                reverted = applied
                status = "reflex_reverted"
            if self.reflexes is not None:
                self.reflexes.record(reflex, ok=satisfied, eta_cog=tick.eta_cog, parent_hash=tick.hash)
        elif tick.directive == Directive.RECALL_SINK.value and self.revert_on_recall and applied:
            self._restore(snapshots)
            reverted = applied

        if applied and not reverted:
            # Learn only from edits that survived the gates: a reverted edit is
            # evidence about the branch, not a procedure worth compiling.
            self._archive_edits(proposal, snapshots, tick, source="reflex" if reflex else "router")

        if self.kernel.registry is not None:
            self.kernel.registry.consume_turn()

        outcome = TurnOutcome(
            turn=len(self.history) + 1,
            status=status,
            directive=tick.directive,
            eta_cog=tick.eta_cog,
            verified_drop_bits=tick.verified_drop_bits,
            spectral_entropy_bits=tick.spectral_entropy_bits,
            truncation_error=tick.truncation_error,
            volume_law=tick.volume_law,
            applied=applied,
            reverted=reverted,
            rationale=proposal.rationale if proposal else getattr(artifact, "rationale", ""),
            hash=tick.hash,
            writes_locked=tick.writes_locked,
            reflex=reflex.name if reflex is not None else "",
            rejected=proposal.rejected if proposal else (),
        )
        self.history.append(outcome)
        return outcome

    # -- brief --------------------------------------------------------------
    def _brief(self, task: str, graph: RepoGraph, coupling: CouplingMatrix) -> dict[str, Any]:
        report = coupling.report(self.kernel.chi_max, self.kernel.tau)
        files = [info.path for info in graph.modules.values()]
        coupled = [name for name, _ in coupling.top_rows(k=8)]
        sources: dict[str, str] = {}
        for name in coupled:
            module = name.split("::", 1)[0]
            try:
                text = (self.root / module).read_text(encoding="utf-8")
            except OSError:
                continue
            sources[module] = text[: self.max_brief_source_chars]
        return {
            "task": task,
            "files": files[: self.max_brief_files],
            "coupled_entities": coupled,
            "sources": sources,
            "constraints": [
                {"id": constraint.id, "kind": constraint.kind, "paths": list(constraint.paths)}
                for constraint in self.constraints
            ],
            "coupling": {
                "shape": list(coupling.matrix.shape),
                "spectral_entropy_bits": round(report.spectral_entropy_bits, 4),
                "truncation_error": round(report.truncation_error, 4),
                "volume_law": report.is_volume_law,
                "fingerprint": coupling.fingerprint()[:16],
            },
            "writes_locked": self.kernel.writes_locked,
            "lock_reason": self.kernel.lock_reason,
            "satisfied_gates": sorted(self.kernel.satisfied_gates),
            "governor": self.kernel.governor.snapshot(),
        }

    # -- learning ------------------------------------------------------------
    def compile_reflexes(self, *, min_support: int = 2) -> ReflexCompileReport:
        """Phase-3 consolidation. Off the hot path: call it between turns."""
        if self.reflexes is None:
            self.reflexes = ReflexLibrary(MeraGraph(self.kernel.sink))
        return self.reflexes.compile(min_support=min_support)

    def _archive_edits(
        self,
        proposal: TurnProposal,
        snapshots: Mapping[str, str | None],
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

    # -- workspace helpers --------------------------------------------------
    def _resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes repository root: {relative!r}")
        return target

    def _scope(self, graph: RepoGraph) -> list[str]:
        return [info.path for info in graph.modules.values()]

    def _snapshot(self, paths: Sequence[str]) -> dict[str, str | None]:
        captured: dict[str, str | None] = {}
        for path in paths:
            target = self._resolve(path)
            content = target.read_text(encoding="utf-8") if target.exists() else None
            captured[path] = content
            self.kernel.sink.put(
                "file_snapshot",
                {"path": path, "content": content},
                scale=self.kernel.scale,
                parent_hash=self.kernel.last_hash,
                eta_cog=self.kernel.governor.last_eta,
            )
        return captured

    def _restore(self, snapshots: Mapping[str, str | None]) -> None:
        for path, content in snapshots.items():
            target = self._resolve(path)
            if content is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

    def _write(self, edits: Sequence[Edit]) -> tuple[str, ...]:
        written: list[str] = []
        for edit in edits:
            target = self._resolve(edit.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.content, encoding="utf-8")
            written.append(edit.path)
        return tuple(written)
