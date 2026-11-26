"""AgentSession: a transactional turn loop over a real repository.

Per turn:

1. scan the repo, build the bipartite incidence matrix, measure entanglement;
2. ask the fast tier for edits (or the deep tier for a reframing, if writes are
   isolated by a previous SHIFT_COORDINATE);
3. snapshot the touched files, apply an allowlisted edit batch through the
   operator registry, and roll the whole batch back if any write fails;
4. run the dark-state gates once on the *resulting workspace* and tick the kernel;
5. on RECALL_SINK or a failed reflex, restore the snapshot and re-align gate state.

The compiler/kernel can evaluate a candidate branch, but no unverified branch
leaves the workspace or is credited as a durable entropy drop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .core.coupling import Constraint, CouplingMatrix, RepoGraph, RepoScanner
from .core.governor import CognitiveStepCost, Directive
from .core.kernel import EarlyAGIKernel, ShiftArtifact, TickResult
from .llm.router import Edit, Router, TurnProposal, is_safe_relative
from .memory.mera_graph import MeraGraph
from .memory.reflex import CompiledReflex, ReflexCompileReport, ReflexLibrary, content_sha
from .operators.registry import WriteLocked

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
    cost: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
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
        if self.cost is not None:
            out["cost"] = dict(self.cost)
        return out


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
        kernel_checkpoint = self.kernel.checkpoint()
        registry = self.kernel.registry
        registry_checkpoint = registry.snapshot() if registry is not None else None
        if registry is not None:
            registry.consume_turn()

        proposal: TurnProposal | None = None
        artifact: ShiftArtifact | None = None
        reflex: CompiledReflex | None = None
        applied: tuple[str, ...] = ()
        reverted: tuple[str, ...] = ()
        snapshots: dict[str, bytes | None] = {}
        status = "no_change"

        if self.kernel.writes_locked:
            artifact = self.router.reason({**brief, "mode": "reason"})
            status = "reframed"
        else:
            if self.reflexes is not None:
                reflex = self.reflexes.match(self.root, task=task)
            if reflex is not None:
                # A reflex replays a cached *procedure*, not a cached decision:
                # its edits still face the allowlist, write lock, and gates.
                proposal = TurnProposal(
                    edits=tuple(Edit(edit.path, edit.content) for edit in reflex.edits),
                    rationale=f"reflex {reflex.name}: replayed {len(reflex.edits)} learned edit(s), 0 tokens",
                )
            else:
                proposal = self.router.execute({**brief, "mode": "execute"})
            proposal = self._enforce_allowlist(
                proposal, brief["files"] if isinstance(brief.get("files"), list) else []
            )
            if proposal.edits:
                snapshots = self._capture(proposal.paths)
                try:
                    if self.apply_edits:
                        applied = self._write(proposal.edits, snapshots)
                        self._archive_snapshots(snapshots)
                        status = "edited"
                    else:
                        status = "proposed"
                    snapshots_ok = True
                except Exception as exc:
                    snapshots_ok = False
                    write_error = f"{type(exc).__name__}: {exc}"
                    self._restore(snapshots)
                    self.kernel.restore_checkpoint(kernel_checkpoint)
                    if registry is not None and registry_checkpoint is not None:
                        registry.restore_checkpoint(registry_checkpoint)
                    self._archive_failure(task, proposal, write_error)
                    return self._failure_outcome(coupling, proposal, snapshots, write_error)
                if not snapshots_ok:  # pragma: no cover - defensive; except returns above
                    raise RuntimeError("unreachable snapshot state")

        state = self._state(task, graph, coupling)
        cost = CognitiveStepCost(
            discarded_tokens=(proposal.tokens if proposal else 0) + (artifact.tokens if artifact else 0),
            tool_cost=float(len(applied)) + (1.0 if artifact is not None else 0.0),
            pruned_branches=len(proposal.rejected) if proposal else 0,
            latency_ms=(proposal.latency_ms if proposal else 0.0)
            + (artifact.latency_ms if artifact else 0.0),
        )
        gate_results, verified_drop = self.kernel.evaluate_gates(state)
        tick = self.kernel.tick(
            state,
            coupling.matrix,
            cost,
            pruned_tails=list(proposal.rejected) if proposal else [],
            shift_artifact=artifact,
            label=status,
            gate_results=gate_results,
            verified_drop=verified_drop,
            commit_gate_state=False,
        )

        reverted_this_turn = False
        if reflex is not None:
            # A reflex must reproduce the verified effect it was compiled from.
            passed_now = {name for name, verdict in tick.gates.items() if verdict == "pass"}
            satisfied = set(reflex.expected_gates).issubset(passed_now)
            if not satisfied and self.revert_on_recall and applied:
                self._restore(snapshots)
                reverted = applied
                status = "reflex_reverted"
                reverted_this_turn = True
            if self.reflexes is not None:
                self.reflexes.record(reflex, ok=satisfied, eta_cog=tick.eta_cog, parent_hash=tick.hash)
        elif tick.directive == Directive.RECALL_SINK.value and self.revert_on_recall and applied:
            self._restore(snapshots)
            reverted = applied
            status = "recalled"
            reverted_this_turn = True

        if reverted_this_turn:
            restored_results = self.kernel.reconcile_gates(state)
            tick = replace(
                tick,
                verified_drop_bits=0.0,
                gates={name: result.verdict for name, result in restored_results.items()},
            )
        else:
            self.kernel.apply_gate_results(gate_results)

        if applied and not reverted:
            # Learn only from edits that survived the gates: a reverted edit is
            # evidence about the branch, not a procedure worth compiling.
            assert proposal is not None
            self._archive_edits(proposal, snapshots, tick, source="reflex" if reflex else "router")

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
            cost=cost.as_dict(),
        )
        self.history.append(outcome)
        return outcome

    # -- brief --------------------------------------------------------------
    def _brief(self, task: str, graph: RepoGraph, coupling: CouplingMatrix) -> dict[str, Any]:
        report = coupling.report(self.kernel.chi_max, self.kernel.tau)
        files = [info.path for info in graph.modules.values()]
        coupled = [name for name, _ in coupling.top_rows(k=8)]
        sources: dict[str, str] = {}
        binary_files: list[str] = []
        for name in coupled:
            module = name.split("::", 1)[0]
            try:
                raw = (self.root / module).read_bytes()
            except OSError:
                continue
            try:
                sources[module] = raw.decode("utf-8")[: self.max_brief_source_chars]
            except UnicodeDecodeError:
                binary_files.append(module)
                sources[module] = raw.decode("utf-8", errors="replace")[: self.max_brief_source_chars]
        return {
            "task": task,
            "files": files[: self.max_brief_files],
            "binary_files": binary_files,
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

    def _state(self, task: str, graph: RepoGraph, coupling: CouplingMatrix) -> dict[str, Any]:
        report = coupling.report(self.kernel.chi_max, self.kernel.tau)
        return {
            "repo_root": str(self.root),
            "paths": self._scope(graph),
            "task": task,
            "coupling": {
                "shape": list(coupling.matrix.shape),
                "spectral_entropy_bits": round(report.spectral_entropy_bits, 6),
                "truncation_error": round(report.truncation_error, 6),
                "volume_law": report.is_volume_law,
                "fingerprint": coupling.fingerprint()[:16],
            },
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
        )
        self.history.append(outcome)
        return outcome

    # -- workspace helpers --------------------------------------------------
    def _resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes repository root: {relative!r}")
        return target

    def _scope(self, graph: RepoGraph) -> list[str]:
        return [info.path for info in graph.modules.values()]

    def _enforce_allowlist(self, proposal: TurnProposal, files: Sequence[str]) -> TurnProposal:
        """Reject any edit the brief did not declare, regardless of router type."""
        allowed = set(files)
        edits: list[Edit] = []
        rejected: list[Mapping[str, Any]] = list(proposal.rejected)
        for edit in proposal.edits:
            if not is_safe_relative(edit.path):
                rejected.append({"reason": "path is not a safe relative path", "path": edit.path})
                continue
            if edit.path not in allowed:
                rejected.append({"reason": "path is not in the declared edit allowlist", "path": edit.path})
                continue
            edits.append(edit)
        return replace(proposal, edits=tuple(edits), rejected=tuple(rejected))

    def _capture(self, paths: Sequence[str]) -> dict[str, bytes | None]:
        captured: dict[str, bytes | None] = {}
        for path in paths:
            target = self._resolve(path)
            captured[path] = target.read_bytes() if target.exists() else None
        return captured

    def _restore(self, snapshots: Mapping[str, bytes | None]) -> None:
        for path, content in snapshots.items():
            target = self._resolve(path)
            if content is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def _write(self, edits: Sequence[Edit], snapshots: Mapping[str, bytes | None]) -> tuple[str, ...]:
        if self.kernel.writes_locked:
            raise WriteLocked(f"writes are isolated: {self.kernel.lock_reason}")
        written: list[str] = []
        registry = self.kernel.registry
        for edit in edits:
            if registry is not None and "write_file" in registry:
                # The registry is the real write boundary: write locks and CAS
                # guards are enforced here, not merely in the planner.
                registry.apply(
                    "write_file",
                    path=edit.path,
                    content=edit.content,
                    expected=snapshots.get(edit.path),
                )
            else:
                target = self._resolve(edit.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.content, encoding="utf-8")
            written.append(edit.path)
        return tuple(written)
