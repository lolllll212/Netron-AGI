"""Repository path resolution, edit allowlists, snapshots, and writes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from early_agi.core.coupling import RepoGraph
from early_agi.llm.router import Edit, TurnProposal, is_safe_relative
from early_agi.operators.registry import WriteLocked

__all__ = ["_WorkspaceMethods"]


class _WorkspaceMethods:
    """Workspace and transactional write helpers."""

    def _resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes repository root: {relative!r}")
        return target

    def _scope(self, graph: RepoGraph) -> list[str]:
        return [info.path for info in graph.modules.values()]

    @staticmethod
    def _surgical_proposal(proposal: TurnProposal) -> TurnProposal:
        """After frequent reverts, permit only the smallest verified patch."""
        if len(proposal.edits) <= 1:
            return proposal
        rejected = (*proposal.rejected, {"reason": "autonomy policy: surgical single-file patch"})
        return replace(proposal, edits=proposal.edits[:1], rejected=rejected)

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
