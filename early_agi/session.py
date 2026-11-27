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
leaves the workspace or is credited as a durable entropy drop. The mixins in
``early_agi.session_parts`` are implementation details; this module keeps the
public ``AgentSession`` and ``TurnOutcome`` identities.
"""

from __future__ import annotations

from .session_parts import archives as _archives
from .session_parts import learning as _learning
from .session_parts import loop as _loop
from .session_parts import models as _models
from .session_parts import workspace as _workspace
from .session_parts.archives import _ArchiveMethods
from .session_parts.learning import _LearningMethods
from .session_parts.loop import _TurnLoop
from .session_parts.models import TurnOutcome
from .session_parts.workspace import _WorkspaceMethods

__all__ = ["AgentSession", "TurnOutcome"]

TurnOutcome.__module__ = __name__

# Compatibility namespace: the original monolith imported these names here.
Constraint = _loop.Constraint
CouplingMatrix = _loop.CouplingMatrix
RepoGraph = _loop.RepoGraph
RepoScanner = _loop.RepoScanner
CognitiveStepCost = _loop.CognitiveStepCost
Directive = _loop.Directive
EarlyAGIKernel = _loop.EarlyAGIKernel
ShiftArtifact = _loop.ShiftArtifact
TickResult = _archives.TickResult
UserCoAdapter = _loop.UserCoAdapter
MeraGraph = _loop.MeraGraph
ReflexLibrary = _loop.ReflexLibrary
CompiledReflex = _loop.CompiledReflex
ReflexCompileReport = _learning.ReflexCompileReport
Synthesizer = _learning.Synthesizer
Edit = _loop.Edit
Router = _loop.Router
TurnProposal = _loop.TurnProposal
is_safe_relative = _workspace.is_safe_relative
WriteLocked = _workspace.WriteLocked
content_sha = _archives.content_sha
replace = _loop.replace
Path = _loop.Path
Any = _loop.Any
Mapping = _loop.Mapping
Sequence = _loop.Sequence
dataclass = _models.dataclass


class AgentSession(_TurnLoop, _LearningMethods, _ArchiveMethods, _WorkspaceMethods):
    """Transactional, backwards-compatible composition of the session parts."""
