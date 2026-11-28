"""Gauge Step 4: replay-gated operator synthesis and hot-chain fusion.

This is the public compatibility façade. Models, construction/promotion, replay
benchmarking, and hot-chain fusion live in ``gauge.synthesis_parts``; the
composed class and the historical module-level monkeypatch seam stay here.
"""

from __future__ import annotations

from .synthesis_parts import core as _core
from .synthesis_parts import fusion as _fusion
from .synthesis_parts import models as _models
from .synthesis_parts import replay as _replay
from .synthesis_parts.core import _SynthesisCore
from .synthesis_parts.fusion import _FusionMixin
from .synthesis_parts.models import ReplayCase, SynthesisResult
from .synthesis_parts.replay import _ReplayMixin

__all__ = ["ReplayCase", "SynthesisResult", "Synthesizer"]

ReplayCase.__module__ = __name__
SynthesisResult.__module__ = __name__

# Compatibility namespace and live monkeypatch seam. ``subprocess`` is the same
# stdlib module object used by the replay mixin, so patching ``...run`` here
# still blocks or replaces every replay sandbox call.
copy = _replay.copy
hashlib = _fusion.hashlib
json = _replay.json
math = _replay.math
re = _core.re
subprocess = _replay.subprocess
sys = _replay.sys
time = _replay.time
Mapping = _replay.Mapping
Sequence = _replay.Sequence
Any = _replay.Any
dataclass = _models.dataclass
field = _models.field
stratified_episodes = _replay.stratified_episodes
canonical_json = _fusion.canonical_json
ShiftArtifact = _models.ShiftArtifact
MeraGraph = _core.MeraGraph
ReflexLibrary = _core.ReflexLibrary
DarkStateGate = _core.DarkStateGate
OperatorRegistry = _core.OperatorRegistry
OperatorSpec = _replay.OperatorSpec
ColdSink = _core.ColdSink
_GAIN_KEY = _models._GAIN_KEY
_SAFE_BUILTINS = _models._SAFE_BUILTINS
_SANDBOX_WORKER = _models._SANDBOX_WORKER


class Synthesizer(_SynthesisCore, _ReplayMixin, _FusionMixin):
    """Replay, benchmark, promote, and fuse Gauge operators."""
