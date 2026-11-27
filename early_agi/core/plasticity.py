"""Public compatibility façade for runtime plasticity primitives.

Implementation lives in :mod:`early_agi.core.adaptation`: surprise EMAs,
structural signatures, user co-adaptation, and reflex decay now share one
boundary. Import paths and exported class identities remain unchanged.
"""

from __future__ import annotations

from .adaptation.co_adaptation import AutonomyProfile, UserCoAdapter
from .adaptation.signatures import AnalogicalMatcher, AnalogyMatch, StructuralSignature
from .adaptation.surprise import AdaptiveThresholds, HomeostaticRegulator, SynapticDecay

__all__ = [
    "AdaptiveThresholds",
    "AnalogicalMatcher",
    "AnalogyMatch",
    "AutonomyProfile",
    "HomeostaticRegulator",
    "StructuralSignature",
    "SynapticDecay",
    "UserCoAdapter",
]

for _public_class in (
    AdaptiveThresholds,
    AnalogicalMatcher,
    AnalogyMatch,
    AutonomyProfile,
    HomeostaticRegulator,
    StructuralSignature,
    SynapticDecay,
    UserCoAdapter,
):
    _public_class.__module__ = __name__
