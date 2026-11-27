"""Deterministic execution core: cold sink, sensor, governor, kernel, adaptation.

Runtime plasticity, offline adaptability, structural signatures, and surprise
tracking all share :mod:`early_agi.core.adaptation`. The historical
``core.plasticity`` / ``core.adaptability`` modules remain compatibility façades.
"""

from __future__ import annotations

from .plasticity import (
    AdaptiveThresholds,
    AnalogicalMatcher,
    AnalogyMatch,
    AutonomyProfile,
    HomeostaticRegulator,
    StructuralSignature,
    SynapticDecay,
    UserCoAdapter,
)

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
