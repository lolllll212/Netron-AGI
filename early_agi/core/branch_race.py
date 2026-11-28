"""Compatibility facade for offline branch racing."""

from .race_parts.models import (
    BranchRaceResult,
    BranchResult,
    BranchSpec,
    RaceConfig,
    RaceFixtureSpec,
    RouterConfig,
)
from .race_parts.racer import MockedOpenAICompatibleRouter, race_fixture_spec, run_branch_race

__all__ = [
    "BranchRaceResult",
    "BranchResult",
    "BranchSpec",
    "MockedOpenAICompatibleRouter",
    "RaceConfig",
    "RaceFixtureSpec",
    "RouterConfig",
    "race_fixture_spec",
    "run_branch_race",
]
