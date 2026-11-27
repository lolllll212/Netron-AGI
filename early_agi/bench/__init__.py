"""Offline-first A/B repair benchmark for Netron-AGI adaptability."""

from __future__ import annotations

from .contracts import (
    FileEdit,
    PatchRequest,
    PatchResponse,
    PatchRouter,
    is_safe_relative,
    validated_edits,
)
from .fixtures import BugFixture, PatchCandidate, builtin_fixtures, tangled_fixtures
from .gates import GateReport, GaugeRunReport, evaluate_baseline_gates, load_gauge_program
from .race_benchmark import RaceABBenchmark, run_race_ab_benchmark
from .routers import LiveRouterDisabled, OfflineFixtureRouter, OpenAICompatiblePatchRouter
from .runner import ABBenchmark, run_ab_benchmark

__all__ = [
    "ABBenchmark",
    "BugFixture",
    "FileEdit",
    "GateReport",
    "GaugeRunReport",
    "LiveRouterDisabled",
    "OfflineFixtureRouter",
    "OpenAICompatiblePatchRouter",
    "PatchCandidate",
    "PatchRequest",
    "PatchResponse",
    "PatchRouter",
    "RaceABBenchmark",
    "builtin_fixtures",
    "evaluate_baseline_gates",
    "is_safe_relative",
    "load_gauge_program",
    "run_ab_benchmark",
    "run_race_ab_benchmark",
    "tangled_fixtures",
    "validated_edits",
]
