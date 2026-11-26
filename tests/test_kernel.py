from __future__ import annotations

import numpy as np

from early_agi.core.cold_sink import ColdSink
from early_agi.core.governor import CognitiveStepCost, ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel, validate_shift_artifact
from early_agi.operators.dark_state import DarkStateGate

AREA_LAW = np.outer([1.0, 0.8, 0.2, 0.1], [1.0, 0.5, 0.2, 0.1])
VOLUME_LAW = np.random.default_rng(42).standard_normal((8, 8))


def make_kernel(sink: ColdSink, *, gates: list[DarkStateGate] | None = None, lock: bool = True) -> EarlyAGIKernel:
    gates = gates or [
        DarkStateGate("syntax_ok", lambda state: bool(state.get("syntax_ok")), 2.0),
        DarkStateGate("tests_pass", lambda state: bool(state.get("tests_pass")), 4.0),
    ]
    return EarlyAGIKernel(
        sink,
        governor=ThermodynamicGovernor(),
        gates=gates,
        chi_max=2,
        tau=0.15,
        lock_writes_on_shift=lock,
    )


def test_clean_area_law_insight_crystallises() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        result = kernel.tick(
            {"syntax_ok": True, "tests_pass": False},
            AREA_LAW,
            CognitiveStepCost(discarded_tokens=200, tool_cost=1.0),
        )
        assert result.directive == "CRYSTALLIZE"
        assert result.verified_drop_bits == 2.0
        assert result.volume_law is False
        assert kernel.writes_locked is False


def test_volume_law_shifts_and_isolates_writes() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        result = kernel.tick(
            {"syntax_ok": True, "tests_pass": True},
            VOLUME_LAW,
            CognitiveStepCost(tool_cost=2.0),
        )
        assert result.volume_law is True
        assert result.directive == "SHIFT_COORDINATE"
        assert kernel.writes_locked is True

        held = kernel.tick(
            {"syntax_ok": True, "tests_pass": True},
            VOLUME_LAW,
            CognitiveStepCost(tool_cost=1.0),
        )
        assert held.shift_held is True
        assert held.directive == "SHIFT_COORDINATE"
        assert kernel.writes_locked is True


def test_valid_artifact_releases_the_lock() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        kernel.tick({"syntax_ok": True, "tests_pass": True}, VOLUME_LAW, CognitiveStepCost(tool_cost=2.0))
        assert kernel.writes_locked is True

        result = kernel.tick(
            {"syntax_ok": True, "tests_pass": True},
            VOLUME_LAW,
            CognitiveStepCost(tool_cost=1.0),
            shift_artifact={"kind": "decomposition", "payload": {"steps": ["split", "pin"]}},
        )
        assert kernel.writes_locked is False
        assert result.shift_kind == "decomposition"
        assert sink.count("shift_artifact") == 1


def test_invalid_artifact_keeps_the_lock() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        kernel.tick({"syntax_ok": True, "tests_pass": True}, VOLUME_LAW, CognitiveStepCost(tool_cost=2.0))

        result = kernel.tick(
            {"syntax_ok": True, "tests_pass": True},
            VOLUME_LAW,
            CognitiveStepCost(tool_cost=1.0),
            shift_artifact={
                "kind": "new_operator",
                "payload": {"name": "helper", "source": "def other():\n    pass\n"},
            },
        )
        assert result.shift_held is True
        assert kernel.writes_locked is True
        assert sink.count("shift_rejected") == 1


def test_breaking_a_verified_gate_charges_its_bits_back() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        kernel.tick({"syntax_ok": True, "tests_pass": False}, AREA_LAW, CognitiveStepCost(tool_cost=1.0))
        result = kernel.tick({"syntax_ok": False, "tests_pass": False}, AREA_LAW, CognitiveStepCost(tool_cost=1.0))

        assert result.verified_drop_bits == -2.0
        assert kernel.satisfied_gates == set()


def test_unverifiable_gate_is_neutral() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink, gates=[DarkStateGate("cannot_run", lambda state: None, 3.0)])
        result = kernel.tick({}, AREA_LAW, CognitiveStepCost(tool_cost=1.0))

        assert result.verified_drop_bits == 0.0
        assert result.gates["cannot_run"] == "unverifiable"


def test_pruned_branches_are_archived_before_the_tick() -> None:
    with ColdSink() as sink:
        kernel = make_kernel(sink)
        kernel.tick(
            {"syntax_ok": True, "tests_pass": False},
            AREA_LAW,
            CognitiveStepCost(tool_cost=1.0),
            pruned_tails=[{"hypothesis": "h1"}],
        )
        records = sink.recall("pruned_branch")
        assert records
        assert records[0].payload == {"hypothesis": "h1"}


def test_shift_artifact_validation_rules() -> None:
    assert validate_shift_artifact("decomposition", {"steps": ["one", "two"]})[0]
    assert validate_shift_artifact("invariant", {"check": "1 + 1 == 2"})[0]
    assert not validate_shift_artifact("invariant", {"check": "def broken("})[0]
    assert not validate_shift_artifact("new_operator", {"name": "f", "source": "x = 1\n"})[0]
    assert validate_shift_artifact("new_operator", {"name": "f", "source": "def f():\n    pass\n"})[0]
    assert not validate_shift_artifact("teleport", {})[0]
