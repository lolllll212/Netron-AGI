from __future__ import annotations

import pytest

from early_agi.core.entropy_sensor import EntanglementReport
from early_agi.core.governor import CognitiveStepCost, Directive, ThermodynamicGovernor


def ent(volume: bool = False) -> EntanglementReport:
    return EntanglementReport(
        spectral_entropy_bits=2.0 if volume else 0.2,
        truncation_error=0.9 if volume else 0.0,
        effective_rank=4 if volume else 1,
        is_volume_law=volume,
    )


def test_high_yield_area_law_crystallises() -> None:
    governor = ThermodynamicGovernor()
    eta, directive = governor.evaluate(4.0, CognitiveStepCost(tool_cost=0.5), ent())
    assert eta == pytest.approx(8.0, rel=1e-5)  # 4 / (0.5 + 1e-6) guards against /0
    assert directive is Directive.CRYSTALLIZE


def test_volume_law_shifts_even_with_high_eta() -> None:
    governor = ThermodynamicGovernor()
    _, directive = governor.evaluate(100.0, CognitiveStepCost(tool_cost=0.1), ent(volume=True))
    assert directive is Directive.SHIFT_COORDINATE


def test_dragging_eta_shifts() -> None:
    governor = ThermodynamicGovernor()
    eta, directive = governor.evaluate(1.0, CognitiveStepCost(tool_cost=2.0), ent())
    assert eta < governor.eta_high
    assert directive is Directive.SHIFT_COORDINATE


def test_stall_recalls_only_after_two_non_positive_ticks() -> None:
    governor = ThermodynamicGovernor()
    _, first = governor.evaluate(0.0, CognitiveStepCost(tool_cost=1.0), ent())
    _, second = governor.evaluate(-4.0, CognitiveStepCost(tool_cost=1.0), ent())
    assert first is Directive.SHIFT_COORDINATE
    assert second is Directive.RECALL_SINK
    assert governor.recall_streak == 2


def test_positive_tick_resets_the_stall_counter() -> None:
    governor = ThermodynamicGovernor()
    governor.evaluate(0.0, CognitiveStepCost(tool_cost=1.0), ent())
    governor.evaluate(4.0, CognitiveStepCost(tool_cost=0.5), ent())
    _, directive = governor.evaluate(0.0, CognitiveStepCost(tool_cost=1.0), ent())
    assert directive is Directive.SHIFT_COORDINATE


def test_two_consecutive_shifts_trigger_synthesis() -> None:
    governor = ThermodynamicGovernor()
    governor.evaluate(0.5, CognitiveStepCost(tool_cost=1.0), ent())
    governor.evaluate(0.5, CognitiveStepCost(tool_cost=1.0), ent())
    assert governor.synthesis_due is True
