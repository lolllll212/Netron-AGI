from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from early_agi.config import Settings
from early_agi.core.cold_sink import ColdSink
from early_agi.core.entropy_sensor import EntanglementReport
from early_agi.core.governor import CognitiveStepCost, Directive, ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel, KernelDirective
from early_agi.core.plasticity import (
    AnalogicalMatcher,
    HomeostaticRegulator,
    SynapticDecay,
    UserCoAdapter,
)
from early_agi.llm.router import EXECUTOR_SYSTEM, DualTierRouter, Edit, ScriptedRouter, TurnProposal
from early_agi.memory.mera_graph import MeraGraph
from early_agi.memory.reflex import CompiledReflex, ReflexLibrary
from early_agi.operators.dark_state import DarkStateGate, ast_syntax_gate
from early_agi.operators.registry import build_default_registry
from early_agi.session import AgentSession


def area_law() -> EntanglementReport:
    return EntanglementReport(
        spectral_entropy_bits=0.2,
        truncation_error=0.0,
        effective_rank=1,
        is_volume_law=False,
    )


def workspace(root: Path) -> None:
    (root / "calc.py").write_text("def mul(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )


def multiplies(root: Path) -> DarkStateGate:
    return DarkStateGate(
        "multiplies",
        lambda state: "*" in (root / "calc.py").read_text(encoding="utf-8"),
        4.0,
    )


def test_kernel_directive_alias_exposes_epistemic_probe() -> None:
    assert KernelDirective is Directive
    assert KernelDirective.EPISTEMIC_PROBE.value == "EPISTEMIC_PROBE"


def test_homeostatic_regulator_adapts_six_thresholds() -> None:
    regulator = HomeostaticRegulator(
        base_eta_min=0.20,
        base_eta_high=1.20,
        base_tau=0.15,
        base_chi_max=4,
        ema_alpha=0.50,
    )
    regulator.observe_outcome(predicted_drop=1.0, verified_drop=4.0)
    thresholds = regulator.modulate(analogy_similarity=0.0, risk_level=0.8)

    assert thresholds.eta_min < 0.20
    assert thresholds.eta_high < 1.20
    assert thresholds.tau > 0.15
    assert thresholds.chi_max > 4
    assert thresholds.should_probe_first is True
    assert thresholds.novelty_score > 0.65
    assert set(thresholds.as_dict()) == {
        "eta_min",
        "eta_high",
        "tau",
        "chi_max",
        "should_probe_first",
        "novelty_score",
        "exploration_patience",
        "crystallization_bias",
        "prefer_crystallize",
    }
    assert thresholds.exploration_patience > 0
    assert thresholds.prefer_crystallize is False
    assert regulator.snapshot()["surprise_ema"] > 0.0


def test_governor_emits_probe_before_crystallize() -> None:
    governor = ThermodynamicGovernor()
    governor.enable_plasticity(base_eta_min=0.20, base_eta_high=1.20, ema_alpha=0.50)
    governor.modulate(0.0, 0.8, predicted_drop=0.0, verified_drop=2.0)

    eta, directive = governor.evaluate(4.0, CognitiveStepCost(tool_cost=1.0), area_law())

    assert eta == pytest.approx(4.0, rel=1e-5)
    assert directive is Directive.EPISTEMIC_PROBE
    assert governor.probe_pending is False


def test_kernel_tick_probe_locks_writes_and_registry_allows_only_read() -> None:
    with ColdSink() as sink:
        registry = build_default_registry(Path.cwd())
        kernel = EarlyAGIKernel(
            sink,
            gates=[],
            registry=registry,
            chi_max=4,
            tau=0.15,
            plasticity=True,
        )
        tick = kernel.tick(
            {"syntax_ok": True},
            np.eye(2),
            CognitiveStepCost(tool_cost=1.0),
            gate_results={},
            verified_drop=2.0,
            entanglement=area_law(),
            predicted_drop=0.0,
            analogy_similarity=0.0,
            risk_level=0.8,
            commit_gate_state=False,
        )

        assert tick.directive == Directive.EPISTEMIC_PROBE.value
        assert kernel.writes_locked is True
        probe = registry.epistemic_probe(
            [
                ("inspect_deps", {}),
                ("write_file", {"path": "blocked.txt", "content": "no"}),
            ]
        )
        assert probe["results"][0]["ok"] is True
        assert probe["results"][1]["skipped"] == "mutating operator"
        kernel.release_probe_lock()
        assert kernel.writes_locked is False


def test_kernel_probe_completed_releases_the_probe_directive() -> None:
    with ColdSink() as sink:
        kernel = EarlyAGIKernel(sink, gates=[], chi_max=4, tau=0.15, plasticity=True)
        tick = kernel.tick(
            {"syntax_ok": True},
            np.eye(2),
            CognitiveStepCost(tool_cost=1.0),
            gate_results={},
            verified_drop=2.0,
            entanglement=area_law(),
            predicted_drop=2.0,
            analogy_similarity=1.0,
            risk_level=0.5,
            probe_completed=True,
            commit_gate_state=False,
        )
        assert tick.directive == Directive.CRYSTALLIZE.value
        assert kernel.writes_locked is False
        assert kernel.governor.probe_pending is False


def test_analogical_signature_is_scale_invariant() -> None:
    matcher = AnalogicalMatcher()
    first = matcher.encode(
        np.eye(3),
        num_modules=3,
        num_imports=2,
        num_functions=4,
        num_gates=2,
    )
    scaled = matcher.encode(
        np.eye(3) * 7.0,
        num_modules=3,
        num_imports=2,
        num_functions=4,
        num_gates=2,
    )
    assert np.allclose(first.spectrum, scaled.spectrum)
    matcher.register("rule-1", first, "CRYSTALLIZE", {"eta_mean": 2.0})
    match = matcher.best_match(scaled, min_similarity=0.9)
    assert match is not None
    assert match.strategy_hint == "CRYSTALLIZE"
    assert match.metadata["eta_mean"] == 2.0


def test_mera_promotes_persists_and_recalls_analogies(tmp_path: Path) -> None:
    with ColdSink() as sink:
        for index in range(3):
            sink.put(
                "uv_state",
                {
                    "state": {
                        "files": [f"m{index}.py"],
                        "sources": {f"m{index}.py": "import os\ndef f(): pass"},
                    },
                    "gates": {"g1": "pass", "g2": "pass"},
                    "directive": "CRYSTALLIZE",
                    "index": index,
                },
                scale=0,
                eta_cog=2.0,
            )
        db_path = tmp_path / "mera.db"
        graph = MeraGraph(sink, db_path)
        report = graph.coarse_grain(scale=0)
        assert report.analogies >= 1

        signature = graph.encode_signature(
            np.eye(3),
            num_modules=3,
            num_imports=2,
            num_functions=4,
            num_gates=2,
        )
        assert graph.best_analogy(signature, 0.0) is not None
        graph.close()

        restored = MeraGraph(sink, db_path)
        match = restored.best_analogy(signature, 0.0)
        assert match is not None
        assert match.strategy_hint == "CRYSTALLIZE"

        library = ReflexLibrary(restored)
        hint = library.analogical_hint(
            np.eye(3) * 2.0,
            num_modules=3,
            num_imports=2,
            num_functions=4,
            num_gates=2,
            min_similarity=0.0,
        )
        assert hint is not None
        assert hint.strategy_hint == "CRYSTALLIZE"
        restored.close()


def test_synaptic_decay_fades_confidence_and_fast_weight() -> None:
    decay = SynapticDecay()
    fresh = decay.confidence(support=2, hits=2, misses=0, age_steps=0)
    decay.observe(ok=False, surprise=2.0)
    decay.observe(ok=False, surprise=2.0)
    stale = decay.confidence(support=2, hits=2, misses=2, age_steps=8)

    assert stale < fresh
    assert decay.surprise_ema > 0.0
    assert decay.fast_weight < 1.0
    assert decay.snapshot()["fast_weight"] < 1.0


def test_reflex_library_confidence_gate_and_decay_archive() -> None:
    with ColdSink() as sink:
        graph = MeraGraph(sink)
        library = ReflexLibrary(graph)
        base = CompiledReflex(
            hash="reflex-low",
            name="low-confidence",
            edits=(),
            expected_gates=(),
            pattern=(),
            support=3,
            confidence=0.10,
        )
        with patch.object(library, "reflexes", return_value=[base]):
            assert library.match(Path.cwd()) is None
        live = replace(base, confidence=0.90, hash="reflex-live", name="live")
        with patch.object(library, "reflexes", return_value=[live]):
            assert library.match(Path.cwd()) is live

        library.record(live, ok=False, eta_cog=2.0)
        assert library.decay.fast_weight < 1.0
        payload = sink.records(kinds=("reflex_outcome",))[-1].payload
        assert payload["fast_weight"] < 1.0
        graph.close()


def test_user_co_adapter_replays_feedback_and_updates_routers() -> None:
    adapter = UserCoAdapter()
    adapter.record_turn(
        prompt_text="fix",
        edits_accepted=False,
        was_reverted=True,
        rejected_edits=1,
    )
    adapter.record_turn(
        prompt_text="now",
        edits_accepted=False,
        user_overrode=True,
    )
    profile = adapter.profile()
    assert profile.patch_granularity == "surgical"
    assert profile.probe_before_edit is True
    assert profile.rationale_style == "terse"
    assert profile.accept_rate == 0.0

    with ColdSink() as sink:
        sink.put(
            "user_feedback",
            {
                "prompt_text": "fix",
                "edits_accepted": False,
                "was_reverted": True,
                "user_overrode": True,
                "rejected_edits": 1,
            },
        )
        replayed = UserCoAdapter()
        replayed.replay(sink.records(kinds=("user_feedback",)))
        assert replayed.profile().accept_rate == 0.0
        assert replayed.profile().rationale_style == "terse"

    scripted = ScriptedRouter()
    scripted.set_user_profile(profile)
    assert scripted.user_profile is profile

    dual = DualTierRouter(object(), object())
    dual.set_user_profile(profile)
    messages = dual._messages(EXECUTOR_SYSTEM, {})
    assert "surgical" in messages[0]["content"]
    assert "true" in messages[0]["content"]
    assert "terse" in messages[0]["content"]


def test_agent_session_archives_feedback_and_probes_before_edits(tmp_path: Path) -> None:
    workspace(tmp_path)
    with ColdSink() as sink:
        registry = build_default_registry(tmp_path)
        kernel = EarlyAGIKernel(
            sink,
            gates=[ast_syntax_gate(bits=2.0), multiplies(tmp_path)],
            registry=registry,
            chi_max=2,
            plasticity=True,
        )
        mera = MeraGraph(sink, backend=kernel.backend)
        router = ScriptedRouter(
            [
                TurnProposal(
                    edits=(Edit("calc.py", "def mul(a, b):\n    return a * b\n"),),
                    rejected=({"reason": "unsafe path"},),
                    rationale="multiply, do not add",
                )
            ]
        )
        session = AgentSession(
            tmp_path,
            kernel,
            router,
            mera=mera,
            plasticity=True,
        )
        outcome = session.turn("fix", user_overrode=True)

        assert outcome.directive != Directive.EPISTEMIC_PROBE.value
        assert outcome.user_profile is not None
        assert outcome.probe is not None
        assert outcome.plasticity is not None
        assert router.user_profile is not None
        assert sink.count("user_feedback") >= 1
        payload = sink.records(kinds=("user_feedback",))[-1].payload
        assert payload["user_overrode"] is True
        assert payload["rejected_edits"] >= 1
        assert "accept_rate" in payload
        mera.close()


def test_settings_still_defaults_offline() -> None:
    assert Settings.from_env({}, load_dotenv=False).llm_configured is False
