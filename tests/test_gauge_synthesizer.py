"""Replay-gated operator promotion and deterministic reflex fusion."""

from __future__ import annotations

from early_agi.core.cold_sink import ColdSink
from early_agi.core.kernel import ShiftArtifact
from early_agi.gauge.synthesizer import Synthesizer
from early_agi.memory.mera_graph import MeraGraph
from early_agi.memory.reflex import ReflexLibrary
from early_agi.operators.dark_state import DarkStateGate
from early_agi.operators.registry import OperatorRegistry


class Router:
    def __init__(self, source: str, name: str) -> None:
        self.source = source
        self.name = name
        self.calls: list[dict] = []

    def reason(self, brief: dict) -> ShiftArtifact:
        self.calls.append(brief)
        return ShiftArtifact(
            kind="new_operator",
            payload={"name": self.name, "source": self.source},
            rationale="router-authored candidate",
        )


def _high_drag_sink(count: int = 3) -> ColdSink:
    sink = ColdSink(":memory:")
    for index in range(count):
        sink.put(
            "uv_state",
            {
                "state": {
                    "id": f"episode-{index}",
                    "eta": 0.01,
                    "task": "gauge:replay",
                    "has_regressions": False,
                },
                "eta_cog": 0.02,
                "drag": True,
                "directive": "SHIFT_COORDINATE",
            },
            scale=0,
            eta_cog=0.02,
        )
    return sink


def test_synthesis_replays_last_high_drag_episodes_before_promotion() -> None:
    sink = _high_drag_sink()
    sink.put(
        "uv_state",
        {"state": {"id": "healthy", "eta": 5.0}, "eta_cog": 5.0},
        scale=0,
        eta_cog=5.0,
    )
    registry = OperatorRegistry()
    synthesizer = Synthesizer(sink=sink, registry=registry, replay_limit=3, eta_min=0.15)

    result = synthesizer.synthesize("disentangle_cut", "eta", {"eta": 0.0, "task": "gauge:replay"})

    assert result.accepted is True
    assert result.registered is True
    assert result.replay["cases"] == 3
    assert result.replay["episode_source"] == "cold_sink"
    assert all(not episode["bootstrap"] for episode in result.replay["episodes"])
    assert result.replay["strictly_improved"] is True
    assert result.replay["gates_green"] is True
    assert "disentangle_cut" in registry.names()
    assert registry.apply("disentangle_cut", state={"x": 1})["gauge_metric_gain"] == 0.1
    assert sink.records(kinds=("gauge_synthesis",))


def test_synthesis_rejects_candidate_that_does_not_improve_every_replay() -> None:
    sink = ColdSink(":memory:")
    for episode_id in ("a", "b"):
        sink.put(
            "uv_state",
            {
                "state": {"id": episode_id, "eta": 0.01},
                "eta_cog": 0.01,
                "drag": True,
            },
            scale=0,
            eta_cog=0.01,
        )
    registry = OperatorRegistry()
    source = """def partial_cut(state, *, metric='eta'):
    improved = dict(state)
    if "id" in state and state["id"] == "a":
        improved['gauge_metric_gain'] = 0.1
    else:
        improved['gauge_metric_gain'] = -0.1
    return improved
"""
    router = Router(source, "partial_cut")
    synthesizer = Synthesizer(router=router, sink=sink, registry=registry, replay_limit=2)

    result = synthesizer.synthesize("partial_cut", "eta", {"eta": 0.0})

    assert router.calls
    assert result.origin == "router"
    assert result.accepted is False
    assert result.replay["cases"] == 2
    assert result.replay["strictly_improved"] is False
    assert "partial_cut" not in registry.names()
    assert sink.records(kinds=("gauge_synthesis_rejected",))


def test_synthesis_rejects_red_dark_state_gate() -> None:
    registry = OperatorRegistry()
    sink = ColdSink(":memory:")
    red_gate = DarkStateGate("always_red", lambda state: False, bits=1.0)
    synthesizer = Synthesizer(sink=sink, registry=registry, gates=(red_gate,))

    result = synthesizer.synthesize("blocked_op", "eta", {"eta": 0.0})

    assert result.accepted is False
    assert result.replay["strictly_improved"] is True
    assert result.replay["gates_green"] is False
    assert "always_red" in result.detail
    assert "blocked_op" not in registry.names()


def test_repeated_operator_chains_fuse_into_deterministic_reflexes() -> None:
    sink = ColdSink(":memory:")
    registry = OperatorRegistry()
    graph = MeraGraph(sink, ":memory:")
    reflexes = ReflexLibrary(graph)
    synthesizer = Synthesizer(sink=sink, registry=registry, reflex_library=reflexes)
    chain = ["capture_baseline", "apply_candidate"]
    gates = {"syntax_valid": "pass", "tests_passing": "pass"}

    synthesizer.observe_operator_chain(
        chain,
        ok=True,
        eta=0.6,
        tick_hash="tick-a",
        gates=gates,
        task="gauge:fuse",
    )
    assert synthesizer.compile_hot_chains(min_support=2)["fused"] == 0

    synthesizer.observe_operator_chain(
        chain,
        ok=True,
        eta=0.7,
        tick_hash="tick-b",
        gates=gates,
        task="gauge:fuse",
    )
    report = synthesizer.compile_hot_chains(min_support=2)

    assert report["signatures"] == 1
    assert report["fused"] == 1
    fused_name = report["operators"][0]
    assert fused_name in registry.names()
    fused_result = registry.apply(fused_name, state={"x": 1})
    assert fused_result["fused"] is True
    assert fused_result["chain"] == chain
    loaded = reflexes.operator_reflexes()
    assert len(loaded) == 1
    assert loaded[0].support == 2
    assert loaded[0].chain == tuple(chain)
    matched = reflexes.match_operator_chain(chain, task="gauge:fuse")
    assert matched is not None
    assert matched.fused_operator == fused_name
