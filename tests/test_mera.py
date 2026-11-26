from __future__ import annotations

from early_agi.core.cold_sink import ColdSink
from early_agi.memory.mera_graph import MeraGraph


def seed(sink: ColdSink) -> None:
    for index in range(3):
        sink.put(
            "uv_state",
            {"i": index, "gates": {"syntax_ok": "pass"}, "directive": "CRYSTALLIZE"},
            eta_cog=2.0,
        )
    for index in range(3):
        sink.put(
            "uv_state",
            {"i": index, "gates": {"tests_pass": "fail"}, "directive": "SHIFT_COORDINATE"},
            eta_cog=-0.5,
        )


def test_coarse_graining_emits_rules_at_the_next_scale() -> None:
    with ColdSink() as sink:
        seed(sink)
        graph = MeraGraph(sink)
        report = graph.coarse_grain(scale=0, chi_max=4)

        assert report.records == 6
        assert report.clusters >= 1
        rules = graph.rules(scale=1)
        assert rules
        assert sum(rule["payload"]["support"] for rule in rules) >= 2
        assert graph.levels()[0]["scale"] == 1
        assert graph.members(rules[0]["hash"])
        assert graph.reflex_candidates(min_support=2)


def test_coarse_graining_needs_more_than_one_trace() -> None:
    with ColdSink() as sink:
        sink.put("uv_state", {"i": 0}, eta_cog=1.0)
        report = MeraGraph(sink).coarse_grain()
        assert report.clusters == 0
        assert "need >= 2" in report.skipped
