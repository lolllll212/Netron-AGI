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


def test_race_lineage_preserves_history_and_filters_retired_rules(tmp_path):
    from early_agi.memory.mera_graph import MERGED, SUPERSEDED

    with ColdSink() as sink:
        path = tmp_path / "mera.db"
        graph = MeraGraph(sink, db_path=path)
        old = graph.put_node("rule", {"support": 2, "name": "old"}, scale=1)
        winner = graph.put_node("rule", {"support": 3, "name": "winner"}, scale=1)
        graph.record_race_lineage(winner, [old, old, winner])
        graph.record_race_lineage(winner, [old])
        assert {r["hash"] for r in graph.rules()} == {winner}
        assert {r["hash"] for r in graph.rules(latest_only=False)} == {old, winner}
        for relation in (SUPERSEDED, MERGED):
            assert graph.edges(relation=relation) == [
                {"src": old, "dst": winner, "relation": relation, "weight": 1.0}
            ]
        graph.close()
        reopened = MeraGraph(sink, db_path=path)
        assert {r["hash"] for r in reopened.rules()} == {winner}
        assert len(reopened.rules(latest_only=False)) == 2
        assert len(reopened.edges()) == 2
        reopened.close()


def test_retired_analogy_excluded_by_default_even_after_reopen(tmp_path):
    import numpy as np

    with ColdSink() as sink:
        path = tmp_path / "analogies.db"
        graph = MeraGraph(sink, db_path=path)
        signature = graph.encode_signature(
            np.eye(2), num_modules=2, num_imports=1, num_functions=2, num_gates=1
        )
        old = graph.register_analogy("old-pattern", signature, "module_cut")
        winner = graph.put_node("rule", {"support": 1}, scale=1)
        graph.record_race_lineage(winner, [old])
        assert graph.best_analogy(signature, min_similarity=0.999) is None
        assert (
            graph.best_analogy(signature, min_similarity=0.999, latest_only=False).pattern_id == "old-pattern"
        )
        graph.close()
        graph = MeraGraph(sink, db_path=path)
        assert graph.best_analogy(signature, min_similarity=0.999) is None
        assert (
            graph.best_analogy(signature, min_similarity=0.999, latest_only=False).pattern_id == "old-pattern"
        )
        graph.close()
