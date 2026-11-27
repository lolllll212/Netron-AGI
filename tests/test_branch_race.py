"""Offline race integration and validation."""

import pytest

from early_agi.bench.fixtures import builtin_fixtures
from early_agi.bench.gates import write_workspace
from early_agi.core.branch_race import RaceConfig, RaceFixtureSpec, run_branch_race
from early_agi.core.cold_sink import ColdSink


def test_race_commits_green_winner_and_archives_losers(tmp_path):
    fixture = builtin_fixtures()[0]
    root = tmp_path / "workspace"
    write_workspace(fixture, root)
    before = {p: (root / p).read_text() for p in fixture.editable_paths}
    config = RaceConfig(
        fixture=RaceFixtureSpec.from_fixture(fixture),
        branches=("a", "b", "c"),
        executor="thread",
        mera_path=str(tmp_path / "mera.db"),
    )
    with ColdSink(tmp_path / "sink.db") as sink:
        result = run_branch_race(source_root=root, config=config, sink=sink)
        assert result.ok
        assert result.winner is not None
        assert result.winner_hash
        assert len(result.pruned_hashes) == 2
        assert result.lineage_edges == 4
        assert sink.count("race_winner") == 1
        assert sink.count("pruned_branch") == 2
        assert len(sink.leases(status="COMMITTED")) == 1
        assert result.as_dict()["network_calls"] == 0
        assert result.as_dict()["offline"] is True
    assert before == {p: (root / p).read_text() for p in fixture.editable_paths}


@pytest.mark.parametrize("branches", [("a",), ("a", "a"), ("a", "b", "c", "d", "e")])
def test_invalid_branch_sets_rejected(tmp_path, branches):
    config = RaceConfig(fixture=RaceFixtureSpec.from_fixture(builtin_fixtures()[0]), branches=branches)
    with pytest.raises(ValueError):
        run_branch_race(source_root=tmp_path, config=config)
