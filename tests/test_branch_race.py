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


@pytest.mark.parametrize("git_repo", [False, True])
@pytest.mark.parametrize("all_fail", [False, True])
def test_spawned_mock_transport_sandbox_teardown_and_source_immutability(tmp_path, git_repo, all_fail):
    import subprocess

    fixture = builtin_fixtures()[0]
    root = tmp_path / "source"
    write_workspace(fixture, root)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout

    if git_repo:
        git("init")
        git("add", ".")
        git("-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "fixture")
    before = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    sandboxes = tmp_path / "sandboxes"
    config = RaceConfig(
        fixture=RaceFixtureSpec.from_fixture(fixture),
        branches=("a", "b", "c"),
        executor="process",
        router_name="mocked_live",
        endpoint="mock://local",
        model="test-model",
        max_tokens_per_branch=1 if all_fail else 4096,
        mera_path=str(tmp_path / "mera.db"),
    )
    result = run_branch_race(
        source_root=root, config=config, sink_path=tmp_path / "sink.db", sandbox_parent=sandboxes
    )
    assert result.executor == "process"  # A thread fallback must not hide pickling failures.
    assert result.ok is not all_fail
    assert set(result.sandbox_modes) == {"git_worktree" if git_repo else "temp_copy"}
    assert result.as_dict()["network_calls"] == 0
    assert len(result.pruned_hashes) == (3 if all_fail else 2)
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert not list(sandboxes.iterdir())
    if git_repo:
        assert git("status", "--porcelain") == ""
        assert git("worktree", "list", "--porcelain").count("worktree ") == 1
    with ColdSink(tmp_path / "sink.db") as sink:
        assert sink.count("race_winner") == (0 if all_fail else 1)
