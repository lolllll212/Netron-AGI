"""Scoped WAL writes and optimistic single-winner leases."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from early_agi.core.cold_sink import ColdSink, SinkError


def test_concurrent_scoped_writes_preserve_content_hashes(tmp_path):
    path = tmp_path / "sink.db"
    with ColdSink(path) as sink:
        original = sink.put("state", {"value": 1})
        assert sink.schema_version() == 3
        assert sink._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def write(index):
        with ColdSink(path) as sink:
            return sink.put("state", {"value": 1}, scope={"run_id": "race", "branch_id": str(index)})

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert set(pool.map(write, range(12))) == {original}
    with ColdSink(path) as sink:
        assert sink.count() == 1
        assert len(sink.commits(original)) == 13
        assert len(sink.recall("state", scope={"branch_id": "3"})) == 1
        assert sink.recall("state", scope={"branch_id": "absent"}) == []


def test_single_winner_and_expired_lease(tmp_path):
    with ColdSink(tmp_path / "sink.db") as sink:
        leases = [sink.open_branch_lease(run_id="r", parent_hash=None, branch_id=b) for b in ("a", "b")]
        digest = sink.put("race_winner", {"ok": True})
        sink.claim_winner(leases[0], winner_hash=digest)
        with pytest.raises(SinkError, match="winner"):
            sink.claim_winner(leases[1], winner_hash=digest)
        assert sink.prune_losers("r", None, winner_lease_id=leases[0]) == 1
        assert {row["status"] for row in sink.leases(run_id="r")} == {"COMMITTED", "PRUNED"}
        sink.open_branch_lease(run_id="other", parent_hash=None, branch_id="a")
        assert sink.expire_leases(now=10**12) == 1
