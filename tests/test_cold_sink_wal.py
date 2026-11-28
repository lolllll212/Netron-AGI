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


def _contending_writer(path, index, barrier):
    # Separate connections in spawned processes; synchronize after lease creation
    # so every worker really attempts to claim the same parent.
    with ColdSink(path) as sink:
        lease = sink.open_branch_lease(run_id="process-race", parent_hash="parent", branch_id=str(index))
        barrier.wait(timeout=30)
        for value in range(15):
            digest = sink.put("contended", {"worker": index, "value": value})
        try:
            sink.claim_winner(lease, winner_hash=digest)
            return True
        except SinkError as exc:
            assert "winner" in str(exc)
            return False


def test_four_process_wal_contention_and_single_winner(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    path = str(tmp_path / "process.db")
    with ColdSink(path):
        pass
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(4)
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
            futures = [pool.submit(_contending_writer, path, i, barrier) for i in range(4)]
            assert sum(f.result(timeout=90) for f in futures) == 1
    with ColdSink(path) as sink:
        assert sink.count("contended") == 60
        assert len(sink.leases(status="COMMITTED")) == 1


@pytest.mark.parametrize("compress", [True, False])
def test_existing_v2_database_migrates_in_place(tmp_path, compress):
    import hashlib
    import sqlite3
    import zlib

    path = tmp_path / "v2.db"
    raw = '{"legacy":true}'
    digest = hashlib.sha256(f"0\x1flegacy\x1f{raw}".encode()).hexdigest()
    blob = zlib.compress(raw.encode()) if compress else raw.encode()
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE states (hash TEXT PRIMARY KEY, parent_hash TEXT, scale INTEGER NOT NULL,
                kind TEXT NOT NULL, payload BLOB NOT NULL, eta_cog REAL, created_at REAL NOT NULL);
            CREATE TABLE state_commits (id INTEGER PRIMARY KEY AUTOINCREMENT, hash TEXT NOT NULL,
                parent_hash TEXT, eta_cog REAL, created_at REAL NOT NULL);
            CREATE TABLE sink_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO sink_meta VALUES ('format_version', '2');
        """)
        conn.execute("INSERT INTO sink_meta VALUES ('compression', ?)", ("zlib" if compress else "raw",))
        conn.execute("INSERT INTO states VALUES (?, NULL, 0, 'legacy', ?, 2.0, 1.0)", (digest, blob))
        conn.execute("INSERT INTO state_commits VALUES (1, ?, NULL, 2.0, 1.0)", (digest,))
        rootpage = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='states'").fetchone()[0]
    for _ in range(2):
        with ColdSink(path, compress=compress) as sink:
            assert sink.schema_version() == 3
            assert sink.get(digest).payload == {"legacy": True}
            assert (
                sink._conn.execute("SELECT payload FROM states WHERE hash=?", (digest,)).fetchone()[0] == blob
            )
            assert (
                sink._conn.execute("SELECT rootpage FROM sqlite_master WHERE name='states'").fetchone()[0]
                == rootpage
            )
            assert sink.commits(digest)[0]["id"] == 1
            with pytest.raises(sqlite3.IntegrityError):
                sink._conn.execute("DELETE FROM states")
    with ColdSink(path, compress=compress) as sink:
        assert sink.put("legacy", {"legacy": True}, scope={"run_id": "new"}) == digest
        assert sink.recall("legacy", scope={"run_id": "new"})[0].hash == digest
