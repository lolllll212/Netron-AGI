from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from early_agi.core.cold_sink import ColdSink


def test_roundtrip_and_dedupe() -> None:
    with ColdSink() as sink:
        first = sink.put("uv_state", {"b": 1, "a": [1, 2]}, eta_cog=2.0)
        second = sink.put("uv_state", {"a": [1, 2], "b": 1}, eta_cog=9.0)
        assert first == second
        record = sink.get(first)
        assert record is not None
        assert record.payload == {"a": [1, 2], "b": 1}
        assert record.eta_cog == 2.0  # first commit wins; nothing is overwritten


def test_scale_and_kind_are_part_of_the_key() -> None:
    with ColdSink() as sink:
        keys = {
            sink.put("uv_state", {"x": 1}, scale=0),
            sink.put("uv_state", {"x": 1}, scale=1),
            sink.put("rule", {"x": 1}, scale=0),
        }
        assert len(keys) == 3


def test_append_only_is_enforced_by_the_database() -> None:
    with ColdSink() as sink:
        sink.put("uv_state", {"x": 1})
        with pytest.raises(sqlite3.IntegrityError):
            sink._conn.execute("UPDATE states SET kind = 'tampered'")  # noqa: SLF001
        with pytest.raises(sqlite3.IntegrityError):
            sink._conn.execute("DELETE FROM states")  # noqa: SLF001
        assert sink.count() == 1


def test_recall_lineage_and_min_eta() -> None:
    with ColdSink() as sink:
        root = sink.put("uv_state", {"n": 0})
        branch = sink.put("pruned_branch", {"n": 1}, parent_hash=root, eta_cog=0.5)
        leaf = sink.put("uv_state", {"n": 2}, parent_hash=branch, eta_cog=1.0)

        assert [record.payload["n"] for record in sink.lineage(leaf)] == [0, 1, 2]
        assert sink.recall("pruned_branch")[0].hash == branch
        assert sink.recall("pruned_branch", min_eta=0.6) == []
        assert sink.children(root)[0].hash == branch


def test_numpy_payloads_are_archived_exactly() -> None:
    with ColdSink() as sink:
        digest = sink.put("uv_state", {"m": np.eye(2), "d": np.float64(0.5)})
        record = sink.get(digest)
        assert record is not None
        assert record.payload["m"] == [[1.0, 0.0], [0.0, 1.0]]
        assert record.payload["d"] == 0.5
        stats = sink.stats()
        assert stats["records"] == 1
        assert stats["kinds"]["uv_state"]["count"] == 1
