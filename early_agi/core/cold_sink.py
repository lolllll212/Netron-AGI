"""Lossless, content-addressed cold sink (Maxwell's demon with exact undo).

Every state the kernel touches — including every branch it prunes — is committed
here *before* it can influence anything else. Nothing is ever overwritten: the
store is append-only, enforced by SQLite triggers, so "deleting" a branch is
physically impossible and RECALL_SINK is always an exact restore.

Records are keyed by ``sha256(scale \\x1f kind \\x1f canonical_json)``, which makes
the store idempotent (the same state at the same scale is stored once) and
verifiable (reads re-hash the payload and refuse to return corrupt rows).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = ["ColdSink", "SinkError", "SinkRecord", "canonical_json"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS states (
    hash        TEXT PRIMARY KEY,
    parent_hash TEXT,
    scale       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     BLOB NOT NULL,
    eta_cog     REAL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kind_scale ON states(kind, scale);
CREATE INDEX IF NOT EXISTS idx_parent ON states(parent_hash);
CREATE INDEX IF NOT EXISTS idx_created ON states(created_at);

CREATE TRIGGER IF NOT EXISTS states_no_update
BEFORE UPDATE ON states
BEGIN
    SELECT RAISE(ABORT, 'cold sink is append-only: updates are forbidden');
END;

CREATE TRIGGER IF NOT EXISTS states_no_delete
BEFORE DELETE ON states
BEGIN
    SELECT RAISE(ABORT, 'cold sink is append-only: deletes are forbidden');
END;
"""


class SinkError(RuntimeError):
    """Raised when a record cannot be stored, decoded or verified."""


def _json_default(value: Any) -> Any:
    """Best-effort lossless encoding for common non-JSON payload values."""
    if hasattr(value, "tolist"):  # numpy arrays/scalars, torch tensors
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, bytes):
        return value.decode("utf-8", "backslashreplace")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and hasattr(value, "name"):  # enum
        return value.value
    return repr(value)


def canonical_json(payload: Mapping[str, Any] | Any) -> str:
    """Deterministic JSON used both for hashing and for storage."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _digest(scale: int, kind: str, raw: str) -> str:
    return hashlib.sha256(f"{scale}\x1f{kind}\x1f{raw}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SinkRecord:
    hash: str
    parent_hash: str | None
    scale: int
    kind: str
    payload: dict[str, Any]
    eta_cog: float | None
    created_at: float

    @property
    def short(self) -> str:
        return self.hash[:10]

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "scale": self.scale,
            "kind": self.kind,
            "payload": self.payload,
            "eta_cog": self.eta_cog,
            "created_at": self.created_at,
        }


class ColdSink:
    """Append-only SQLite store. Safe to share between threads."""

    def __init__(self, db_path: str | Path = ":memory:", *, compress: bool = True) -> None:
        self.db_path = str(db_path)
        self._compress = compress
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None, timeout=30.0
            )
            with self._lock:
                if self.db_path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=FULL;")
                self._conn.executescript(_SCHEMA)
        except sqlite3.DatabaseError as exc:  # pragma: no cover - environmental
            raise SinkError(f"cannot open cold sink at {self.db_path!r}: {exc}") from exc

    # -- write path ---------------------------------------------------------
    def put(
        self,
        kind: str,
        payload: Mapping[str, Any] | Any,
        scale: int = 0,
        parent_hash: str | None = None,
        eta_cog: float | None = None,
    ) -> str:
        """Commit a state and return its content hash (idempotent)."""
        if not kind:
            raise SinkError("kind is required")
        if not isinstance(scale, int) or scale < 0:
            raise SinkError(f"scale must be a non-negative int, got {scale!r}")
        raw = canonical_json(payload)
        digest = _digest(scale, kind, raw)
        blob = zlib.compress(raw.encode("utf-8"), 6) if self._compress else raw.encode("utf-8")
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO states
                    (hash, parent_hash, scale, kind, payload, eta_cog, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (digest, parent_hash, scale, kind, blob, eta_cog, time.time()),
                )
            except sqlite3.DatabaseError as exc:
                raise SinkError(f"cannot commit {kind!r}: {exc}") from exc
        return digest

    # -- read path ----------------------------------------------------------
    def get(self, digest: str, *, verify: bool = True) -> SinkRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at "
                "FROM states WHERE hash = ?",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        record = self._decode(row)
        if verify:
            raw = canonical_json(record.payload)
            if _digest(record.scale, record.kind, raw) != digest:
                raise SinkError(f"payload of {digest[:10]} failed hash verification")
        return record

    def recall(
        self,
        kind: str = "pruned_branch",
        limit: int = 5,
        *,
        min_eta: float | None = None,
        scale: int | None = None,
    ) -> list[SinkRecord]:
        """Most recent records of a kind, newest first."""
        sql = "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at FROM states WHERE kind = ?"
        params: list[Any] = [kind]
        if min_eta is not None:
            sql += " AND eta_cog IS NOT NULL AND eta_cog >= ?"
            params.append(min_eta)
        if scale is not None:
            sql += " AND scale = ?"
            params.append(scale)
        sql += " ORDER BY created_at DESC, hash DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    def records(
        self,
        kinds: Sequence[str] | None = None,
        *,
        scale: int | None = None,
        min_eta: float | None = None,
        limit: int | None = None,
        oldest_first: bool = True,
    ) -> list[SinkRecord]:
        sql = "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at FROM states WHERE 1=1"
        params: list[Any] = []
        if kinds:
            sql += f" AND kind IN ({','.join('?' for _ in kinds)})"
            params.extend(kinds)
        if scale is not None:
            sql += " AND scale = ?"
            params.append(scale)
        if min_eta is not None:
            sql += " AND eta_cog IS NOT NULL AND eta_cog >= ?"
            params.append(min_eta)
        sql += " ORDER BY created_at ASC, hash ASC" if oldest_first else " ORDER BY created_at DESC, hash DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    def lineage(self, digest: str, *, limit: int = 256) -> list[SinkRecord]:
        """Walk ``parent_hash`` back to the root; returned oldest-first."""
        chain: list[SinkRecord] = []
        seen: set[str] = set()
        cursor: str | None = digest
        while cursor and cursor not in seen and len(chain) < limit:
            seen.add(cursor)
            record = self.get(cursor, verify=False)
            if record is None:
                break
            chain.append(record)
            cursor = record.parent_hash
        chain.reverse()
        return chain

    def children(self, digest: str) -> list[SinkRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at "
                "FROM states WHERE parent_hash = ? ORDER BY created_at ASC",
                (digest,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    # -- introspection ------------------------------------------------------
    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind is None:
                row = self._conn.execute("SELECT COUNT(*) FROM states").fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM states WHERE kind = ?", (kind,)).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            kinds = self._conn.execute(
                "SELECT kind, COUNT(*), MIN(scale), MAX(scale) FROM states GROUP BY kind ORDER BY kind"
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM states").fetchone()
        return {
            "db_path": self.db_path,
            "records": int(total[0]),
            "bytes": int(total[1]),
            "kinds": {
                str(kind): {"count": int(n), "min_scale": int(lo), "max_scale": int(hi)}
                for kind, n, lo, hi in kinds
            },
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ColdSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------
    def _decode(self, row: Iterable[Any]) -> SinkRecord:
        digest, parent, scale, kind, blob, eta, created = row
        try:
            raw = zlib.decompress(bytes(blob)) if self._compress else bytes(blob)
            payload = json.loads(raw.decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SinkError(f"record {str(digest)[:10]} is unreadable: {exc}") from exc
        return SinkRecord(
            hash=str(digest),
            parent_hash=str(parent) if parent else None,
            scale=int(scale),
            kind=str(kind),
            payload=payload,
            eta_cog=float(eta) if eta is not None else None,
            created_at=float(created),
        )
