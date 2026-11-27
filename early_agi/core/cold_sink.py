"""Lossless, content-addressed cold sink with scoped WAL branch coordination.

v3 is an additive, no-rebuild migration from v2: ``states`` stays append-only and
content-addressed; nullable scope columns and a mutable ``branch_leases`` table
carry multi-process race coordination. Every write uses ``BEGIN IMMEDIATE`` so
WAL readers never hit SQLITE_BUSY_SNAPSHOT when upgrading to a writer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ColdSink", "SinkError", "SinkRecord", "SinkScope", "canonical_json"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS states (
    hash        TEXT PRIMARY KEY,
    parent_hash TEXT,
    scale       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     BLOB NOT NULL,
    eta_cog     REAL,
    created_at  REAL NOT NULL,
    user_id     TEXT,
    agent_id    TEXT,
    run_id      TEXT,
    branch_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_kind_scale ON states(kind, scale);
CREATE INDEX IF NOT EXISTS idx_parent ON states(parent_hash);
CREATE INDEX IF NOT EXISTS idx_created ON states(created_at);

CREATE TABLE IF NOT EXISTS sink_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_commits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hash        TEXT NOT NULL,
    parent_hash TEXT,
    eta_cog     REAL,
    created_at  REAL NOT NULL,
    user_id     TEXT,
    agent_id    TEXT,
    run_id      TEXT,
    branch_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_commits_hash ON state_commits(hash, id);

-- Mutable coordination only. Losing trajectories remain append-only states.
CREATE TABLE IF NOT EXISTS branch_leases (
    lease_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    parent_hash TEXT NOT NULL,
    branch_id   TEXT NOT NULL,
    worker_pid  INTEGER,
    status      TEXT NOT NULL CHECK (status IN ('ACTIVE', 'COMMITTED', 'PRUNED', 'EXPIRED')),
    winner_hash TEXT,
    eta_cog     REAL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    UNIQUE (run_id, parent_hash, branch_id)
);

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

_V3_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_states_run_branch
    ON states(run_id, branch_id, scale);
CREATE INDEX IF NOT EXISTS idx_states_user_agent
    ON states(user_id, agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_commits_scope
    ON state_commits(run_id, branch_id);
CREATE INDEX IF NOT EXISTS idx_branch_leases_run_parent
    ON branch_leases(run_id, parent_hash, status);
-- NULL != NULL in SQLite UNIQUE indexes, so root forks use parent_hash = "".
CREATE UNIQUE INDEX IF NOT EXISTS idx_branch_leases_single_winner
    ON branch_leases(run_id, parent_hash)
    WHERE status = 'COMMITTED';
"""

_SCOPE_COLUMNS = ("user_id", "agent_id", "run_id", "branch_id")


class SinkError(RuntimeError):
    """Raised when a record cannot be stored, decoded or verified."""


def _normalise_scope(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


@dataclass(frozen=True, slots=True)
class SinkScope:
    """Explicit entity/run/branch scope for a v3 cold-sink commit."""

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    branch_id: str | None = None

    @classmethod
    def coerce(cls, value: SinkScope | Mapping[str, Any] | Any | None) -> SinkScope:
        if value is None or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                _normalise_scope(value.get("user_id")),
                _normalise_scope(value.get("agent_id")),
                _normalise_scope(value.get("run_id")),
                _normalise_scope(value.get("branch_id")),
            )
        return cls(
            _normalise_scope(getattr(value, "user_id", None)),
            _normalise_scope(getattr(value, "agent_id", None)),
            _normalise_scope(getattr(value, "run_id", None)),
            _normalise_scope(getattr(value, "branch_id", None)),
        )

    def values(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.user_id, self.agent_id, self.run_id, self.branch_id)

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in zip(_SCOPE_COLUMNS, self.values(), strict=True) if value}


def _json_default(value: Any) -> Any:
    """Best-effort deterministic encoding for common non-JSON payload values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    return {"__repr__": repr(value)}


def _json_object_hook(obj: dict[str, Any]) -> Any:
    if set(obj) == {"__bytes__"}:
        try:
            return base64.b64decode(str(obj["__bytes__"]))
        except Exception:
            return obj
    return obj


def canonical_json(payload: Mapping[str, Any] | Any) -> str:
    """Deterministic strict JSON used both for hashing and for storage."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _digest(scale: int, kind: str, raw: str) -> str:
    """Hash immutable content only; parent/eta/scope are provenance metadata."""
    return hashlib.sha256(f"{scale}\x1f{kind}\x1f{raw}".encode()).hexdigest()


@dataclass(frozen=True)
class SinkRecord:
    hash: str
    parent_hash: str | None
    scale: int
    kind: str
    payload: dict[str, Any]
    eta_cog: float | None
    created_at: float
    scope: SinkScope = field(default_factory=SinkScope)

    @property
    def short(self) -> str:
        return self.hash[:10]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "scale": self.scale,
            "kind": self.kind,
            "payload": self.payload,
            "eta_cog": self.eta_cog,
            "created_at": self.created_at,
        }
        scoped = self.scope.as_dict()
        if scoped:
            out["scope"] = scoped
        return out


class ColdSink:
    """Append-only SQLite store with WAL-safe, scoped branch leases."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        compress: bool = True,
        scope: SinkScope | Mapping[str, Any] | None = None,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.db_path = str(db_path)
        self._compress = compress
        self.scope = SinkScope.coerce(scope)
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=self.busy_timeout_ms / 1000.0,
            )
            with self._lock:
                if self.db_path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=FULL;")
                self._conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
                self._conn.executescript(_SCHEMA)
                self._migrate_v3()
                self._ensure_meta()
        except SinkError:
            self._conn.close()
            raise
        except sqlite3.DatabaseError as exc:  # pragma: no cover
            raise SinkError(f"cannot open cold sink at {self.db_path!r}: {exc}") from exc

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        """Serialize writers with BEGIN IMMEDIATE; never upgrade a WAL snapshot."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(sqlite3.DatabaseError):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v3(self) -> None:
        """Add nullable scope columns in place; never rebuild ``states``."""
        for table in ("states", "state_commits"):
            existing = {str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for column in _SCOPE_COLUMNS:
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self._conn.executescript(_V3_INDEXES)
        self._conn.execute("INSERT OR REPLACE INTO sink_meta (key, value) VALUES ('format_version', '3')")

    def _ensure_meta(self) -> None:
        requested = "zlib" if self._compress else "raw"
        row = self._conn.execute("SELECT value FROM sink_meta WHERE key = 'compression'").fetchone()
        if row is None:
            inferred = requested
            sample = self._conn.execute("SELECT payload FROM states LIMIT 1").fetchone()
            if sample is not None:
                try:
                    zlib.decompress(bytes(sample[0]))
                    inferred = "zlib"
                except zlib.error:
                    inferred = "raw"
            if inferred != requested:
                raise SinkError(
                    f"cold sink at {self.db_path!r} uses {inferred} payloads, "
                    f"but compress={self._compress!r} was requested"
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO sink_meta (key, value) VALUES ('compression', ?)",
                (inferred,),
            )
            self._conn.execute("INSERT OR REPLACE INTO sink_meta (key, value) VALUES ('format_version', '3')")
        elif str(row[0]) != requested:
            raise SinkError(
                f"cold sink at {self.db_path!r} uses {row[0]!r} payloads, "
                f"but compress={self._compress!r} was requested"
            )

    def _active_scope(self, scope: SinkScope | Mapping[str, Any] | Any | None) -> SinkScope:
        return self.scope if scope is None else SinkScope.coerce(scope)

    @staticmethod
    def _scope_filter(scope: SinkScope) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in zip(_SCOPE_COLUMNS, scope.values(), strict=True):
            if value is not None:
                clauses.append(f" AND {column} = ?")
                params.append(value)
        return "".join(clauses), params

    def put(
        self,
        kind: str,
        payload: Mapping[str, Any] | Any,
        scale: int = 0,
        parent_hash: str | None = None,
        eta_cog: float | None = None,
        *,
        scope: SinkScope | Mapping[str, Any] | Any | None = None,
    ) -> str:
        """Commit a state and return its content hash (idempotent for identical content)."""
        if not kind:
            raise SinkError("kind is required")
        if not isinstance(scale, int) or scale < 0:
            raise SinkError(f"scale must be a non-negative int, got {scale!r}")
        active = self._active_scope(scope)
        try:
            raw = canonical_json(payload)
            digest = _digest(scale, kind, raw)
        except (TypeError, ValueError) as exc:
            raise SinkError(f"cannot canonicalise {kind!r} payload: {exc}") from exc
        blob = zlib.compress(raw.encode("utf-8"), 6) if self._compress else raw.encode("utf-8")
        created = time.time()
        values = active.values()
        try:
            with self._immediate() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO states
                    (hash, parent_hash, scale, kind, payload, eta_cog, created_at,
                     user_id, agent_id, run_id, branch_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (digest, parent_hash, scale, kind, blob, eta_cog, created, *values),
                )
                # Every commit is retained, even when content dedupes. states keeps
                # first-commit scope; state_commits keeps every scoped observation.
                conn.execute(
                    """
                    INSERT INTO state_commits
                    (hash, parent_hash, eta_cog, created_at,
                     user_id, agent_id, run_id, branch_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (digest, parent_hash, eta_cog, created, *values),
                )
        except sqlite3.DatabaseError as exc:
            raise SinkError(f"cannot commit {kind!r}: {exc}") from exc
        return digest

    def get(self, digest: str, *, verify: bool = True) -> SinkRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at, "
                "user_id, agent_id, run_id, branch_id FROM states WHERE hash = ?",
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
        scope: SinkScope | Mapping[str, Any] | Any | None = None,
    ) -> list[SinkRecord]:
        active = self._active_scope(scope)
        clause, scoped = self._scope_filter(active)
        # Select the first observation in the requested scope, not the scope of
        # the deduplicated content row. Unscoped recall preserves v2 ordering.
        commit_filter = clause
        sql = f"""
            SELECT s.hash, c.parent_hash, s.scale, s.kind, s.payload, c.eta_cog, s.created_at,
                   c.user_id, c.agent_id, c.run_id, c.branch_id
            FROM states s
            JOIN (
                SELECT hash, MIN(id) AS id FROM state_commits WHERE 1=1 {commit_filter} GROUP BY hash
            ) first ON first.hash = s.hash
            JOIN state_commits c ON c.id = first.id
            WHERE s.kind = ?
        """
        params: list[Any] = [*scoped, kind]
        if min_eta is not None:
            sql += " AND c.eta_cog >= ?"
            params.append(min_eta)
        if scale is not None:
            sql += " AND s.scale = ?"
            params.append(scale)
        sql += " ORDER BY c.created_at DESC, s.hash DESC LIMIT ?"
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
        scope: SinkScope | Mapping[str, Any] | Any | None = None,
    ) -> list[SinkRecord]:
        sql = (
            "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at, "
            "user_id, agent_id, run_id, branch_id FROM states WHERE 1=1"
        )
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
        clause, scoped = self._scope_filter(self._active_scope(scope))
        sql += clause
        params.extend(scoped)
        sql += (
            " ORDER BY created_at ASC, hash ASC" if oldest_first else " ORDER BY created_at DESC, hash DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    def lineage(self, digest: str, *, limit: int = 256) -> list[SinkRecord]:
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

    def commits(self, digest: str) -> list[dict[str, Any]]:
        """Return all append-only provenance commits for a content hash."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, hash, parent_hash, eta_cog, created_at, "
                "user_id, agent_id, run_id, branch_id FROM state_commits "
                "WHERE hash = ? ORDER BY id ASC",
                (digest,),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "hash": str(row[1]),
                "parent_hash": str(row[2]) if row[2] else None,
                "eta_cog": float(row[3]) if row[3] is not None else None,
                "created_at": float(row[4]),
                "scope": SinkScope(
                    _normalise_scope(row[5]),
                    _normalise_scope(row[6]),
                    _normalise_scope(row[7]),
                    _normalise_scope(row[8]),
                ).as_dict(),
            }
            for row in rows
        ]

    def children(
        self,
        digest: str,
        *,
        scope: SinkScope | Mapping[str, Any] | Any | None = None,
    ) -> list[SinkRecord]:
        sql = (
            "SELECT hash, parent_hash, scale, kind, payload, eta_cog, created_at, "
            "user_id, agent_id, run_id, branch_id FROM states WHERE parent_hash = ?"
        )
        clause, params = self._scope_filter(self._active_scope(scope))
        sql += clause + " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, [digest, *params]).fetchall()
        return [self._decode(row) for row in rows]

    def count(
        self,
        kind: str | None = None,
        *,
        scope: SinkScope | Mapping[str, Any] | Any | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) FROM states WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        clause, scoped = self._scope_filter(self._active_scope(scope))
        sql += clause
        params.extend(scoped)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row[0])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            kinds = self._conn.execute(
                "SELECT kind, COUNT(*), MIN(scale), MAX(scale) FROM states GROUP BY kind ORDER BY kind"
            ).fetchall()
            total = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM states"
            ).fetchone()
            leases = self._conn.execute("SELECT COUNT(*) FROM branch_leases").fetchone()
            version = self._conn.execute(
                "SELECT value FROM sink_meta WHERE key = 'format_version'"
            ).fetchone()
        return {
            "db_path": self.db_path,
            "records": int(total[0]),
            "bytes": int(total[1]),
            "compression": "zlib" if self._compress else "raw",
            "format_version": int(version[0]) if version is not None else 3,
            "branch_leases": int(leases[0]),
            "kinds": {
                str(kind): {"count": int(n), "min_scale": int(lo), "max_scale": int(hi)}
                for kind, n, lo, hi in kinds
            },
        }

    def schema_version(self) -> int:
        return int(self.stats()["format_version"])

    # -- WAL branch leases --------------------------------------------------
    def open_branch_lease(
        self,
        *,
        run_id: str,
        parent_hash: str | None,
        branch_id: str,
        ttl_s: float = 60.0,
        worker_pid: int | None = None,
        lease_id: str | None = None,
    ) -> str:
        """Create one optimistic ACTIVE lease per (run, parent, branch)."""
        run = str(run_id or "")
        parent = "" if parent_hash is None else str(parent_hash)
        branch = str(branch_id)
        if not branch:
            raise SinkError("branch_id is required")
        now = time.time()
        expires = now + max(0.1, float(ttl_s))
        lease = (
            lease_id
            or hashlib.sha256(
                f"{run}\x1f{parent}\x1f{branch}\x1f{now}\x1f{os.getpid()}".encode()
            ).hexdigest()[:24]
        )
        pid = int(worker_pid if worker_pid is not None else os.getpid())
        error: str | None = None
        try:
            with self._immediate() as conn:
                conn.execute(
                    "UPDATE branch_leases SET status = 'EXPIRED' WHERE status = 'ACTIVE' AND expires_at < ?",
                    (now,),
                )
                winner = conn.execute(
                    "SELECT lease_id FROM branch_leases "
                    "WHERE run_id = ? AND parent_hash = ? AND status = 'COMMITTED' LIMIT 1",
                    (run, parent),
                ).fetchone()
                if winner is not None:
                    error = f"race {run!r}/{parent[:10]!r} already has a committed winner"
                else:
                    existing = conn.execute(
                        "SELECT lease_id, status, expires_at FROM branch_leases "
                        "WHERE run_id = ? AND parent_hash = ? AND branch_id = ?",
                        (run, parent, branch),
                    ).fetchone()
                    if existing is not None:
                        status, expiry = str(existing[1]), float(existing[2])
                        if status == "COMMITTED" or (status == "ACTIVE" and expiry >= now):
                            error = f"branch {branch!r} already has an active/committed lease"
                        else:
                            conn.execute("DELETE FROM branch_leases WHERE lease_id = ?", (existing[0],))
                    if error is None:
                        conn.execute(
                            """
                            INSERT INTO branch_leases
                            (lease_id, run_id, parent_hash, branch_id, worker_pid, status,
                             winner_hash, eta_cog, created_at, expires_at)
                            VALUES (?, ?, ?, ?, ?, 'ACTIVE', NULL, NULL, ?, ?)
                            """,
                            (lease, run, parent, branch, pid, now, expires),
                        )
        except sqlite3.DatabaseError as exc:
            raise SinkError(f"cannot open branch lease: {exc}") from exc
        if error is not None:
            raise SinkError(error)
        return lease

    def claim_winner(self, lease_id: str, *, winner_hash: str, eta_cog: float | None = None) -> None:
        """Atomically commit exactly one winner per (run_id, parent_hash)."""
        now = time.time()
        error: str | None = None
        try:
            with self._immediate() as conn:
                row = conn.execute(
                    "SELECT run_id, parent_hash, status, expires_at FROM branch_leases WHERE lease_id = ?",
                    (lease_id,),
                ).fetchone()
                if row is None:
                    error = f"unknown branch lease {lease_id!r}"
                else:
                    status, expiry = str(row[2]), float(row[3])
                    if status != "ACTIVE":
                        error = f"branch lease {lease_id!r} is {status}, not ACTIVE"
                    elif expiry < now:
                        conn.execute(
                            "UPDATE branch_leases SET status = 'EXPIRED' WHERE lease_id = ?",
                            (lease_id,),
                        )
                        error = f"branch lease {lease_id!r} expired"
                    else:
                        conn.execute(
                            "UPDATE branch_leases SET status = 'COMMITTED', winner_hash = ?, eta_cog = ? "
                            "WHERE lease_id = ?",
                            (str(winner_hash), eta_cog, lease_id),
                        )
        except sqlite3.IntegrityError as exc:
            raise SinkError("a winner is already committed for this parent hash") from exc
        except sqlite3.DatabaseError as exc:
            raise SinkError(f"cannot claim branch winner: {exc}") from exc
        if error is not None:
            raise SinkError(error)

    def prune_losers(self, run_id: str, parent_hash: str | None, *, winner_lease_id: str) -> int:
        run = str(run_id or "")
        parent = "" if parent_hash is None else str(parent_hash)
        with self._immediate() as conn:
            cursor = conn.execute(
                "UPDATE branch_leases SET status = 'PRUNED' "
                "WHERE run_id = ? AND parent_hash = ? AND status = 'ACTIVE' AND lease_id != ?",
                (run, parent, winner_lease_id),
            )
            return int(cursor.rowcount)

    def expire_leases(self, *, now: float | None = None) -> int:
        stamp = float(now if now is not None else time.time())
        with self._immediate() as conn:
            cursor = conn.execute(
                "UPDATE branch_leases SET status = 'EXPIRED' WHERE status = 'ACTIVE' AND expires_at < ?",
                (stamp,),
            )
            return int(cursor.rowcount)

    def leases(
        self,
        *,
        run_id: str | None = None,
        parent_hash: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT lease_id, run_id, parent_hash, branch_id, worker_pid, status, "
            "winner_hash, eta_cog, created_at, expires_at FROM branch_leases WHERE 1=1"
        )
        params: list[Any] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(str(run_id))
        if parent_hash is not None:
            sql += " AND parent_hash = ?"
            params.append(str(parent_hash))
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY created_at ASC, lease_id ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "lease_id": str(row[0]),
                "run_id": str(row[1]),
                "parent_hash": str(row[2]),
                "branch_id": str(row[3]),
                "worker_pid": int(row[4]) if row[4] is not None else None,
                "status": str(row[5]),
                "winner_hash": str(row[6]) if row[6] else None,
                "eta_cog": float(row[7]) if row[7] is not None else None,
                "created_at": float(row[8]),
                "expires_at": float(row[9]),
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ColdSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _decode(self, row: Iterable[Any]) -> SinkRecord:
        digest, parent, scale, kind, blob, eta, created, user, agent, run, branch = row
        try:
            raw = zlib.decompress(bytes(blob)) if self._compress else bytes(blob)
            payload = json.loads(raw.decode("utf-8"), object_hook=_json_object_hook)
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
            scope=SinkScope(
                _normalise_scope(user),
                _normalise_scope(agent),
                _normalise_scope(run),
                _normalise_scope(branch),
            ),
        )
