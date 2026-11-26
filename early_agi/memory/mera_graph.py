"""MERA-style scale memory: L0 (UV traces) -> Lk (IR rules).

The cold sink holds every trace at scale 0. Consolidation is the renormalisation
step: read the high-eta traces, build a gate/directive feature matrix, truncate
it with the *same* SVD backend the kernel uses, and emit one rule node per
cluster at scale+1. Rules are written back through the sink, so coarse nodes are
as lossless as the traces they summarise.

Rule *invention* by a model is Phase 4. Reflex compilation — turning a recurring
gated edit batch into a zero-LLM replay — lives in :mod:`early_agi.memory.reflex`
and consumes the same store: the ``applied_edit`` traces plus the ``rule`` nodes
emitted here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import RelaxationBackend
from ..backends.classical_svd import ClassicalSVDBackend
from ..core.cold_sink import ColdSink, SinkRecord

__all__ = ["CoarseGrainReport", "MeraGraph"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    hash        TEXT PRIMARY KEY,
    scale       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    parent_hash TEXT,
    eta_cog     REAL,
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_scale ON nodes(scale, kind);

CREATE TABLE IF NOT EXISTS edges (
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight   REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, relation)
);

CREATE TABLE IF NOT EXISTS levels (
    scale      INTEGER PRIMARY KEY,
    clusters   INTEGER NOT NULL,
    support    INTEGER NOT NULL,
    truncation REAL NOT NULL,
    entropy    REAL NOT NULL,
    rule       TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class CoarseGrainReport:
    scale: int
    records: int
    features: int
    clusters: int
    rules: tuple[str, ...] = ()
    truncation_error: float = 0.0
    spectral_entropy_bits: float = 0.0
    skipped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "records": self.records,
            "features": self.features,
            "clusters": self.clusters,
            "rules": list(self.rules),
            "truncation_error": round(self.truncation_error, 4),
            "spectral_entropy_bits": round(self.spectral_entropy_bits, 4),
            "skipped": self.skipped,
        }


def _features(record: SinkRecord) -> tuple[str, ...]:
    payload = record.payload
    gates = payload.get("gates") or {}
    passed = tuple(sorted(name for name, verdict in gates.items() if verdict == "pass"))
    directive = str(payload.get("directive", ""))
    return tuple([*(f"gate:{name}" for name in passed), f"directive:{directive}"])


class MeraGraph:
    def __init__(
        self,
        sink: ColdSink,
        db_path: str | Path = ":memory:",
        *,
        backend: RelaxationBackend | None = None,
    ) -> None:
        self.sink = sink
        self.db_path = str(db_path)
        self.backend = backend or ClassicalSVDBackend()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.executescript(_SCHEMA)

    # -- indexing -----------------------------------------------------------
    def put_node(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        scale: int,
        parent_hash: str | None = None,
        eta_cog: float | None = None,
        weight: float = 1.0,
    ) -> str:
        digest = self.sink.put(kind, payload, scale=scale, parent_hash=parent_hash, eta_cog=eta_cog)
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes (hash, scale, kind, payload, parent_hash, eta_cog, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                digest,
                scale,
                kind,
                json.dumps(dict(payload), sort_keys=True, default=str),
                parent_hash,
                eta_cog,
                weight,
                time.time(),
            ),
        )
        return digest

    def link(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, relation, weight) VALUES (?, ?, ?, ?)",
            (src, dst, relation, float(weight)),
        )

    # -- renormalisation ----------------------------------------------------
    def coarse_grain(
        self,
        scale: int = 0,
        *,
        min_eta: float | None = None,
        limit: int = 500,
        chi_max: int = 8,
        tau: float = 0.15,
        kinds: Sequence[str] = ("uv_state",),
    ) -> CoarseGrainReport:
        records = self.sink.records(kinds=kinds, scale=scale, min_eta=min_eta, limit=limit)
        if len(records) < 2:
            return CoarseGrainReport(
                scale=scale, records=len(records), features=0, clusters=0, skipped="need >= 2 traces"
            )

        vocabulary = sorted({feature for record in records for feature in _features(record)})
        if len(vocabulary) < 2:
            return CoarseGrainReport(
                scale=scale,
                records=len(records),
                features=len(vocabulary),
                clusters=0,
                skipped="feature matrix is degenerate",
            )
        index = {feature: position for position, feature in enumerate(vocabulary)}
        matrix = np.zeros((len(records), len(vocabulary)), dtype=np.float64)
        for row, record in enumerate(records):
            for feature in _features(record):
                matrix[row, index[feature]] = 1.0

        report = self.backend.measure_entanglement(matrix, chi_max=chi_max, tau=tau)
        k = max(1, min(int(report.effective_rank) or 1, len(records) - 1, chi_max))
        u, _, _ = np.linalg.svd(matrix, full_matrices=False)
        # Partition by orthant in the truncated Schmidt basis: deterministic, and
        # unlike argmax on |U| it separates the leading contrast at k >= 2.
        signs = np.where(u[:, :k] >= 0.0, 1, -1)
        buckets: dict[tuple[int, ...], list[SinkRecord]] = {}
        for row_index, record in enumerate(records):
            key = tuple(int(value) for value in signs[row_index])
            buckets.setdefault(key, []).append(record)

        rule_hashes: list[str] = []
        support_total = 0
        for members in buckets.values():
            if len(members) < 2:
                continue
            shared = Counter(feature for member in members for feature in set(_features(member)))
            pattern = sorted(feature for feature, count in shared.items() if count == len(members))
            directives = Counter(str(member.payload.get("directive", "")) for member in members)
            etas = [member.eta_cog for member in members if member.eta_cog is not None]
            payload = {
                "pattern": pattern,
                "support": len(members),
                "eta_mean": round(float(np.mean(etas)), 6) if etas else None,
                "directive": directives.most_common(1)[0][0] if directives else "",
                "members": [member.hash for member in members],
                "source_scale": scale,
            }
            digest = self.put_node(
                "rule",
                payload,
                scale=scale + 1,
                eta_cog=payload["eta_mean"],
                weight=float(len(members)),
            )
            rule_hashes.append(digest)
            support_total += len(members)
            for member in members:
                self.link(member.hash, digest, "absorbed_by", 1.0 / len(members))

        self._conn.execute(
            "INSERT OR REPLACE INTO levels (scale, clusters, support, truncation, entropy, rule, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                scale + 1,
                len(rule_hashes),
                support_total,
                report.truncation_error,
                report.spectral_entropy_bits,
                json.dumps({"vocabulary": vocabulary[:32], "k": k}, sort_keys=True),
                time.time(),
            ),
        )
        return CoarseGrainReport(
            scale=scale,
            records=len(records),
            features=len(vocabulary),
            clusters=len(rule_hashes),
            rules=tuple(rule_hashes),
            truncation_error=report.truncation_error,
            spectral_entropy_bits=report.spectral_entropy_bits,
        )

    # -- queries ------------------------------------------------------------
    def nodes(self, scale: int | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT hash, scale, kind, payload, parent_hash, eta_cog, weight FROM nodes WHERE 1=1"
        params: list[Any] = []
        if scale is not None:
            sql += " AND scale = ?"
            params.append(scale)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY scale ASC, created_at ASC, hash ASC"
        return [
            {
                "hash": row[0],
                "scale": row[1],
                "kind": row[2],
                "payload": json.loads(row[3]),
                "parent_hash": row[4],
                "eta_cog": row[5],
                "weight": row[6],
            }
            for row in self._conn.execute(sql, params).fetchall()
        ]

    def rules(self, *, scale: int | None = None, min_support: int = 1) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes(scale=scale, kind="rule")
            if int(node["payload"].get("support", 0)) >= min_support
        ]

    def members(self, rule_hash: str) -> list[str]:
        return self.sources(rule_hash, "absorbed_by")

    def sources(self, digest: str, relation: str = "compiled_from") -> list[str]:
        """Inbound edges of one relation: what this node was built from."""
        return [
            row[0]
            for row in self._conn.execute(
                "SELECT src FROM edges WHERE dst = ? AND relation = ? ORDER BY src", (digest, relation)
            ).fetchall()
        ]

    def levels(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT scale, clusters, support, truncation, entropy, rule, created_at FROM levels ORDER BY scale"
        ).fetchall()
        return [
            {
                "scale": row[0],
                "clusters": row[1],
                "support": row[2],
                "truncation_error": row[3],
                "spectral_entropy_bits": row[4],
                "rule": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]

    def reflex_candidates(self, min_support: int = 2) -> list[dict[str, Any]]:
        """Recurring rules: the input to Phase-3 reflex compilation."""
        return self.rules(min_support=min_support)

    def close(self) -> None:
        self._conn.close()
