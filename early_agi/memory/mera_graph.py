"""MERA-style scale memory with backend-driven truncation."""

from __future__ import annotations

import ast
import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import RelaxationBackend
from ..backends.classical_svd import ClassicalSVDBackend
from ..core.cold_sink import ColdSink, SinkRecord, canonical_json
from ..core.plasticity import AnalogicalMatcher, AnalogyMatch, StructuralSignature

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
    analogies: int = 0

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
            "analogies": self.analogies,
        }


def _features(record: SinkRecord) -> tuple[str, ...]:
    payload = record.payload
    gates = payload.get("gates") or {}
    passed = tuple(sorted(name for name, verdict in gates.items() if verdict == "pass"))
    directive = str(payload.get("directive", ""))
    return tuple([*(f"gate:{name}" for name in passed), f"directive:{directive}"])


def _motif_counts(records: Sequence[SinkRecord]) -> tuple[int, int, int, int]:
    """AST motif ratios used by the scale-invariant analogical signature."""
    files: set[str] = set()
    imports = 0
    functions = 0
    gates = 0
    for record in records:
        state = record.payload.get("state")
        if not isinstance(state, Mapping):
            state = {}
        for path in state.get("files") or state.get("paths") or ():
            files.add(str(path))
        workspace = record.payload.get("workspace")
        if isinstance(workspace, Mapping) and not files:
            for path in workspace.get("files") or ():
                files.add(str(path))
        sources = state.get("sources")
        if isinstance(sources, Mapping):
            for source in sources.values():
                text = str(source)
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    imports += sum(
                        1 for line in text.splitlines() if line.strip().startswith(("import ", "from "))
                    )
                    functions += sum(
                        1 for line in text.splitlines() if line.strip().startswith(("def ", "async def "))
                    )
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports += 1
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions += 1
        gate_map = record.payload.get("gates")
        if isinstance(gate_map, Mapping):
            gates = max(gates, len(gate_map))
    return max(1, len(files)), imports, functions, gates


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
        self.analogies = AnalogicalMatcher()
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
        self._load_analogies()

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
        encoded = canonical_json(dict(payload))
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO nodes (hash, scale, kind, payload, parent_hash, eta_cog, weight, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (digest, scale, kind, encoded, parent_hash, eta_cog, weight, time.time()),
            )
        return digest

    def _load_analogies(self) -> None:
        for node in self.nodes(kind="analogy"):
            payload = node["payload"]
            pattern_id = str(payload.get("pattern_id") or node["hash"])
            signature = StructuralSignature.from_payload(payload.get("signature", {}))
            self.analogies.register(
                pattern_id,
                signature,
                str(payload.get("strategy_hint", "")),
                payload.get("metadata", {}),
            )

    def encode_signature(
        self,
        coupling_matrix: Any,
        *,
        num_modules: int,
        num_imports: int,
        num_functions: int,
        num_gates: int,
    ) -> StructuralSignature:
        """Encode a coordinate-free structural fingerprint for analogical recall."""
        return self.analogies.encode(
            coupling_matrix,
            num_modules=num_modules,
            num_imports=num_imports,
            num_functions=num_functions,
            num_gates=num_gates,
        )

    def register_analogy(
        self,
        pattern_id: str,
        signature: StructuralSignature,
        strategy_hint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist an L1/L2 strategy alongside its structural signature."""
        payload = {
            "pattern_id": pattern_id,
            "signature": signature.as_dict(),
            "strategy_hint": strategy_hint,
            "metadata": dict(metadata or {}),
        }
        digest = self.put_node("analogy", payload, scale=2, weight=1.0)
        self.analogies.register(pattern_id, signature, strategy_hint, metadata)
        return digest

    def best_analogy(
        self, signature: StructuralSignature, min_similarity: float = 0.72
    ) -> AnalogyMatch | None:
        """Recall the closest structural strategy learned in any repository."""
        return self.analogies.best_match(signature, min_similarity=min_similarity)

    def link(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (src, dst, relation, weight) VALUES (?, ?, ?, ?)",
                (src, dst, relation, float(weight)),
            )

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
                scale=scale,
                records=len(records),
                features=0,
                clusters=0,
                skipped="need >= 2 traces",
                analogies=0,
            )

        vocabulary = sorted({feature for record in records for feature in _features(record)})
        if len(vocabulary) < 2:
            return CoarseGrainReport(
                scale=scale,
                records=len(records),
                features=len(vocabulary),
                clusters=0,
                skipped="feature matrix is degenerate",
                analogies=0,
            )
        index = {feature: position for position, feature in enumerate(vocabulary)}
        matrix = np.zeros((len(records), len(vocabulary)), dtype=np.float64)
        for row, record in enumerate(records):
            for feature in _features(record):
                matrix[row, index[feature]] = 1.0

        report = self.backend.measure_entanglement(matrix, chi_max=chi_max, tau=tau)
        num_modules, num_imports, num_functions, num_gates = _motif_counts(records)
        signature = self.encode_signature(
            matrix,
            num_modules=num_modules,
            num_imports=num_imports,
            num_functions=num_functions,
            num_gates=num_gates,
        )
        k = max(1, min(int(report.effective_rank) or 1, len(records) - 1, chi_max))
        # Honour both halves of the backend contract; do not bypass the seam with
        # a hard-coded numpy decomposition.
        np.asarray(self.backend.truncate(matrix, chi_max=k))
        left_basis = getattr(self.backend, "left_basis", None)
        if left_basis is not None:
            u = np.asarray(left_basis(matrix, k=k), dtype=np.float64)
        else:  # pragma: no cover - compatibility fallback for older custom backends
            u, _, _ = np.linalg.svd(matrix, full_matrices=False)
        if u.ndim != 2:
            u = u.reshape(len(records), -1)
        k = min(k, u.shape[1]) if u.shape[1] else 0
        signs = np.where(u[:, :k] >= 0.0, 1, -1) if k else np.ones((len(records), 1), dtype=int)
        buckets: dict[tuple[int, ...], list[SinkRecord]] = {}
        for row_index, record in enumerate(records):
            key = tuple(int(value) for value in signs[row_index])
            buckets.setdefault(key, []).append(record)

        rule_hashes: list[str] = []
        support_total = 0
        analogy_count = 0
        for members in buckets.values():
            if len(members) < 2:
                continue
            shared = Counter(feature for member in members for feature in set(_features(member)))
            pattern = sorted(feature for feature, count in shared.items() if count == len(members))
            directives = Counter(str(member.payload.get("directive", "")) for member in members)
            etas = [member.eta_cog for member in members if member.eta_cog is not None]
            strategy_hint = directives.most_common(1)[0][0] if directives else ""
            payload = {
                "pattern": pattern,
                "support": len(members),
                "eta_mean": round(float(np.mean(etas)), 6) if etas else None,
                "directive": strategy_hint,
                "members": [member.hash for member in members],
                "source_scale": scale,
                "structural_signature": signature.as_dict(),
                "strategy_hint": strategy_hint,
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
            self.register_analogy(
                digest,
                signature,
                strategy_hint,
                metadata={
                    "source_scale": scale,
                    "support": len(members),
                    "pattern": list(pattern),
                    "eta_mean": payload["eta_mean"],
                },
            )
            analogy_count += 1
            for member in members:
                self.link(member.hash, digest, "absorbed_by", 1.0 / len(members))

        with self._lock:
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
            analogies=analogy_count,
        )

    def nodes(self, scale: int | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT hash, scale, kind, payload, parent_hash, eta_cog, weight, created_at FROM nodes WHERE 1=1"
        )
        params: list[Any] = []
        if scale is not None:
            sql += " AND scale = ?"
            params.append(scale)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY scale ASC, created_at ASC, hash ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "hash": row[0],
                "scale": row[1],
                "kind": row[2],
                "payload": json.loads(row[3]),
                "parent_hash": row[4],
                "eta_cog": row[5],
                "weight": row[6],
                "created_at": row[7],
            }
            for row in rows
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
        with self._lock:
            rows = self._conn.execute(
                "SELECT src FROM edges WHERE dst = ? AND relation = ? ORDER BY src", (digest, relation)
            ).fetchall()
        return [row[0] for row in rows]

    def levels(self) -> list[dict[str, Any]]:
        with self._lock:
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
        return self.rules(min_support=min_support)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MeraGraph:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
