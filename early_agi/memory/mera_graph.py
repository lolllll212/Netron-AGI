"""MERA-style scale memory with backend-driven truncation."""

from __future__ import annotations

import ast
import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import RelaxationBackend
from ..backends.classical_svd import ClassicalSVDBackend
from ..core.adaptation.signatures import AnalogicalMatcher, AnalogyMatch, StructuralSignature
from ..core.cold_sink import ColdSink, SinkRecord, SinkScope, canonical_json

__all__ = ["MERGED", "SUPERSEDED", "CoarseGrainReport", "MeraGraph"]

SUPERSEDED = "SUPERSEDED"
MERGED = "MERGED"
_LINEAGE_RETIRE_RELATIONS = (SUPERSEDED, MERGED)

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
CREATE INDEX IF NOT EXISTS idx_edges_relation_src ON edges(relation, src);

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
    guidance: int = 0

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
            "guidance": self.guidance,
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
        self._conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        with self._lock:
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=30000;")
            self._conn.executescript(_SCHEMA)
        self._load_analogies()

    @staticmethod
    def _count(value: Any, default: int = 0) -> int:
        if isinstance(value, (list, tuple, set)):
            return len(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _default_signature(self, payload: Mapping[str, Any]) -> StructuralSignature:
        modules = max(1, self._count(payload.get("num_modules", payload.get("modules")), 1))
        matrix = np.eye(modules, dtype=np.float64)
        return self.encode_signature(
            matrix,
            num_modules=modules,
            num_imports=self._count(payload.get("num_imports", payload.get("imports")), 0),
            num_functions=self._count(payload.get("num_functions", payload.get("functions")), 0),
            num_gates=self._count(payload.get("num_gates", payload.get("gates")), 0),
        )

    @staticmethod
    def _decomposition_strategy(
        payload: Mapping[str, Any], strategy: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "module_cut": "dependency_cut",
            "test_locking_order": ["read-only probe", "lock tests", "crystallize"],
            "probe_depth": 1,
        }
        if isinstance(strategy, Mapping):
            result.update(dict(strategy))
        for key in ("module_cut", "test_locking_order", "probe_depth"):
            if key in payload:
                result[key] = payload[key]
        return result

    def put_node(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        scale: int,
        parent_hash: str | None = None,
        eta_cog: float | None = None,
        weight: float = 1.0,
        signature: StructuralSignature | None = None,
        strategy: Mapping[str, Any] | None = None,
        scope: SinkScope | Mapping[str, Any] | None = None,
    ) -> str:
        """Persist a node, stamping every L1/L2 node with a structural signature."""
        data = dict(payload)
        structural: StructuralSignature | None = None
        if scale >= 1:
            raw = data.get("structural_signature")
            if isinstance(raw, Mapping):
                structural = StructuralSignature.from_payload(raw)
            elif signature is not None:
                structural = signature
            else:
                nested = data.get("signature")
                structural = (
                    StructuralSignature.from_payload(nested)
                    if isinstance(nested, Mapping) and "spectrum" in nested
                    else self._default_signature(data)
                )
            data["structural_signature"] = structural.as_dict()
            data["decomposition_strategy"] = self._decomposition_strategy(data, strategy)
        digest = self.sink.put(
            kind,
            data,
            scale=scale,
            parent_hash=parent_hash,
            eta_cog=eta_cog,
            scope=scope,
        )
        encoded = canonical_json(data)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO nodes (hash, scale, kind, payload, parent_hash, eta_cog, weight, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (digest, scale, kind, encoded, parent_hash, eta_cog, weight, time.time()),
                )
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(Exception):
                    self._conn.execute("ROLLBACK")
                raise
        if scale >= 1 and structural is not None:
            meta = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), Mapping) else {}
            meta.update(data["decomposition_strategy"])
            meta.update({"kind": kind, "scale": scale, "node_hash": digest})
            self.analogies.register(
                f"node:{digest}",
                structural,
                str(data.get("strategy_hint") or data.get("directive") or "module_cut"),
                meta,
            )
        return digest

    def _load_analogies(self) -> None:
        all_nodes = self.nodes()
        ordered = [node for node in all_nodes if node["kind"] == "analogy"]
        ordered.extend(node for node in all_nodes if node["kind"] != "analogy")
        for node in ordered:
            payload = node["payload"]
            raw = payload.get("structural_signature", payload.get("signature"))
            if node["scale"] < 1 or not isinstance(raw, Mapping) or "spectrum" not in raw:
                continue
            signature = StructuralSignature.from_payload(raw)
            metadata = (
                dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {}
            )
            strategy = payload.get("decomposition_strategy")
            if isinstance(strategy, Mapping):
                metadata.update(dict(strategy))
            self.analogies.register(
                str(payload.get("pattern_id") or f"node:{node['hash']}"),
                signature,
                str(payload.get("strategy_hint") or payload.get("directive") or "module_cut"),
                metadata,
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
        meta = dict(metadata or {})
        payload = {
            "pattern_id": pattern_id,
            "signature": signature.as_dict(),
            "structural_signature": signature.as_dict(),
            "strategy_hint": strategy_hint,
            "metadata": meta,
        }
        digest = self.put_node("analogy", payload, scale=2, weight=1.0, signature=signature, strategy=meta)
        self.analogies.register(pattern_id, signature, strategy_hint, meta)
        return digest

    def best_analogy(
        self,
        signature: StructuralSignature,
        min_similarity: float = 0.75,
        *,
        latest_only: bool = True,
    ) -> AnalogyMatch | None:
        """Recall the closest structural strategy learned in any repository."""
        excluded = self._retired_pattern_ids() if latest_only else frozenset()
        return self.analogies.best_match(
            signature,
            min_similarity=min_similarity,
            exclude_pattern_ids=excluded,
        )

    def recall_decomposition_strategy(
        self,
        signature: StructuralSignature,
        min_similarity: float = 0.75,
        *,
        latest_only: bool = True,
    ) -> dict[str, Any] | None:
        """Zero-shot recall of module cuts and test-locking order at cosine >= .75."""
        excluded = self._retired_pattern_ids() if latest_only else frozenset()
        match = self.analogies.best_match(
            signature,
            min_similarity=min_similarity,
            exclude_pattern_ids=excluded,
        )
        if match is None:
            return None
        metadata = dict(match.metadata)
        return {
            "pattern_id": match.pattern_id,
            "similarity": match.similarity,
            "strategy_hint": match.strategy_hint,
            "module_cut": metadata.get("module_cut", "dependency_cut"),
            "test_locking_order": list(
                metadata.get("test_locking_order", ["read-only probe", "lock tests", "crystallize"])
            ),
            "metadata": metadata,
        }

    def link(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO edges (src, dst, relation, weight) VALUES (?, ?, ?, ?)",
                    (src, dst, relation, float(weight)),
                )
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(Exception):
                    self._conn.execute("ROLLBACK")
                raise

    def supersede(self, old_hash: str, new_hash: str, *, weight: float = 1.0) -> None:
        """Retire ``old_hash`` without deleting it: append a SUPERSEDED edge."""
        self.link(old_hash, new_hash, SUPERSEDED, weight)

    def mark_merged(self, merged_hash: str, winner_hash: str, *, weight: float = 1.0) -> None:
        """Record that a losing branch was merged into the winner, append-only."""
        self.link(merged_hash, winner_hash, MERGED, weight)

    def record_race_lineage(self, winner_hash: str, loser_hashes: Sequence[str]) -> None:
        """Retire every losing race branch while keeping full edge audit history."""
        for loser in dict.fromkeys(str(item) for item in loser_hashes if str(item) != str(winner_hash)):
            self.supersede(loser, winner_hash)
            self.mark_merged(loser, winner_hash)

    def edges(self, *, relation: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT src, dst, relation, weight FROM edges WHERE 1=1"
        params: list[Any] = []
        if relation is not None:
            sql += " AND relation = ?"
            params.append(str(relation))
        sql += " ORDER BY src ASC, dst ASC, relation ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "src": str(row[0]),
                "dst": str(row[1]),
                "relation": str(row[2]),
                "weight": float(row[3]),
            }
            for row in rows
        ]

    def _retired_hashes(self) -> set[str]:
        placeholders = ",".join("?" for _ in _LINEAGE_RETIRE_RELATIONS)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT src FROM edges WHERE relation IN ({placeholders})",
                _LINEAGE_RETIRE_RELATIONS,
            ).fetchall()
        return {str(row[0]) for row in rows}

    def _retired_pattern_ids(self) -> set[str]:
        retired = self._retired_hashes()
        if not retired:
            return set()
        excluded: set[str] = set(retired)
        for pattern_id, _signature, _hint, meta in self.analogies.patterns:
            bare = str(pattern_id).removeprefix("node:")
            node_hash = str(meta.get("node_hash", ""))
            if bare in retired or node_hash in retired:
                excluded.add(str(pattern_id))
        return excluded

    def coarse_grain(
        self,
        scale: int = 0,
        *,
        min_eta: float | None = None,
        limit: int = 500,
        chi_max: int = 8,
        tau: float = 0.15,
        kinds: Sequence[str] = ("uv_state",),
        latest_only: bool = True,
    ) -> CoarseGrainReport:
        records = self.sink.records(kinds=kinds, scale=scale, min_eta=min_eta, limit=limit)
        if latest_only:
            retired = self._retired_hashes()
            if retired:
                records = [record for record in records if record.hash not in retired]
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
        guidance_count = 0
        decomposition = {
            "module_cut": "dependency_cut",
            "test_locking_order": ["read-only probe", "lock tests", "crystallize"],
            "probe_depth": 1,
        }
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
                signature=signature,
                strategy=decomposition,
            )
            rule_hashes.append(digest)
            support_total += len(members)
            metadata = {
                "source_scale": scale,
                "support": len(members),
                "pattern": list(pattern),
                "eta_mean": payload["eta_mean"],
                **decomposition,
            }
            self.register_analogy(digest, signature, strategy_hint, metadata=metadata)
            analogy_count += 1

            # L0 stays verbatim in ColdSink; the L1/L2 guidance is abstract and
            # carries no concrete paths or function names.
            guidance_payload = {
                "abstraction_level": f"L{max(1, scale + 1)}",
                "sequence": ["read-only probe", "lock tests", "module cut", "crystallize"],
                "module_cut": decomposition["module_cut"],
                "test_locking_order": list(decomposition["test_locking_order"]),
                "probe_depth": decomposition["probe_depth"],
                "support": len(members),
                "eta_mean": payload["eta_mean"],
                "strategy_hint": strategy_hint,
                "source_scale": scale,
                "path_independent": True,
                "no_function_names": True,
                "member_hashes": [member.hash for member in members],
            }
            guidance_digest = self.put_node(
                "guidance",
                guidance_payload,
                scale=scale + 1,
                eta_cog=payload["eta_mean"],
                weight=float(len(members)),
                signature=signature,
                strategy=decomposition,
            )
            guidance_count += 1
            self.link(digest, guidance_digest, "distilled_into", 1.0)
            self.register_analogy(
                guidance_digest,
                signature,
                strategy_hint or "module_cut",
                metadata={**metadata, "guidance": True},
            )
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
            analogies=analogy_count + guidance_count,
            guidance=guidance_count,
        )

    def nodes(
        self,
        scale: int | None = None,
        kind: str | None = None,
        *,
        latest_only: bool = True,
    ) -> list[dict[str, Any]]:
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
        if latest_only:
            placeholders = ",".join("?" for _ in _LINEAGE_RETIRE_RELATIONS)
            sql += f" AND hash NOT IN (SELECT src FROM edges WHERE relation IN ({placeholders}))"
            params.extend(_LINEAGE_RETIRE_RELATIONS)
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

    def rules(
        self,
        *,
        scale: int | None = None,
        min_support: int = 1,
        latest_only: bool = True,
    ) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes(scale=scale, kind="rule", latest_only=latest_only)
            if int(node["payload"].get("support", 0)) >= min_support
        ]

    def members(self, rule_hash: str, *, latest_only: bool = True) -> list[str]:
        return self.sources(rule_hash, "absorbed_by", latest_only=latest_only)

    def sources(self, digest: str, relation: str = "compiled_from", *, latest_only: bool = True) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT src FROM edges WHERE dst = ? AND relation = ? ORDER BY src", (digest, relation)
            ).fetchall()
        retired = self._retired_hashes() if latest_only else set()
        return [str(row[0]) for row in rows if str(row[0]) not in retired]

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

    def reflex_candidates(self, min_support: int = 2, *, latest_only: bool = True) -> list[dict[str, Any]]:
        return self.rules(min_support=min_support, latest_only=latest_only)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MeraGraph:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
