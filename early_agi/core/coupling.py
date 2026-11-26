"""Bipartite incidence matrix construction: entities x constraints.

This is the honest answer to "where does ``coupling_matrix`` come from?".
No random embeddings: rows are real active entities (files, functions, classes),
columns are real active constraints (gates, hypotheses), and

    M[i, j] = 1[entity i is in constraint j's scope]
            + dependencies from entity i into j's scope
            + dependencies from j's scope into entity i

Dependencies come from the import graph and the resolved call graph. SVD of
``M`` therefore measures something exact and reproducible: whether the active
problem is cleanly modular (low S_SVD, cheap truncation) or globally entangled
(high S_SVD, truncation error overflows ``chi_max``).
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .entropy_sensor import EntanglementReport, entanglement_report, matrix_fingerprint

__all__ = ["Constraint", "CouplingMatrix", "ModuleInfo", "RepoGraph", "RepoScanner"]

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "node_modules",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
}


@dataclass(frozen=True)
class Constraint:
    """A column of the incidence matrix."""

    id: str
    kind: str = "constraint"  # "constraint" | "hypothesis"
    paths: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    description: str = ""

    def matches(self, entity: str) -> bool:
        if entity in self.entities:
            return True
        module = entity.split("::", 1)[0]
        for prefix in self.paths:
            clean = prefix.replace("\\", "/").strip().lstrip("./").rstrip("/")
            if not clean or clean == ".":
                return True  # whole repository
            if module == clean or module.startswith(clean + "/"):
                return True
        return False


@dataclass
class ModuleInfo:
    path: str
    imports: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    calls: dict[str, tuple[str, ...]] = field(default_factory=dict)
    names: tuple[str, ...] = ()
    syntax_error: str | None = None


@dataclass
class RepoGraph:
    root: Path
    modules: dict[str, ModuleInfo]
    edges: dict[str, Counter[str]]
    entities: tuple[str, ...]
    truncated: bool = False

    # -- queries ------------------------------------------------------------
    def dependents(self, entity: str) -> Counter[str]:
        return self.edges.get(entity, Counter())

    def fan_in(self, entity: str) -> int:
        return sum(1 for src in self.edges.values() if entity in src)

    def coupling(
        self,
        constraints: Sequence[Constraint] = (),
        *,
        gates: Mapping[str, Sequence[str]] | None = None,
    ) -> CouplingMatrix:
        columns = list(constraints)
        for name, paths in (gates or {}).items():
            columns.append(
                Constraint(
                    id=f"gate:{name}",
                    kind="constraint",
                    paths=tuple(paths),
                    description="dark-state verifier scope",
                )
            )
        if not columns:
            columns.append(
                Constraint(
                    id="constraint:all", kind="constraint", paths=(".",), description="whole repository"
                )
            )
        return build_coupling(self, columns)


@dataclass(frozen=True)
class CouplingMatrix:
    matrix: np.ndarray
    rows: tuple[str, ...]
    cols: tuple[str, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.rows), len(self.cols)):
            raise ValueError(
                f"shape {self.matrix.shape} does not match {len(self.rows)} entities x {len(self.cols)} constraints"
            )

    # -- physics ------------------------------------------------------------
    def report(self, chi_max: int = 4, tau: float = 0.15) -> EntanglementReport:
        return entanglement_report(self.matrix, chi_max=chi_max, tau=tau)

    def fingerprint(self) -> str:
        return matrix_fingerprint(self.matrix)

    def density(self) -> float:
        return float(np.count_nonzero(self.matrix)) / float(self.matrix.size) if self.matrix.size else 0.0

    def top_rows(self, constraint: str | None = None, *, k: int = 5) -> list[tuple[str, float]]:
        if not self.rows:
            return []
        if constraint is not None:
            if constraint not in self.cols:
                raise KeyError(f"unknown constraint {constraint!r}")
            column = self.matrix[:, self.cols.index(constraint)]
        else:
            column = self.matrix.sum(axis=1)
        order = np.argsort(-column, kind="stable")
        return [(self.rows[i], float(column[i])) for i in order[:k] if column[i] > 0]

    def top_cols(self, *, k: int = 5) -> list[tuple[str, float]]:
        if not self.cols:
            return []
        totals = self.matrix.sum(axis=0)
        order = np.argsort(-totals, kind="stable")
        return [(self.cols[i], float(totals[i])) for i in order[:k] if totals[i] > 0]

    def as_dict(self, chi_max: int = 4, tau: float = 0.15) -> dict[str, Any]:
        report = self.report(chi_max, tau)
        return {
            "shape": list(self.matrix.shape),
            "entities": list(self.rows),
            "constraints": list(self.cols),
            "density": round(self.density(), 6),
            "fingerprint": self.fingerprint(),
            "meta": dict(self.meta),
            "entanglement": report.as_dict(),
        }

    def summary(self, chi_max: int = 4, tau: float = 0.15) -> str:
        report = self.report(chi_max, tau)
        top = ", ".join(f"{name} ({weight:.1f})" for name, weight in self.top_rows(k=3))
        return (
            f"{len(self.rows)} entities x {len(self.cols)} constraints | "
            f"S_SVD={report.spectral_entropy_bits:.2f} bits | "
            f"trunc={report.truncation_error:.2f} | rank={report.effective_rank} | "
            f"{'VOLUME-LAW' if report.is_volume_law else 'area-law'} | top: {top or '<none>'}"
        )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
class RepoScanner:
    """Parse a repository into an import/call dependency graph."""

    def __init__(self, root: str | Path, *, max_files: int = 200, max_entities: int = 500) -> None:
        self.root = Path(root).resolve()
        self.max_files = max(1, max_files)
        self.max_entities = max(1, max_entities)

    def files(self) -> list[Path]:
        found: list[Path] = []
        for path in sorted(self.root.rglob("*.py")):
            relative = path.relative_to(self.root)
            if set(relative.parts) & IGNORED_DIRS:
                continue
            if path.is_file():
                found.append(path)
            if len(found) >= self.max_files:
                break
        return found

    def scan(self) -> RepoGraph:
        paths = self.files()
        truncated = len(paths) >= self.max_files
        modules: dict[str, ModuleInfo] = {}
        dotted: dict[str, str] = {}

        for path in paths:
            module_id = path.relative_to(self.root).as_posix()
            dotted[self._dotted(module_id)] = module_id

        for path in paths:
            module_id = path.relative_to(self.root).as_posix()
            modules[module_id] = self._parse(path, module_id, dotted)

        edges: dict[str, Counter[str]] = defaultdict(Counter)

        # module import edges
        for module_id, info in modules.items():
            for imported in info.imports:
                if imported != module_id:
                    edges[module_id][imported] += 1

        # function call edges (resolved by simple name across modules)
        simple_index: dict[str, list[str]] = defaultdict(list)
        for info in modules.values():
            for function in info.functions:
                simple_index[function.split("::", 1)[1].split(".")[-1]].append(function)

        for info in modules.values():
            for caller, callees in info.calls.items():
                for simple in callees:
                    for target in simple_index.get(simple, ()):
                        if target != caller:
                            edges[caller][target] += 1

        entities = self._select_entities(modules, edges)
        return RepoGraph(
            root=self.root,
            modules=modules,
            edges={key: value for key, value in edges.items()},
            entities=entities,
            truncated=truncated,
        )

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _dotted(module_id: str) -> str:
        stem = module_id[:-3] if module_id.endswith(".py") else module_id
        if stem.endswith("/__init__"):
            stem = stem[: -len("/__init__")]
        return stem.replace("/", ".")

    def _parse(self, path: Path, module_id: str, dotted: Mapping[str, str]) -> ModuleInfo:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            return ModuleInfo(path=module_id, syntax_error=f"{type(exc).__name__}: {exc}")

        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve(alias.name, dotted)
                    if resolved:
                        imports.add(resolved)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = module_id.split("/")[:-1]
                    keep = max(0, len(package) - (node.level - 1))
                    prefix = ".".join(package[:keep])
                    base = f"{prefix}.{base}" if prefix and base else prefix or base
                for alias in node.names:
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    resolved = self._resolve(candidate, dotted) or self._resolve(base, dotted)
                    if resolved:
                        imports.add(resolved)

        functions: list[str] = []
        calls: dict[str, tuple[str, ...]] = {}
        top_names: list[str] = []

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualified = f"{prefix}{child.name}"
                    entity = f"{module_id}::{qualified}"
                    functions.append(entity)
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        callees: list[str] = []
                        for inner in ast.walk(child):
                            if isinstance(inner, ast.Call):
                                target = inner.func
                                if isinstance(target, ast.Name):
                                    callees.append(target.id)
                                elif isinstance(target, ast.Attribute):
                                    callees.append(target.attr)
                        calls[entity] = tuple(callees)
                    visit(child, f"{qualified}.")
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    for target in getattr(child, "targets", [getattr(child, "target", None)]):
                        if isinstance(target, ast.Name):
                            top_names.append(target.id)

        visit(tree, "")
        return ModuleInfo(
            path=module_id,
            imports=tuple(sorted(imports)),
            functions=tuple(functions),
            calls=calls,
            names=tuple(sorted(set(top_names))),
        )

    @staticmethod
    def _resolve(candidate: str, dotted: Mapping[str, str]) -> str | None:
        if candidate in dotted:
            return dotted[candidate]
        parts = candidate.split(".")
        while len(parts) > 1:
            parts.pop()
            probe = ".".join(parts)
            if probe in dotted:
                return dotted[probe]
        return None

    def _select_entities(
        self, modules: Mapping[str, ModuleInfo], edges: Mapping[str, Counter[str]]
    ) -> tuple[str, ...]:
        scored: list[tuple[int, str]] = [(0, module_id) for module_id in modules]
        for info in modules.values():
            for function in info.functions:
                inbound = sum(1 for src in edges.values() if function in src)
                scored.append((inbound, function))
        # highest fan-in first, then name, so the cut is deterministic
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [name for _, name in scored[: self.max_entities]]
        return tuple(sorted(chosen))


def build_coupling(graph: RepoGraph, constraints: Sequence[Constraint]) -> CouplingMatrix:
    rows = graph.entities
    cols = [constraint.id for constraint in constraints]
    matrix = np.zeros((len(rows), len(cols)), dtype=np.float64)

    scopes: list[set[str]] = []
    for constraint in constraints:
        scopes.append({entity for entity in rows if constraint.matches(entity)})

    for column, scope in enumerate(scopes):
        if not scope:
            continue
        for index, entity in enumerate(rows):
            # dependencies from the scope into this entity (the coupling that
            # drags an otherwise-unrelated entity into the constraint)
            inbound = float(
                sum(
                    counter.get(entity, 0)
                    for source, counter in graph.edges.items()
                    if source in scope and source != entity
                )
            )
            if entity in scope:
                # presence + dependencies *into* the scope (internal cohesion)
                outbound = float(
                    sum(
                        count
                        for target, count in graph.edges.get(entity, Counter()).items()
                        if target in scope and target != entity
                    )
                )
                matrix[index, column] = 1.0 + inbound + outbound
            else:
                matrix[index, column] = inbound

    meta = {
        "root": str(graph.root),
        "files": len(graph.modules),
        "entities": len(rows),
        "dependencies": sum(len(counter) for counter in graph.edges.values()),
        "dependency_weight": int(sum(sum(counter.values()) for counter in graph.edges.values())),
        "truncated": graph.truncated,
        "syntax_errors": sorted(
            module.path for module in graph.modules.values() if module.syntax_error is not None
        ),
    }
    return CouplingMatrix(matrix=matrix, rows=tuple(rows), cols=tuple(cols), meta=meta)
