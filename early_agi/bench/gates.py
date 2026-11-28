"""Local gates, transactional fixture IO, and the Gauge-side A/B arm.

No wall-clock measurements are written into reports. Offline latency and token
estimates are deterministic; real correctness comes from parsing every Python
file and running the fixture's pytest files in a temporary workspace.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..config import Settings
from ..core.entropy_sensor import EntanglementReport
from ..core.governor import CognitiveStepCost, ThermodynamicGovernor
from ..gauge.vm import GaugeVM
from ..operators.dark_state import GateResult, ast_syntax_gate, pytest_gate
from ..operators.registry import build_default_registry
from .contracts import FileEdit, is_safe_relative, validated_edits
from .fixtures import BugFixture

__all__ = [
    "GateReport",
    "GaugeRunReport",
    "apply_edits",
    "attempt_cost",
    "baseline_eta",
    "capture_snapshot",
    "evaluate_baseline_gates",
    "gate_latency_ms",
    "load_gauge_program",
    "read_sources",
    "restore_snapshot",
    "run_netron_gauge",
    "write_workspace",
]

_IGNORED_PARTS = frozenset(
    {".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist", "node_modules", "venv"}
)


def _clear_pycache(root: Path) -> None:
    """Remove stale bytecode before a full-file edit is re-imported by pytest."""
    directories = [
        *root.rglob("__pycache__"),
        *root.rglob(".pytest_cache"),
    ]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    for compiled in root.rglob("*.py[co]"):
        if compiled.is_file():
            compiled.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class GateReport:
    passed: bool
    verdicts: dict[str, str]
    constraints: tuple[str, ...]
    verified_drop_bits: float
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "verdicts": dict(sorted(self.verdicts.items())),
            "constraints": list(self.constraints),
            "verified_drop_bits": round(self.verified_drop_bits, 6),
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class GaugeRunReport:
    passed: bool
    eta_cog: float
    verified_drop_bits: float
    gates: dict[str, str]
    constraints: tuple[str, ...]
    preflight_probe: bool
    probe_operators: tuple[str, ...]
    directive: str
    backend: str
    program: str
    violation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "eta_cog": round(self.eta_cog, 6),
            "verified_drop_bits": round(self.verified_drop_bits, 6),
            "gates": dict(sorted(self.gates.items())),
            "constraints": list(self.constraints),
            "preflight_probe": self.preflight_probe,
            "probe_operators": list(self.probe_operators),
            "directive": self.directive,
            "backend": self.backend,
            "program": self.program,
            "violation_count": self.violation_count,
        }


class MeteredGaugeVM(GaugeVM):
    """GaugeVM whose per-run cost includes router tokens and verifier work."""

    def __init__(self, root: str | Path, *, metered_cost: CognitiveStepCost, **kwargs: Any) -> None:
        super().__init__(root, **kwargs)
        self._metered_cost = metered_cost

    def _reset(self, name: str) -> None:
        super()._reset(name)
        self.execution.cost.discarded_tokens += self._metered_cost.discarded_tokens
        self.execution.cost.tool_cost += self._metered_cost.tool_cost
        self.execution.cost.pruned_branches += self._metered_cost.pruned_branches
        self.execution.cost.latency_ms += self._metered_cost.latency_ms


@lru_cache(maxsize=1)
def load_gauge_program() -> str:
    """Read the packaged copy of `examples/netron_plastic.gauge`."""
    return files("early_agi.bench").joinpath("netron_plastic.gauge").read_text(encoding="utf-8")


def gate_latency_ms(targets: Sequence[str]) -> float:
    """Deterministic latency charge used by both arms' `eta_cog` costs."""
    return round(8.0 + 4.0 * len(tuple(targets)), 3)


def attempt_cost(
    *,
    tokens: int,
    latency_ms: float,
    edit_count: int,
    target_count: int,
    attempt: int,
    gate_latency: float,
) -> CognitiveStepCost:
    """Build the same `CognitiveStepCost` shape for baseline and Gauge ticks."""
    return CognitiveStepCost(
        discarded_tokens=max(0, int(tokens)),
        tool_cost=1.0 + float(edit_count) + float(target_count),
        pruned_branches=max(0, int(attempt) - 1),
        latency_ms=float(latency_ms) + float(gate_latency),
    )


def baseline_eta(report: GateReport, cost: CognitiveStepCost) -> tuple[float, str]:
    """Compute the plain retry arm's eta_cog with the project's governor."""
    eta_cog, directive = ThermodynamicGovernor().evaluate(
        report.verified_drop_bits,
        cost,
        EntanglementReport(0.0, 0.0, 0, False),
    )
    return round(float(eta_cog), 6), directive.value


def _resolve(root: Path, relative: str) -> Path:
    if not is_safe_relative(relative):
        raise ValueError(f"unsafe workspace path: {relative!r}")
    base = root.resolve()
    target = (base / relative).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"workspace path escapes the fixture root: {relative!r}")
    return target


def write_workspace(fixture: BugFixture, root: str | Path) -> Path:
    """Materialise one deterministic bug repository."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    for relative, content in fixture.files:
        target = _resolve(base, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return base


def read_sources(root: str | Path) -> dict[str, str]:
    base = Path(root)
    sources: dict[str, str] = {}
    for path in sorted(base.rglob("*.py")):
        try:
            parts = set(path.relative_to(base).parts)
        except ValueError:
            continue
        if parts & _IGNORED_PARTS:
            continue
        sources[path.relative_to(base).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def capture_snapshot(
    root: str | Path,
    edits: Sequence[FileEdit],
    *,
    allowed_paths: Sequence[str],
) -> dict[str, bytes | None]:
    checked = validated_edits(edits, allowed_paths)
    base = Path(root)
    snapshot: dict[str, bytes | None] = {}
    for edit in checked:
        target = _resolve(base, edit.path)
        snapshot[edit.path] = target.read_bytes() if target.exists() else None
    return snapshot


def apply_edits(
    root: str | Path,
    edits: Sequence[FileEdit],
    *,
    allowed_paths: Sequence[str],
) -> None:
    checked = validated_edits(edits, allowed_paths)
    base = Path(root)
    for edit in checked:
        target = _resolve(base, edit.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")


def restore_snapshot(root: str | Path, snapshot: Mapping[str, bytes | None]) -> None:
    base = Path(root)
    for relative, content in snapshot.items():
        target = _resolve(base, relative)
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def evaluate_baseline_gates(
    root: str | Path,
    targets: Sequence[str],
    *,
    timeout_s: float,
) -> GateReport:
    """Run the same AST + pytest verifier used by the plain retry baseline."""
    base = Path(root)
    _clear_pycache(base)
    sources = read_sources(base)
    state: dict[str, Any] = {
        "repo_root": str(base),
        "paths": sorted(sources),
        "sources": sources,
        "targets": tuple(targets),
    }
    syntax = ast_syntax_gate("syntax_valid", bits=2.0).evaluate(state)
    tests = pytest_gate(
        "tests_passing",
        bits=4.0,
        targets=tuple(targets),
        timeout=timeout_s,
    ).evaluate(state)
    results = (syntax, tests)
    verdicts = {result.name: result.verdict for result in results}
    constraints = tuple(
        dict.fromkeys(
            str(result.constraint or f"assert {result.name}") for result in results if result.passed is False
        )
    )
    verified_drop = sum(_gate_bits(result) for result in results)
    passed = all(result.passed is True for result in results)
    return GateReport(
        passed=passed,
        verdicts=verdicts,
        constraints=constraints,
        verified_drop_bits=verified_drop,
        latency_ms=gate_latency_ms(targets),
    )


def _gate_bits(result: GateResult) -> float:
    if result.passed is True:
        return float(result.bits)
    if result.passed is False:
        return -float(result.bits)
    return 0.0


def run_netron_gauge(
    root: str | Path,
    *,
    sink: Any,
    backend: Any,
    metered_cost: CognitiveStepCost,
    timeout_s: float,
    program_source: str | None = None,
) -> GaugeRunReport:
    """Run `netron_plastic.gauge` with bounded-chi and real Gauge gates."""
    source = program_source or load_gauge_program()
    _clear_pycache(Path(root))
    settings = Settings.from_env(
        {
            "EARLY_AGI_BACKEND": "bounded_chi",
            "EARLY_AGI_TIMEOUT": str(float(timeout_s)),
            "EARLY_AGI_SINK": ":memory:",
            "EARLY_AGI_MERA": ":memory:",
            "EARLY_AGI_DIRECTIVE_DB": ":memory:",
        },
        load_dotenv=False,
    )
    vm = MeteredGaugeVM(
        root,
        metered_cost=metered_cost,
        settings=settings,
        sink=sink,
        backend=backend,
        registry=build_default_registry(root),
    )
    result = vm.run(source, name="netron_plastic.gauge")
    data = result.as_dict()
    gates = {str(name): str(verdict) for name, verdict in dict(data.get("gates") or {}).items()}
    tick = data.get("tick") if isinstance(data.get("tick"), Mapping) else {}
    governor = data.get("governor") if isinstance(data.get("governor"), Mapping) else {}
    plasticity = data.get("plasticity") if isinstance(data.get("plasticity"), Mapping) else {}
    manifolds = data.get("manifolds") if isinstance(data.get("manifolds"), Mapping) else {}
    fix = manifolds.get("fix") if isinstance(manifolds.get("fix"), Mapping) else {}
    eta_cog = float(tick.get("eta_cog") or governor.get("eta_cog") or 0.0)
    verified_drop = float(tick.get("verified_drop_bits") or 0.0)
    constraints = _dedupe_strings(
        [
            *(str(item) for item in (plasticity.get("distilled_constraints") or [])),
            *(str(item) for item in (fix.get("active_constraints") or [])),
        ]
    )
    if not constraints:
        constraints = tuple(f"assert {name}" for name, verdict in sorted(gates.items()) if verdict == "fail")
    preflight = (
        plasticity.get("preflight_probe") if isinstance(plasticity.get("preflight_probe"), Mapping) else {}
    )
    probe_operators = tuple(str(item) for item in (preflight.get("operators") or []))
    trace_probe = any(
        str(item.get("op")) == "EPISTEMIC_PROBE"
        for item in (data.get("trace") or [])
        if isinstance(item, Mapping)
    )
    violations = data.get("violations") or []
    violation_count = len(violations) if isinstance(violations, list) else 0
    passed = (
        bool(data.get("ok"))
        and bool(gates)
        and all(verdict == "pass" for verdict in gates.values())
        and violation_count == 0
    )
    return GaugeRunReport(
        passed=passed,
        eta_cog=round(eta_cog, 6),
        verified_drop_bits=round(verified_drop, 6),
        gates=gates,
        constraints=constraints,
        preflight_probe=bool(preflight or trace_probe),
        probe_operators=probe_operators,
        directive=str(governor.get("directive") or tick.get("directive") or ""),
        backend=str(data.get("backend_impl") or data.get("backend") or getattr(backend, "name", "")),
        program=str(data.get("program") or "netron_plastic.gauge"),
        violation_count=violation_count,
    )


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value)[:240] for value in values if str(value)))
