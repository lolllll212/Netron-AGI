"""Offline end-to-end demo: ``python -m early_agi.demo``.

Everything here runs with no network and no API key: the scripted router stands
in for the dual-tier clients, and the real verifiers (AST, pytest) run against a
throwaway repository.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .backends import available_backends, get_backend
from .config import Settings
from .core.cold_sink import ColdSink
from .core.coupling import Constraint, RepoScanner
from .core.governor import CognitiveStepCost, ThermodynamicGovernor
from .core.kernel import EarlyAGIKernel
from .llm.router import Edit, ScriptedRouter, TurnProposal
from .memory.mera_graph import MeraGraph
from .memory.reflex import ReflexLibrary
from .operators.dark_state import DarkStateGate, ast_syntax_gate, pytest_gate, verifier_availability
from .operators.registry import build_default_registry
from .session import AgentSession


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(label: str, payload: Any) -> None:
    print(f"{label:<42} {json.dumps(payload, sort_keys=True)}")


def regimes(settings: Settings | None = None) -> ColdSink:
    """Pillars 1-3 over the three physical regimes."""
    settings = settings or Settings.from_env()
    section("1. Kernel regimes (lossless sink + SVD sensor + eta_cog governor)")
    sink = ColdSink(settings.sink_path)
    kernel = EarlyAGIKernel(
        sink=sink,
        backend=get_backend(settings.backend),
        governor=ThermodynamicGovernor(eta_high=settings.eta_high, eta_min=settings.eta_min),
        gates=[
            DarkStateGate("syntax_ok", lambda state: bool(state.get("syntax_ok")), 2.0),
            DarkStateGate("tests_pass", lambda state: bool(state.get("tests_pass")), 4.0),
        ],
        chi_max=settings.chi_max,
        tau=settings.tau,
    )
    area_law = np.outer([1.0, 0.8, 0.2, 0.1], [1.0, 0.5, 0.2, 0.1])
    rng = np.random.default_rng(42)
    volume_law = rng.standard_normal((8, 8))

    first = kernel.tick(
        {"syntax_ok": True, "tests_pass": False},
        area_law,
        CognitiveStepCost(discarded_tokens=200, tool_cost=1.0),
        label="area-law insight",
    )
    show("area-law + gain ->", first.as_dict())

    second = kernel.tick(
        {"syntax_ok": True, "tests_pass": True},
        volume_law,
        CognitiveStepCost(discarded_tokens=600, tool_cost=2.0, pruned_branches=1),
        pruned_tails=[{"hypothesis": "patch_in_place_v1", "reason": "high_coupling"}],
        label="volume-law spaghetti",
    )
    show("volume-law ->", second.as_dict())
    show("writes isolated?", {"locked": kernel.writes_locked, "reason": kernel.lock_reason})

    held = kernel.tick(
        {"syntax_ok": True, "tests_pass": True},
        volume_law,
        CognitiveStepCost(discarded_tokens=1500, tool_cost=3.0, pruned_branches=2),
        label="executor tried to patch anyway",
    )
    show("patch attempt while locked ->", held.as_dict())

    reframed = kernel.tick(
        {"syntax_ok": True, "tests_pass": True},
        volume_law,
        CognitiveStepCost(tool_cost=1.0),
        shift_artifact={
            "kind": "decomposition",
            "payload": {"steps": ["split parser from resolver", "pin the resolver's invariant"]},
            "rationale": "the two halves only communicate through one dict",
        },
        label="deep reasoner reframed",
    )
    show("coordinate shift accepted ->", reframed.as_dict())
    show("cold sink", sink.stats())
    show("pruned branches recoverable", [record.payload for record in sink.recall("pruned_branch")])
    return sink


def repo_coupling(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    section("2. Coupling matrix from the import/call graph (this package)")
    root = Path(__file__).resolve().parent
    scanner = RepoScanner(root, max_files=settings.max_files)
    graph = scanner.scan()
    constraints = [
        Constraint(id="gate:syntax_ok", paths=(".",)),
        Constraint(id="gate:tests_pass", paths=("tests",)),
        Constraint(id="hypothesis:deterministic_kernel", paths=("core/",)),
    ]
    coupling = graph.coupling(constraints)
    show("incidence matrix", {"shape": list(coupling.matrix.shape), "density": round(coupling.density(), 4)})
    show("meta", coupling.meta)
    print("  " + coupling.summary(chi_max=2, tau=0.10))
    print("  most coupled entities:", coupling.top_rows(k=4))
    print("  constraint weight:", coupling.top_cols(k=4))


def commutators() -> None:
    section("3. Non-commuting operators and the write lock (Trap 2)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        registry = build_default_registry(root)
        names = ["read_file", "write_file", "ast_refactor", "run_pytest"]
        print(registry.as_table(names, path="mod.py"))
        print("\n  2.0 = write/write, 1.0 = write/read, 0.0 = commute")

        registry.isolate_writes(reason="SHIFT_COORDINATE")
        try:
            registry.apply("write_file", path="mod.py", content="VALUE = 2\n")
        except Exception as exc:
            print(f"\n  blocked while isolated: {type(exc).__name__}: {exc}")
        print(f"  read still allowed -> {registry.apply('read_file', path='mod.py').strip()!r}")
        registry.release_writes()

        registry.consume_turn()
        registry.apply("run_pytest", paths=())
        registry.apply("ast_refactor", path="mod.py", mapping={"VALUE": "TOTAL"})
        for violation in registry.audit_turn():
            print("  audit:", violation.describe())


def mera(sink: ColdSink, settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    section("4. MERA consolidation: L0 traces -> L1 rules")
    graph = MeraGraph(sink, settings.mera_path, backend=get_backend(settings.backend))
    report = graph.coarse_grain(scale=0, chi_max=4)
    show("coarse-grain report", report.as_dict())
    for rule in graph.rules(scale=1):
        show(
            "  rule",
            {
                "scale": rule["scale"],
                "support": rule["payload"]["support"],
                "pattern": rule["payload"]["pattern"],
            },
        )
    show("levels", graph.levels())


def _fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text("def mul(a, b):\n    return a + b  # broken\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )


def session_loop(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    section("5. Phase-1 loop: edit -> verify -> eta_cog -> directive")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture(root)
        sink = ColdSink(settings.sink_path)
        kernel = EarlyAGIKernel(
            sink=sink,
            backend=get_backend(settings.backend),
            governor=ThermodynamicGovernor(eta_high=settings.eta_high, eta_min=settings.eta_min),
            gates=[ast_syntax_gate(bits=2.0), pytest_gate(bits=4.0, timeout=120.0)],
            chi_max=settings.chi_max,
            tau=settings.tau,
        )
        router = ScriptedRouter(
            executions=[
                TurnProposal(
                    edits=(Edit("calc.py", "def mul(a, b):\n    return a * b\n"),),
                    rationale="operator precedence: multiply, do not add",
                ),
                TurnProposal(rationale="nothing left to change"),
            ],
            shifts=[
                {"kind": "invariant", "payload": {"check": "mul(2, 3) == 6"}, "rationale": "pin the contract"}
            ],
        )
        scanner = RepoScanner(root, max_files=settings.max_files)
        session = AgentSession(
            root,
            kernel,
            router,
            scanner=scanner,
            constraints=[Constraint(id="gate:tests_pass", paths=(".",))],
        )
        for _ in range(3):
            outcome = session.turn("make mul(2, 3) return 6")
            show(f"turn {outcome.turn} ({outcome.status}) ->", outcome.as_dict())
        show("verifier availability", verifier_availability())
        show("backends", {"available": list(available_backends())})


def _multiplies(root: Path):
    return lambda state: "*" in (root / "calc.py").read_text(encoding="utf-8")


def reflex_compilation(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    section("6. Phase-3 reflex compilation: a repeated win becomes a zero-LLM procedure")
    fix = "def mul(a, b):\n    return a * b\n"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        sink = ColdSink(settings.sink_path)
        for index in range(2):
            root = base / f"learn{index}"
            _fixture(root)
            kernel = EarlyAGIKernel(
                sink=sink,
                backend=get_backend(settings.backend),
                gates=[DarkStateGate("multiplies", _multiplies(root), 4.0)],
                chi_max=2,
            )
            router = ScriptedRouter(
                [TurnProposal(edits=(Edit("calc.py", fix),), rationale="multiply, do not add")]
            )
            outcome = AgentSession(root, kernel, router).turn("make mul(2, 3) return 6")
            show(
                f"  learning turn {index + 1}",
                {
                    "directive": outcome.directive,
                    "verified_bits": outcome.verified_drop_bits,
                    "applied": list(outcome.applied),
                },
            )

        library = ReflexLibrary(MeraGraph(sink, settings.mera_path, backend=get_backend(settings.backend)))
        show("compile report", library.compile(min_support=2).as_dict())
        for reflex in library.reflexes():
            show("  reflex", reflex.as_dict())

        root = base / "replay"
        _fixture(root)
        kernel = EarlyAGIKernel(
            sink=sink,
            backend=get_backend(settings.backend),
            gates=[DarkStateGate("multiplies", _multiplies(root), 4.0)],
            chi_max=2,
        )
        router = ScriptedRouter(strict=True)  # no script left: only a reflex can act
        outcome = AgentSession(root, kernel, router, reflexes=library).turn("make mul(2, 3) return 6")
        show(
            "  reflex turn",
            {
                "status": outcome.status,
                "directive": outcome.directive,
                "reflex": outcome.reflex,
                "applied": list(outcome.applied),
            },
        )
        show("  router calls (must be empty)", {"calls": router.calls})


def backend_showcase(settings: Settings | None = None) -> None:
    """Exercise the bounded-χ backend and its optional thermodynamic seam."""
    settings = settings or Settings.from_env()
    section("7. Backend seams: bounded-χ truncation and thermodynamic sampling")
    selected = get_backend(settings.backend)
    bounded = get_backend("bounded_chi")
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(8, 6))
    rank = min(settings.chi_max, *matrix.shape)
    factor = bounded.factorize(
        matrix,
        power_iterations=2,
        oversampling=2,
        seed=17,
        options={"chi_max": rank},
    )
    report = bounded.measure_entanglement(matrix, chi_max=rank, tau=settings.tau)
    couplings = np.array([[0.0, 1.0, 0.4], [1.0, 0.0, -0.2], [0.4, -0.2, 0.0]])
    sample = bounded.block_gibbs(
        couplings,
        biases=[0.1, 0.0, -0.1],
        beta=0.8,
        schedule={"n_warmup": 4, "n_samples": 4, "thinning": 1, "seed": 17},
        adapter="auto",
    )
    show(
        "backend selector",
        {
            "settings": settings.backend,
            "implementation": getattr(selected, "name", type(selected).__name__),
            "available": list(available_backends()),
        },
    )
    show("bounded-χ factorization", factor.as_dict())
    show("bounded-χ entanglement", report.as_dict())
    show("block-Gibbs sample", sample.as_dict())
    show("THRML adapter", bounded.thrml_adapter().as_dict())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="early-agi",
        description="Offline end-to-end demo for the early-AGI kernel and Gauge runtime.",
    )
    parser.add_argument(
        "--backend",
        choices=available_backends(),
        help="relaxation backend to use (default: EARLY_AGI_BACKEND or classical_svd)",
    )
    parser.add_argument("--no-dotenv", action="store_true", help="ignore a local .env file")
    args = parser.parse_args(argv)

    settings = Settings.from_env(load_dotenv=not args.no_dotenv)
    if args.backend:
        settings = replace(settings, backend=args.backend)
        settings.validate()
    print("early-agi demo | numpy + sqlite3 only | no network required")
    show("effective settings", settings.to_dict())
    sink = regimes(settings)
    repo_coupling(settings)
    commutators()
    mera(sink, settings)
    session_loop(settings)
    reflex_compilation(settings)
    backend_showcase(settings)
    print(
        "\nDone. Try --backend bounded_chi, or Gauge: "
        "python -m early_agi.gauge run examples/netron_solve.gauge --backend bounded_chi. "
        "Set EARLY_AGI_LLM_ENABLED=1 plus endpoint/model variables for live routing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
