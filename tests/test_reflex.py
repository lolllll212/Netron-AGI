from __future__ import annotations

from pathlib import Path

from early_agi.core.cold_sink import ColdSink
from early_agi.core.governor import ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel
from early_agi.llm.router import Edit, ScriptedRouter, TurnProposal
from early_agi.memory.mera_graph import MeraGraph
from early_agi.memory.reflex import ReflexLibrary, compile_reflexes
from early_agi.operators.dark_state import DarkStateGate
from early_agi.session import AgentSession

BROKEN = "def mul(a, b):\n    return a + b\n"
FIXED = "def mul(a, b):\n    return a * b\n"


def fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text(BROKEN, encoding="utf-8")


def multiplies(root: Path, bits: float = 4.0) -> DarkStateGate:
    return DarkStateGate(
        "multiplies",
        lambda state, root=root: "*" in (root / "calc.py").read_text(encoding="utf-8"),
        bits,
    )


def learn_once(sink: ColdSink, root: Path) -> None:
    fixture(root)
    kernel = EarlyAGIKernel(sink, gates=[multiplies(root)], chi_max=2)
    router = ScriptedRouter(
        [TurnProposal(edits=(Edit("calc.py", FIXED),), rationale="multiply, do not add")]
    )
    outcome = AgentSession(root, kernel, router).turn("make mul(2, 3) return 6")
    assert outcome.directive == "CRYSTALLIZE"
    assert outcome.applied == ("calc.py",)


def compiled_library(sink: ColdSink, *roots: Path) -> ReflexLibrary:
    for root in roots:
        learn_once(sink, root)
    library = ReflexLibrary(MeraGraph(sink, ":memory:"))
    library.compile(min_support=2)
    return library


def test_a_recurring_gated_win_compiles_into_a_reflex(tmp_path: Path) -> None:
    with ColdSink() as sink:
        learn_once(sink, tmp_path / "one")
        learn_once(sink, tmp_path / "two")

        library = ReflexLibrary(MeraGraph(sink, ":memory:"))
        report = library.compile(min_support=2)

        assert report.reflexes == 1
        (reflex,) = library.reflexes()
        assert reflex.support == 2
        assert reflex.expected_gates == ("multiplies",)
        assert reflex.edits[0].content == FIXED
        assert reflex.paths == ("calc.py",)
        assert sink.count("reflex") == 1


def test_a_single_win_is_not_enough_to_compile(tmp_path: Path) -> None:
    with ColdSink() as sink:
        learn_once(sink, tmp_path / "one")
        report = compile_reflexes(MeraGraph(sink, ":memory:"), min_support=2)

        assert report.reflexes == 0
        assert sink.count("reflex") == 0


def test_reflex_fires_without_touching_the_router(tmp_path: Path) -> None:
    with ColdSink() as sink:
        library = compiled_library(sink, tmp_path / "one", tmp_path / "two")
        root = tmp_path / "three"
        fixture(root)
        kernel = EarlyAGIKernel(sink, gates=[multiplies(root)], chi_max=2)
        router = ScriptedRouter(strict=True)  # any model call would raise

        outcome = AgentSession(root, kernel, router, reflexes=library).turn("make mul(2, 3) return 6")

        assert outcome.reflex.startswith("reflex/")
        assert outcome.directive == "CRYSTALLIZE"
        assert outcome.applied == ("calc.py",)
        assert router.calls == []
        assert (root / "calc.py").read_text(encoding="utf-8") == FIXED
        (reflex,) = library.reflexes()
        assert reflex.hits == 1


def test_reflex_refuses_a_pre_state_it_never_saw(tmp_path: Path) -> None:
    with ColdSink() as sink:
        library = compiled_library(sink, tmp_path / "one", tmp_path / "two")
        root = tmp_path / "three"
        fixture(root)
        (root / "calc.py").write_text("def mul(a, b):\n    return a - b\n", encoding="utf-8")
        kernel = EarlyAGIKernel(sink, gates=[multiplies(root)], chi_max=2)
        router = ScriptedRouter([TurnProposal(rationale="drifted pre-state")])

        outcome = AgentSession(root, kernel, router, reflexes=library).turn("make mul(2, 3) return 6")

        assert outcome.reflex == ""
        assert router.calls == ["execute"]


def test_reflex_is_retired_after_repeated_misses(tmp_path: Path) -> None:
    with ColdSink() as sink:
        library = compiled_library(sink, tmp_path / "one", tmp_path / "two")
        root = tmp_path / "three"
        fixture(root)
        (reflex,) = library.reflexes()
        assert library.match(root) is not None

        library.record(reflex, ok=False, parent_hash="tick-1")
        library.record(reflex, ok=False, parent_hash="tick-2")

        assert library.match(root) is None


def test_a_reflex_that_cannot_reproduce_its_effect_is_reverted(tmp_path: Path) -> None:
    with ColdSink() as sink:
        library = compiled_library(sink, tmp_path / "one", tmp_path / "two")
        root = tmp_path / "three"
        fixture(root)
        # The learned gate is no longer in the kernel, so the effect is unverifiable
        # and the replay must be undone rather than trusted.
        kernel = EarlyAGIKernel(
            sink,
            gates=[DarkStateGate("renamed_gate", lambda state: False, 1.0)],
            chi_max=2,
        )
        router = ScriptedRouter(strict=True)

        outcome = AgentSession(root, kernel, router, reflexes=library).turn("make mul(2, 3) return 6")

        assert outcome.reflex.startswith("reflex/")
        assert outcome.status == "reflex_reverted"
        assert outcome.reverted == ("calc.py",)
        assert (root / "calc.py").read_text(encoding="utf-8") == BROKEN
        (reflex,) = library.reflexes()
        assert reflex.misses == 1


def test_reverted_edits_are_never_learned(tmp_path: Path) -> None:
    with ColdSink() as sink:
        root = tmp_path / "repo"
        fixture(root)
        kernel = EarlyAGIKernel(
            sink,
            governor=ThermodynamicGovernor(stall_ticks=1),
            gates=[],
            chi_max=2,
        )
        router = ScriptedRouter([TurnProposal(edits=(Edit("calc.py", FIXED),))])

        outcome = AgentSession(root, kernel, router).turn("unverified edit")

        assert outcome.directive == "RECALL_SINK"
        assert outcome.reverted == ("calc.py",)
        assert (root / "calc.py").read_text(encoding="utf-8") == BROKEN
        assert sink.count("applied_edit") == 0
