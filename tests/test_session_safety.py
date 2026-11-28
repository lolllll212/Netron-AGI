from __future__ import annotations

from pathlib import Path

import pytest

from early_agi.core.cold_sink import ColdSink
from early_agi.core.governor import ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel
from early_agi.llm.router import Edit, ScriptedRouter, TurnProposal, parse_turn
from early_agi.memory.mera_graph import MeraGraph
from early_agi.memory.reflex import ReflexLibrary
from early_agi.operators.dark_state import DarkStateGate
from early_agi.operators.registry import (
    OperatorSpec,
    WriteLocked,
    build_default_registry,
)
from early_agi.session import AgentSession

BROKEN = "def mul(a, b):\n    return a + b\n"
FIXED = "def mul(a, b):\n    return a * b\n"


def workspace(root: Path) -> None:
    (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "b.py").write_text("VALUE = 2\n", encoding="utf-8")


def kernel(sink: ColdSink, **kwargs) -> EarlyAGIKernel:
    options = {"chi_max": 2, "lock_writes_on_shift": False}
    options.update(kwargs)
    return EarlyAGIKernel(sink, **options)


def test_session_enforces_brief_file_allowlist_even_for_scripted_router(tmp_path: Path) -> None:
    workspace(tmp_path)
    sink = ColdSink()
    router = ScriptedRouter([TurnProposal(edits=(Edit("b.py", "VALUE = 99\n"),))])
    session = AgentSession(tmp_path, kernel(sink), router, max_brief_files=1)

    outcome = session.turn("edit outside the brief")

    assert outcome.applied == ()
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert any("allowlist" in str(item.get("reason", "")) for item in outcome.rejected)


def test_parse_turn_honours_the_allowlist_for_live_routers() -> None:
    proposal = parse_turn(
        {
            "edits": [
                {"path": "a.py", "content": "A\n"},
                {"path": "b.py", "content": "B\n"},
            ]
        },
        allowed_paths=["a.py"],
    )
    assert proposal.paths == ("a.py",)
    assert len(proposal.rejected) == 1


def test_partial_edit_batch_is_rolled_back_transactionally(tmp_path: Path) -> None:
    workspace(tmp_path)
    registry = build_default_registry(tmp_path)

    def failing_write(path: str, content: str, *, expected: str | bytes | None = None):
        if path == "a.py":
            (tmp_path / path).write_text(content, encoding="utf-8")
            return {"path": path, "written": True}
        raise FileExistsError("simulated write failure")

    registry.replace(
        OperatorSpec(
            name="write_file",
            fn=failing_write,
            kind="WRITE",
            reads=frozenset({"{path}"}),
            writes=frozenset({"{path}"}),
            cost=1.0,
        )
    )
    sink = ColdSink()
    router = ScriptedRouter([TurnProposal(edits=(Edit("a.py", "CHANGED\n"), Edit("b.py", "CHANGED\n")))])
    session = AgentSession(tmp_path, kernel(sink, registry=registry), router)

    outcome = session.turn("partial batch")

    assert outcome.status == "write_failed"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert sink.count("write_failure") == 1
    assert sink.count("applied_edit") == 0
    assert sink.count("file_snapshot") == 0


def test_non_utf8_source_does_not_crash_the_brief(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"VALUE = b'\xff'\n")
    session = AgentSession(tmp_path, kernel(ColdSink()), ScriptedRouter())

    outcome = session.turn("inspect binary source")

    assert outcome.status == "no_change"


def test_write_lock_is_enforced_at_the_registry_write_boundary(tmp_path: Path) -> None:
    workspace(tmp_path)
    registry = build_default_registry(tmp_path)
    sink = ColdSink()
    current_kernel = EarlyAGIKernel(
        sink,
        registry=registry,
        governor=ThermodynamicGovernor(eta_high=1e9),
        gates=[],
        chi_max=2,
    )
    session = AgentSession(tmp_path, current_kernel, ScriptedRouter([TurnProposal()]))
    session.turn("force a coordinate shift")
    assert current_kernel.writes_locked

    with pytest.raises(WriteLocked):
        session._write([Edit("a.py", "VALUE = 3\n")], {"a.py": b"VALUE = 1\n"})
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_gate_state_is_reconciled_after_recall_and_revert(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(BROKEN, encoding="utf-8")
    sink = ColdSink()
    gate = DarkStateGate(
        "multiplies",
        lambda state: "*" in (tmp_path / "calc.py").read_text(encoding="utf-8"),
        4.0,
    )
    current_kernel = kernel(sink, gates=[gate])
    router = ScriptedRouter(
        [
            TurnProposal(edits=(Edit("calc.py", FIXED),)),
            TurnProposal(edits=(Edit("calc.py", BROKEN),)),
            TurnProposal(edits=(Edit("calc.py", FIXED + "# repeat\n"),)),
        ]
    )
    session = AgentSession(tmp_path, current_kernel, router)
    outcomes = [session.turn(str(index)) for index in range(3)]

    assert outcomes[0].verified_drop_bits == 4.0
    assert outcomes[1].directive == "RECALL_SINK"
    assert outcomes[1].reverted == ("calc.py",)
    assert outcomes[2].verified_drop_bits == 0.0
    assert current_kernel.satisfied_gates == {"multiplies"}


def test_reflex_matching_is_task_aware(tmp_path: Path) -> None:
    task = "make mul(2, 3) return 6"

    def learn(sink: ColdSink, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "calc.py").write_text(BROKEN, encoding="utf-8")
        gate = DarkStateGate(
            "multiplies", lambda state, r=root: "*" in (r / "calc.py").read_text(encoding="utf-8"), 4.0
        )
        current_kernel = kernel(sink, gates=[gate])
        router = ScriptedRouter([TurnProposal(edits=(Edit("calc.py", FIXED),))])
        AgentSession(root, current_kernel, router).turn(task)

    with ColdSink() as sink:
        learn(sink, tmp_path / "one")
        learn(sink, tmp_path / "two")
        library = ReflexLibrary(MeraGraph(sink, ":memory:"))
        assert library.compile(min_support=2).reflexes == 1

        root = tmp_path / "three"
        root.mkdir()
        (root / "calc.py").write_text(BROKEN, encoding="utf-8")

        assert library.match(root, task=task) is not None
        assert library.match(root, task="do something completely unrelated") is None


def test_deep_tier_tokens_and_latency_are_charged_to_the_turn() -> None:
    from early_agi.core.kernel import ShiftArtifact

    sink = ColdSink()
    shift = ShiftArtifact(
        kind="invariant",
        payload={"check": "True"},
        rationale="deep tier reframed",
        tokens=31,
        latency_ms=12.5,
    )
    router = ScriptedRouter(executions=[TurnProposal()], shifts=[shift])
    current_kernel = EarlyAGIKernel(
        sink,
        governor=ThermodynamicGovernor(eta_high=1e9),
        gates=[],
        chi_max=2,
    )
    session = AgentSession(Path("."), current_kernel, router)
    session.turn("force shift")
    outcome = session.turn("reframe")

    assert outcome.status == "reframed"
    assert outcome.cost is not None
    assert outcome.cost["discarded_tokens"] == 31
    assert outcome.cost["latency_ms"] == 12.5
