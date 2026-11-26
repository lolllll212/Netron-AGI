from __future__ import annotations

from pathlib import Path

from early_agi.core.cold_sink import ColdSink
from early_agi.core.governor import ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel
from early_agi.llm.router import Edit, ScriptedRouter, TurnProposal
from early_agi.operators.dark_state import DarkStateGate, ast_syntax_gate
from early_agi.session import AgentSession


def workspace(root: Path) -> None:
    (root / "calc.py").write_text("def mul(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )


def multiplies(root: Path) -> DarkStateGate:
    return DarkStateGate(
        "multiplies",
        lambda state: "*" in (root / "calc.py").read_text(encoding="utf-8"),
        4.0,
    )


def test_edit_that_satisfies_the_gate_crystallises(tmp_path: Path) -> None:
    workspace(tmp_path)
    sink = ColdSink()
    kernel = EarlyAGIKernel(
        sink,
        gates=[ast_syntax_gate(bits=2.0), multiplies(tmp_path)],
        chi_max=2,
    )
    router = ScriptedRouter(
        [
            TurnProposal(
                edits=(Edit("calc.py", "def mul(a, b):\n    return a * b\n"),),
                rationale="multiply, do not add",
            )
        ]
    )
    session = AgentSession(tmp_path, kernel, router)

    outcome = session.turn("make mul multiply")

    assert outcome.status == "edited"
    assert outcome.directive == "CRYSTALLIZE"
    assert outcome.verified_drop_bits == 6.0
    assert outcome.applied == ("calc.py",)
    assert "*" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert sink.count("file_snapshot") == 1


def test_shift_locks_writes_and_routes_to_the_reasoner(tmp_path: Path) -> None:
    workspace(tmp_path)
    kernel = EarlyAGIKernel(
        ColdSink(),
        governor=ThermodynamicGovernor(eta_high=1e9),  # anything but perfect yield drags
        gates=[ast_syntax_gate()],
        chi_max=2,
    )
    router = ScriptedRouter(
        executions=[TurnProposal()],
        shifts=[{"kind": "invariant", "payload": {"check": "True"}, "rationale": "pin it"}],
    )
    session = AgentSession(tmp_path, kernel, router)

    first = session.turn("do nothing useful")
    assert first.directive == "SHIFT_COORDINATE"
    assert kernel.writes_locked is True

    second = session.turn("reframe the problem")
    assert second.status == "reframed"
    assert second.hash != first.hash
    assert "reason" in router.calls


def test_second_stalled_edit_is_reverted(tmp_path: Path) -> None:
    workspace(tmp_path)
    kernel = EarlyAGIKernel(
        ColdSink(),
        gates=[ast_syntax_gate(bits=2.0)],
        chi_max=2,
        lock_writes_on_shift=False,  # aggressive mode: edits keep flowing during a stall
    )
    first_body = "def mul(a, b):\n    return a + b  # note\n"
    second_body = "def mul(a, b):\n    return a + b  # another note\n"
    router = ScriptedRouter(
        [
            TurnProposal(edits=(Edit("calc.py", first_body),)),
            TurnProposal(),
            TurnProposal(edits=(Edit("calc.py", second_body),)),
        ]
    )
    session = AgentSession(tmp_path, kernel, router)

    session.turn("annotate")
    session.turn("wait")
    outcome = session.turn("annotate again")

    assert outcome.directive == "RECALL_SINK"
    assert outcome.reverted == ("calc.py",)
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == first_body
