from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from early_agi.operators.dark_state import (
    DarkStateGate,
    ast_syntax_gate,
    pytest_gate,
    run_pytest,
    verifier_availability,
    z3_gate,
)


def test_ast_gate_passes_and_fails(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    gate = ast_syntax_gate()
    assert gate.evaluate({"repo_root": str(tmp_path)}).passed is True

    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    result = gate.evaluate({"repo_root": str(tmp_path)})
    assert result.passed is False
    assert result.verdict == "fail"


def test_gate_without_sources_is_neutral(tmp_path: Path) -> None:
    result = ast_syntax_gate().evaluate({"repo_root": str(tmp_path)})
    assert result.passed is None
    assert result.verdict == "unverifiable"


def test_gate_exception_is_neutral_not_failure() -> None:
    def explode(state: object) -> bool:
        raise RuntimeError("tool vanished")

    assert DarkStateGate("boom", explode).evaluate({}).passed is None


def test_pytest_gate_reports_failure_and_pass(tmp_path: Path) -> None:
    gate = pytest_gate(timeout=120)
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert gate.evaluate({"repo_root": str(tmp_path)}).passed is True

    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert False\n", encoding="utf-8")
    assert gate.evaluate({"repo_root": str(tmp_path)}).passed is False


def test_pytest_without_tests_is_unverifiable(tmp_path: Path) -> None:
    assert run_pytest(tmp_path, timeout=120)["passed"] is None


def test_z3_gate_is_neutral_when_z3_is_missing() -> None:
    if importlib.util.find_spec("z3") is not None:
        pytest.skip("z3 installed; the missing-verifier path is not exercised")
    assert z3_gate(lambda state: True, name="proved").evaluate({}).passed is None


def test_availability_always_reports_ast() -> None:
    assert verifier_availability()["ast"] is True
