from __future__ import annotations

from pathlib import Path

import pytest

from early_agi.operators.registry import WriteLocked, build_default_registry


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_commutator_matrix(workspace: Path) -> None:
    registry = build_default_registry(workspace)
    assert registry.commutator("read_file", "write_file", path="mod.py") == 1.0
    assert registry.commutator("write_file", "write_file", path="mod.py") == 2.0
    assert registry.commutator("read_file", "run_pytest", path="mod.py") == 0.0
    assert registry.commutes("read_file", "read_file", path="mod.py")


def test_write_isolation_blocks_mutation_only(workspace: Path) -> None:
    registry = build_default_registry(workspace)
    registry.isolate_writes(reason="SHIFT_COORDINATE")

    with pytest.raises(WriteLocked):
        registry.apply("write_file", path="mod.py", content="VALUE = 2\n")
    assert registry.apply("read_file", path="mod.py").strip() == "VALUE = 1"

    registry.release_writes()
    result = registry.apply("write_file", path="mod.py", content="VALUE = 2\n")
    assert result["bytes"] > 0
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_audit_flags_mutation_after_verification(workspace: Path) -> None:
    registry = build_default_registry(workspace)
    registry.apply("run_pytest", paths=())
    registry.apply("write_file", path="mod.py", content="VALUE = 3\n")

    kinds = {violation.kind for violation in registry.audit_turn()}
    assert "stale_verification" in kinds


def test_audit_ignores_verifications_that_failed(workspace: Path) -> None:
    (workspace / "test_mod.py").write_text(
        "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 99\n",
        encoding="utf-8",
    )
    registry = build_default_registry(workspace)
    registry.apply("run_pytest", paths=())
    registry.apply("write_file", path="mod.py", content="VALUE = 3\n")

    assert registry.audit_turn() == []


def test_ast_refactor_preserves_formatting(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text("def  foo(x):\n    return foo(x)\n", encoding="utf-8")
    registry = build_default_registry(tmp_path)

    result = registry.apply("ast_refactor", path="mod.py", mapping={"foo": "bar"})

    assert result["renames"] == 2
    assert path.read_text(encoding="utf-8") == "def  bar(x):\n    return bar(x)\n"


def test_write_file_expected_content_guard(workspace: Path) -> None:
    registry = build_default_registry(workspace)
    with pytest.raises(RuntimeError):
        registry.apply("write_file", path="mod.py", content="VALUE = 2\n", expected="STALE\n")
    assert (workspace / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_operators_cannot_escape_the_root(workspace: Path) -> None:
    registry = build_default_registry(workspace)
    with pytest.raises(PermissionError):
        registry.apply("read_file", path="../outside.py")
