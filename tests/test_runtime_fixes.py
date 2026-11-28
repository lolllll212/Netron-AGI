from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from early_agi.backends.classical_svd import ClassicalSVDBackend
from early_agi.config import ConfigError, Settings
from early_agi.core.cold_sink import ColdSink, SinkError
from early_agi.llm.client import LLMReply, LLMUnavailable, extract_json
from early_agi.llm.router import DualTierRouter, parse_turn
from early_agi.memory.mera_graph import MeraGraph
from early_agi.operators.dark_state import run_z3_smt2
from early_agi.operators.registry import (
    VERIFY,
    WRITE,
    OperatorRegistry,
    OperatorSpec,
)


def test_noisy_json_array_and_usage_accounting() -> None:
    text = 'Sure! [{"path": "a.py", "content": "a"}, {"path": "b.py", "content": "b"}] trailing'
    data = extract_json(text)
    assert isinstance(data, list)
    assert len(data) == 2
    assert parse_turn(data, allowed_paths=["a.py", "b.py"]).paths == ("a.py", "b.py")
    reply = LLMReply(text="", model="fake", usage={"prompt_tokens": 7, "completion_tokens": 5})
    assert reply.tokens == 12


def test_settings_strict_validation_dotenv_and_runtime_wiring(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"EARLY_AGI_TAU": "not-a-number"}, load_dotenv=False)

    default = Settings.from_env({}, load_dotenv=False)
    assert default.llm_configured is False
    assert default.sink_path == ":memory:"
    with pytest.raises(LLMUnavailable):
        DualTierRouter.from_settings(default)

    configured = Settings.from_env(
        {"EARLY_AGI_FAST_MODEL": "fast", "EARLY_AGI_DEEP_MODEL": "deep"},
        load_dotenv=False,
    )
    assert configured.llm_configured is True

    env_file = tmp_path / ".env"
    env_file.write_text("EARLY_AGI_MAX_FILES=17\nEARLY_AGI_CHI_MAX=6\n", encoding="utf-8")
    settings = Settings.from_env({}, dotenv_path=env_file)
    assert settings.max_files == 17
    assert settings.chi_max == 6

    bundle = settings.build_runtime(tmp_path)
    assert bundle.kernel.chi_max == 6
    assert bundle.scanner.max_files == 17
    assert bundle.graph.backend is bundle.kernel.backend
    bundle.sink.close()


def test_cold_sink_preserves_metadata_commits_and_compression_flag(tmp_path: Path) -> None:
    database = tmp_path / "cold.db"
    with ColdSink(database, compress=True) as sink:
        root = sink.put("uv_state", {"x": 1}, parent_hash="root", eta_cog=1.0)
        duplicate = sink.put("uv_state", {"x": 1}, parent_hash="leaf", eta_cog=9.0)
        assert root == duplicate
        assert len(sink.commits(root)) == 2
        record = sink.get(root)
        assert record is not None
        assert record.eta_cog == 1.0  # first commit remains authoritative
        assert sink.commits(root)[1]["parent_hash"] == "leaf"

    with pytest.raises(SinkError):
        ColdSink(database, compress=False)


def test_mera_coarse_graining_uses_the_backend_contract() -> None:
    class SpyBackend(ClassicalSVDBackend):
        def __init__(self) -> None:
            self.truncate_calls = 0
            self.left_basis_calls = 0

        def truncate(self, matrix, chi_max: int = 4):
            self.truncate_calls += 1
            return super().truncate(matrix, chi_max=chi_max)

        def left_basis(self, matrix, k: int = 4):
            self.left_basis_calls += 1
            return super().left_basis(matrix, k=k)

    backend = SpyBackend()
    with ColdSink() as sink:
        for index in range(4):
            sink.put(
                "uv_state",
                {
                    "i": index,
                    "gates": {"g1": "pass", "g2": "pass" if index % 2 else "fail"},
                    "directive": "CRYSTALLIZE" if index < 2 else "RECALL_SINK",
                },
                eta_cog=1.0,
            )
        with MeraGraph(sink, ":memory:", backend=backend) as graph:
            report = graph.coarse_grain(scale=0, chi_max=2)
            assert report.records == 4
            assert backend.truncate_calls > 0
            assert backend.left_basis_calls > 0


def test_tri_state_verifiers_and_wildcard_write_race_audit() -> None:
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            name="unknown_verifier",
            fn=lambda: {"status": "unknown", "detail": "cannot decide"},
            kind=VERIFY,
            reads=frozenset({"*"}),
        )
    )
    registry.register(
        OperatorSpec(
            name="wildcard_write",
            fn=lambda: {"ok": True},
            kind=WRITE,
            writes=frozenset({"*"}),
        )
    )
    registry.register(
        OperatorSpec(
            name="scoped_write",
            fn=lambda: {"ok": True},
            kind=WRITE,
            writes=frozenset({"mod.py"}),
        )
    )
    registry.consume_turn()
    registry.apply("unknown_verifier")
    registry.apply("wildcard_write")
    registry.apply("scoped_write")

    assert registry.log[0].ok is None
    kinds = {violation.kind for violation in registry.audit_turn()}
    assert "write_race" in kinds
    assert "stale_verification" not in kinds


def test_z3_results_map_to_true_tri_state_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStatus:
        def __init__(self, text: str) -> None:
            self.text = text

        def __str__(self) -> str:
            return self.text

    class FakeSolver:
        def __init__(self) -> None:
            self.status = "unknown"

        def set(self, timeout: int) -> None:
            self.timeout = timeout

        def from_string(self, smt2: str) -> None:
            if "false" in smt2:
                self.status = "unsat"
            elif "true" in smt2:
                self.status = "sat"
            else:
                self.status = "unknown"

        def check(self) -> FakeStatus:
            return FakeStatus(self.status)

        def model(self) -> str:
            return "x = 1"

    fake_z3 = SimpleNamespace(Solver=FakeSolver, Z3Exception=type("Z3Exception", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "z3", fake_z3)

    assert run_z3_smt2("(assert false)")["passed"] is True
    assert run_z3_smt2("(assert true)")["passed"] is False
    assert run_z3_smt2("(assert mystery)")["passed"] is None


def test_shift_artifacts_are_semantically_verified() -> None:
    from early_agi.core.kernel import validate_shift_artifact

    assert validate_shift_artifact("invariant", {"check": "1 + 1 == 2"}, state={})[0]
    assert not validate_shift_artifact("invariant", {"check": "1 + 1 == 3"}, state={})[0]
    assert not validate_shift_artifact(
        "invariant", {"check": "__import__('os').system('echo unsafe')"}, state={}
    )[0]
    safe_source = "def pure(state, *, metric='eta'):\n    return dict(state)\n"
    assert validate_shift_artifact("new_operator", {"name": "pure", "source": safe_source}, state={"x": 1})[0]
    unsafe_source = "def mutate(state):\n    state['x'] = 2\n    return state\n"
    assert not validate_shift_artifact(
        "new_operator", {"name": "mutate", "source": unsafe_source}, state={"x": 1}
    )[0]


def test_run_shell_is_disabled_by_default(tmp_path: Path) -> None:
    from early_agi.operators.registry import build_default_registry

    registry = build_default_registry(tmp_path)
    with pytest.raises(PermissionError):
        registry.apply("run_shell", command="echo unsafe")


def test_coupling_scanner_resolves_qualified_calls_and_reports_truncation(tmp_path: Path) -> None:
    from early_agi.core.coupling import RepoScanner

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (package / "b.py").write_text(
        "from pkg.a import helper\n\n\ndef use():\n    return helper()\n", encoding="utf-8"
    )
    graph = RepoScanner(tmp_path).scan()
    assert graph.edges["pkg/b.py::use"]["pkg/a.py::helper"] >= 1
    assert graph.truncated is False

    truncated = RepoScanner(tmp_path, max_files=1).scan()
    assert truncated.truncated is True
