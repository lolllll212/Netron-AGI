"""Three-pillar adaptability regression coverage."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from early_agi.config import Settings
from early_agi.core.adaptability import DirectivePool, local_benchmark
from early_agi.core.cold_sink import ColdSink
from early_agi.core.entropy_sensor import EntanglementReport
from early_agi.core.governor import CognitiveStepCost, Directive, ThermodynamicGovernor
from early_agi.core.kernel import EarlyAGIKernel
from early_agi.core.plasticity import HomeostaticRegulator, UserCoAdapter
from early_agi.gauge.ir import Instruction
from early_agi.gauge.runtime import ManifoldState
from early_agi.gauge.synthesizer import Synthesizer
from early_agi.gauge.vm import GaugeVM
from early_agi.llm.router import EXECUTOR_SYSTEM, DualTierRouter, Edit, ScriptedRouter, TurnProposal
from early_agi.memory.mera_graph import MeraGraph
from early_agi.operators.dark_state import DarkStateGate, pytest_gate, run_z3_smt2
from early_agi.operators.registry import WriteLocked, build_default_registry
from early_agi.session import AgentSession


def area_law() -> EntanglementReport:
    return EntanglementReport(0.2, 0.0, 1, False)


def workspace(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "b.py").write_text("B = 1\n", encoding="utf-8")


def passing_gate() -> DarkStateGate:
    return DarkStateGate("always_pass", lambda state: True, bits=2.0)


def test_low_surprise_tightens_toward_crystallize() -> None:
    regulator = HomeostaticRegulator(base_eta_min=0.20, base_eta_high=1.20, base_tau=0.15)
    regulator.observe_outcome(predicted_drop=2.0, verified_drop=2.0)
    thresholds = regulator.modulate(analogy_similarity=1.0, risk_level=0.5)

    assert thresholds.novelty_score == 0.0
    assert thresholds.prefer_crystallize is True
    assert thresholds.exploration_patience == 0
    assert thresholds.eta_high < 1.20
    assert thresholds.tau < 0.15


def test_high_surprise_grants_exploration_patience() -> None:
    governor = ThermodynamicGovernor()
    governor.enable_plasticity(base_eta_min=0.20, base_eta_high=1.20, ema_alpha=0.50)
    governor.modulate(0.0, 0.8, predicted_drop=0.0, verified_drop=3.0)

    assert governor.exploration_budget > 0
    governor.clear_probe()
    _, directive = governor.evaluate(-1.0, CognitiveStepCost(tool_cost=1.0), area_law())
    assert directive is Directive.SHIFT_COORDINATE
    assert governor.exploration_budget >= 0


def test_probe_directive_allows_only_read_and_verify_ops(tmp_path: Path) -> None:
    workspace(tmp_path)
    registry = build_default_registry(tmp_path)
    probe = registry.epistemic_probe(
        [
            ("read_file", {"path": "a.py"}),
            ("write_file", {"path": "blocked.py", "content": "no"}),
            ("run_pytest", {"paths": ("__missing__.py",), "timeout": 30.0}),
        ]
    )
    assert probe["results"][0]["ok"] is True
    assert probe["results"][1]["skipped"] == "mutating operator"
    assert not (tmp_path / "blocked.py").exists()

    with ColdSink() as sink:
        kernel = EarlyAGIKernel(sink, gates=[], registry=registry, chi_max=2, plasticity=True)
        kernel.engage_probe_lock()
        with pytest.raises(WriteLocked):
            registry.apply("write_file", path="blocked.py", content="no")
        kernel.release_probe_lock()
        registry.apply("write_file", path="allowed.py", content="yes")
        assert (tmp_path / "allowed.py").exists()


def test_pytest_failure_distills_one_minimal_assertion(tmp_path: Path) -> None:
    (tmp_path / "test_failure.py").write_text(
        "def test_failure():\n    assert 2 + 2 == 5\n",
        encoding="utf-8",
    )
    result = pytest_gate(timeout=30).evaluate({"repo_root": str(tmp_path)})

    assert result.passed is False
    assert result.constraint is not None
    assert result.constraint.startswith("assert ")
    assert "Traceback" not in result.detail
    assert "-----" not in result.detail


def test_z3_failure_distills_a_counterexample(monkeypatch) -> None:
    class FakeSolver:
        def set(self, *, timeout: int) -> None:
            del timeout

        def from_string(self, smt2: str) -> None:
            del smt2

        def check(self) -> str:
            return "sat"

        def model(self) -> str:
            return "[x = 3, y = 7]"

    fake = types.ModuleType("z3")
    fake.Solver = FakeSolver  # type: ignore[attr-defined]
    fake.Z3Exception = type("Z3Exception", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "z3", fake)

    result = run_z3_smt2("(check-sat)")

    assert result["passed"] is False
    assert result["model"] == {"x": "3", "y": "7"}
    assert result["constraint"] == "counterexample: x == 3, y == 7"
    assert "counterexample:" in result["detail"]


def test_failed_gate_appends_active_manifold_constraint(tmp_path: Path) -> None:
    vm = GaugeVM(tmp_path, settings=Settings.from_env({}, load_dotenv=False))
    vm.execution.scan_workspace()
    vm.execution.manifolds["fix"] = ManifoldState("fix", "PatchSpace", ["scope in repo.ast.modules"])
    vm.execution.gate_defs["always_fail"] = DarkStateGate("always_fail", lambda state: False, bits=1.0)

    vm.execute(Instruction("GATE_PROJECT", {"gates": ["always_fail"]}))

    manifold = vm.execution.manifolds["fix"]
    assert manifold.active_constraints()
    assert manifold.active_constraints()[-1].startswith("assert ")
    assert vm.execution.distilled_constraints == manifold.active_constraints()
    assert vm.result.manifolds["fix"]["active_constraints"] == manifold.active_constraints()


def test_stratified_replay_keeps_historical_gate_pass_rate() -> None:
    sink = ColdSink(":memory:")
    for task in ("alpha", "beta"):
        for index in range(3):
            sink.put(
                "uv_state",
                {
                    "state": {
                        "id": f"{task}-{index}",
                        "task": task,
                        "gates": {"g": "pass"},
                        "has_regressions": False,
                        "eta": 0.01,
                    },
                    "eta_cog": 0.01,
                    "drag": True,
                },
                scale=0,
                eta_cog=0.01,
            )
    synthesizer = Synthesizer(sink=sink, registry=None, replay_limit=4, eta_min=0.15)
    result = synthesizer.synthesize("stratified", "eta", {"eta": 0.0, "task": "new-task"})

    assert result.accepted is True
    assert {"alpha", "beta"} <= set(result.replay["strata"])
    assert result.replay["stratified"] is True
    assert result.replay["new_task_eta_improved"] is True
    assert result.replay["gate_rate_retained"] is True
    assert result.replay["candidate_gate_pass_rate"] >= result.replay["historical_gate_pass_rate"]
    sink.close()


def test_replay_rejects_candidate_that_degrades_historical_gates() -> None:
    sink = ColdSink(":memory:")
    for index in range(3):
        sink.put(
            "uv_state",
            {
                "state": {"id": f"old-{index}", "task": "old", "eta": 0.01, "has_regressions": False},
                "eta_cog": 0.01,
                "drag": True,
            },
            scale=0,
            eta_cog=0.01,
        )
    gate = DarkStateGate("no_new_gain", lambda state: "gauge_metric_gain" not in state, bits=1.0)
    synthesizer = Synthesizer(sink=sink, registry=None, gates=(gate,), replay_limit=3)
    result = synthesizer.synthesize("regressing", "eta", {"eta": 0.0, "task": "new-task"})

    assert result.accepted is False
    assert result.replay["gate_rate_retained"] is False
    assert "historical DarkStateGate pass rate degraded" in result.replay["detail"]
    sink.close()


def test_every_l1_l2_node_has_signature_and_guidance() -> None:
    sink = ColdSink(":memory:")
    graph = MeraGraph(sink, ":memory:")
    for index in range(3):
        sink.put(
            "uv_state",
            {
                "state": {
                    "files": [f"m{index}.py"],
                    "sources": {f"m{index}.py": "import os\ndef f(): pass"},
                },
                "gates": {"g1": "pass", "g2": "pass"},
                "directive": "CRYSTALLIZE",
                "index": index,
            },
            scale=0,
            eta_cog=2.0,
        )
    report = graph.coarse_grain(scale=0)

    assert report.guidance >= 1
    high_nodes = [node for node in graph.nodes() if node["scale"] >= 1]
    assert high_nodes
    assert all("structural_signature" in node["payload"] for node in high_nodes)
    assert all("decomposition_strategy" in node["payload"] for node in high_nodes)
    assert any(node["kind"] == "guidance" for node in high_nodes)
    graph.close()
    sink.close()


def test_structural_recall_is_zero_shot_across_file_names() -> None:
    sink = ColdSink(":memory:")
    graph = MeraGraph(sink, ":memory:")
    trained_matrix = np.eye(5) + 0.2 * np.ones((5, 5))
    trained = graph.encode_signature(
        trained_matrix, num_modules=5, num_imports=5, num_functions=10, num_gates=2
    )
    graph.register_analogy(
        "alpha-repo-strategy",
        trained,
        "module_cut",
        {
            "module_cut": "dependency_cut_5",
            "test_locking_order": ["read-only probe", "lock tests", "crystallize"],
            "eta_mean": 0.8,
        },
    )
    unseen_matrix = np.eye(12) * 7.0 + 0.2 * np.ones((12, 12))
    unseen = graph.encode_signature(
        unseen_matrix, num_modules=12, num_imports=12, num_functions=24, num_gates=2
    )

    match = graph.recall_decomposition_strategy(unseen, min_similarity=0.75)

    assert match is not None
    assert match["similarity"] >= 0.75
    assert match["module_cut"] == "dependency_cut_5"
    assert match["test_locking_order"][0] == "read-only probe"
    graph.close()
    sink.close()


def test_sqlite_directive_pool_evolves_from_cold_sink_post_mortems() -> None:
    sink = ColdSink(":memory:")
    for index in range(3):
        sink.put("uv_state", {"eta_cog": 0.8 + 0.01 * index}, scale=0, eta_cog=0.8)
    pool = DirectivePool(":memory:")
    router = DualTierRouter(object(), object(), directive_pool=pool)
    report = router.evolve_directives(sink)

    assert report["mutations"] >= 3
    assert router.active_directive().directive_id in report["active_ids"]
    assert "Live evolved adaptation policy" in router._messages(EXECUTOR_SYSTEM, {})[0]["content"]
    pool.close()
    sink.close()


def test_user_profile_and_session_enforce_autonomy_policies(tmp_path: Path) -> None:
    high = UserCoAdapter()
    for _ in range(4):
        high.record_turn(prompt_text="fix calc", edits_accepted=True)
    assert high.profile().multi_file_crystallization is True
    assert high.profile().patch_granularity == "bold"

    reverted = UserCoAdapter()
    for _ in range(2):
        reverted.record_turn(prompt_text="fix calc", edits_accepted=False, was_reverted=True)
    assert reverted.profile().epistemic_probe is True
    assert reverted.profile().patch_granularity == "surgical"

    root = tmp_path / "surgical"
    workspace(root)
    with ColdSink() as sink:
        registry = build_default_registry(root)
        kernel = EarlyAGIKernel(
            sink,
            gates=[passing_gate()],
            registry=registry,
            chi_max=2,
            plasticity=True,
            lock_writes_on_shift=False,
        )
        router = ScriptedRouter(
            [
                TurnProposal(
                    edits=(Edit("a.py", "A = 2\n"), Edit("b.py", "B = 2\n")),
                    rationale="two-file proposal",
                )
            ]
        )
        session = AgentSession(root, kernel, router, user_model=reverted, plasticity=True)
        outcome = session.turn("fix both files")

        assert len(outcome.applied) == 1
        assert outcome.rejected[-1]["reason"] == "autonomy policy: surgical single-file patch"
        assert outcome.probe is not None
        assert outcome.plasticity is not None
        assert outcome.plasticity["directive"] == Directive.EPISTEMIC_PROBE.value
        feedback = sink.records(kinds=("user_feedback",))[-1].payload
        assert feedback["prompt_length"] == 3
        assert feedback["manual_overrides"] == 0
        assert feedback["autonomy_policy"]["epistemic_probe"] is True


def test_high_acceptance_archives_multi_file_autonomy_and_directive(tmp_path: Path) -> None:
    root = tmp_path / "accepted"
    workspace(root)
    with ColdSink() as sink:
        registry = build_default_registry(root)
        kernel = EarlyAGIKernel(
            sink,
            gates=[passing_gate()],
            registry=registry,
            chi_max=2,
            plasticity=True,
            lock_writes_on_shift=False,
        )
        router = ScriptedRouter(
            [TurnProposal(edits=(Edit("calc.py", f"VALUE = {index}\n"),)) for index in range(2, 6)]
        )
        session = AgentSession(root, kernel, router, plasticity=True)
        for index in range(4):
            outcome = session.turn(f"fix calc {index}")
            assert outcome.applied

        feedback = sink.records(kinds=("user_feedback",))[-1].payload
        assert feedback["autonomy_policy"]["multi_file_crystallization"] is True
        assert feedback["adaptation_directive"]["patch_granularity"] in {"surgical", "module"}


def test_benchmark_smoke_has_three_curves() -> None:
    assert set(local_benchmark()) == {
        "discovery@k",
        "zero_regression_retention",
        "cold_start_eta_cog",
    }
