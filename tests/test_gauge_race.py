"""Gauge race syntax, safety checking and offline execution."""

import pytest

from early_agi.backends.bounded_chi import BoundedChiBackend
from early_agi.bench.fixtures import builtin_fixtures
from early_agi.bench.gates import write_workspace
from early_agi.config import Settings
from early_agi.core.branch_race import RaceConfig, RaceFixtureSpec
from early_agi.core.cold_sink import ColdSink
from early_agi.gauge.checker import check_program
from early_agi.gauge.compiler import compile_program
from early_agi.gauge.parser import parse_program
from early_agi.gauge.vm import GaugeVM
from early_agi.operators.registry import build_default_registry

SOURCE = """
scale L0 -> L2

manifold fix : PatchSpace {
}

gate syntax_valid(): true
gate tests_passing(): true

extern op branch_a() deforms(none) commutes(none) after()
extern op branch_b() deforms(none) commutes(none) after()

govern eta_min = 0.15, chi_max = 4 {
    on frustration {
        race [branch_a, branch_b] under [syntax_valid, tests_passing] -> winner
    }
    evolve fix under [syntax_valid, tests_passing]
    collapse(fix) -> sink
}
"""


def test_race_lowers_to_bytecode():
    assert check_program(parse_program(SOURCE)).ok
    assert any(
        child.op == "RACE_BRANCHES"
        for instruction in compile_program(SOURCE).instructions()
        for child in instruction.body
    )


@pytest.mark.parametrize("replacement", ["[branch_a]", "[branch_a, branch_a]", "[branch_a, missing]"])
def test_race_rejects_unsafe_branches(replacement):
    source = SOURCE.replace("[branch_a, branch_b]", replacement)
    assert not check_program(parse_program(source)).ok


def test_race_vm_runs_offline(tmp_path):
    fixture = builtin_fixtures()[0]
    root = tmp_path / "workspace"
    write_workspace(fixture, root)
    settings = Settings.from_env(
        {
            "EARLY_AGI_BACKEND": "bounded_chi",
            "EARLY_AGI_TIMEOUT": "30",
            "EARLY_AGI_SINK": ":memory:",
            "EARLY_AGI_MERA": ":memory:",
            "EARLY_AGI_DIRECTIVE_DB": ":memory:",
        },
        load_dotenv=False,
    )
    config = RaceConfig(
        fixture=RaceFixtureSpec.from_fixture(fixture),
        branches=("branch_a", "branch_b"),
        executor="thread",
        mera_path=":memory:",
    )
    with ColdSink(":memory:") as sink:
        vm = GaugeVM(
            root,
            settings=settings,
            sink=sink,
            backend=BoundedChiBackend(),
            registry=build_default_registry(root),
            race_config=config,
        )
        vm.traps_runner.is_due = lambda event: True
        vm.kernel.governor.recall_streak = 1
        result = vm.run(SOURCE, name="race.gauge")
        assert result.ok
        assert result.plasticity["branch_race"]["ok"]
        assert result.plasticity["branch_race"]["network_calls"] == 0
        assert len(result.races) == 1
        assert result.races[0]["winner_hash"] == vm.execution.variables["winner"]
        assert result.as_dict()["races"] == result.races
        assert vm.execution.variables["winner"]


def test_public_race_example_checks_and_compiles():
    from pathlib import Path

    source = Path("examples/netron_race.gauge").read_text(encoding="utf-8")
    assert check_program(parse_program(source)).ok
    assert "RACE_BRANCHES" in str(compile_program(source).as_dict())


def test_no_race_report_preserves_legacy_json():
    from early_agi.gauge.results import GaugeRunResult

    result = GaugeRunResult("halted", "empty", ".")
    assert result.races == []
    assert "races" not in result.as_dict()


def test_due_trap_without_stall_does_not_fork():
    from types import SimpleNamespace

    from early_agi.gauge.instruction_parts.race import _RaceOps
    from early_agi.gauge.ir import Instruction

    runner = _RaceOps()
    runner.vm = SimpleNamespace(
        traps_runner=SimpleNamespace(is_due=lambda event: True),
        kernel=SimpleNamespace(governor=SimpleNamespace(recall_streak=0)),
    )
    outcome = runner._op_race_branches({}, Instruction("RACE_BRANCHES"))
    assert outcome["skipped"]
    assert "stalled progress" in outcome["reason"]
