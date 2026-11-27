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
        result = vm.run(SOURCE, name="race.gauge")
        assert result.ok
        assert result.plasticity["branch_race"]["ok"]
        assert result.plasticity["branch_race"]["network_calls"] == 0
        assert vm.execution.variables["winner"]
