from __future__ import annotations

from pathlib import Path

from early_agi.config import Settings
from early_agi.gauge.checker import validate_program
from early_agi.gauge.compiler import compile_program
from early_agi.gauge.vm import run_gauge

EXAMPLE = Path(__file__).parents[1] / "examples" / "netron_plastic.gauge"


def _source() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def _all_ops(instructions: list[object]) -> set[str]:
    ops: set[str] = set()
    for instruction in instructions:
        ops.add(str(instruction.op))  # type: ignore[attr-defined]
        nested = getattr(instruction, "body", ())
        ops.update(_all_ops(list(nested)))
    return ops


def test_plastic_gauge_checks_and_compiles_new_syntax() -> None:
    source = _source()
    checked = validate_program(source)
    assert checked.report.ok
    assert checked.report.warnings == ()

    ir = compile_program(source)
    setup_ops = {instruction.op for instruction in ir.setup}
    body_ops = _all_ops(list(ir.body))
    assert "PLASTIC_GOVERN_BEGIN" in setup_ops
    assert "TRAP_HIGH_NOVELTY" in body_ops
    assert "PROBE" in body_ops
    assert "RECALL_ANALOGY" in body_ops


def test_plastic_gauge_runs_probes_and_reports_plasticity(tmp_path: Path) -> None:
    result = run_gauge(
        _source(),
        root=tmp_path,
        settings=Settings.from_env({}, load_dotenv=False),
    )

    assert result.ok is True
    assert result.violations == []
    trace_ops = [entry["op"] for entry in result.trace]
    assert "PLASTIC_GOVERN_BEGIN" in trace_ops
    assert "TRAP_HIGH_NOVELTY" in trace_ops
    assert "PROBE" in trace_ops
    assert "RECALL_ANALOGY" in trace_ops
    assert result.governor["directive"] == "EPISTEMIC_PROBE"
    assert result.plasticity["enabled"] is True
    assert result.plasticity["probe"]["operators"] == ["read_file", "inspect_deps"]
    assert all(item["ok"] for item in result.plasticity["probe"]["results"])
    assert result.plasticity["thresholds"]["should_probe_first"] is False
    assert result.plasticity["analogy"] is None or isinstance(result.plasticity["analogy"], dict)


def test_original_gauge_example_still_checks() -> None:
    source = (Path(__file__).parents[1] / "examples" / "netron_solve.gauge").read_text(encoding="utf-8")
    assert validate_program(source).report.ok
