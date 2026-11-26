from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from early_agi.backends.classical_svd import ClassicalSVDBackend
from early_agi.config import Settings
from early_agi.core.entropy_sensor import EntanglementReport
from early_agi.gauge import check_program, compile_gauge, parse_gauge, run_gauge
from early_agi.gauge.vm import GaugeVM

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (REPO_ROOT / "examples" / "netron_solve.gauge").read_text(encoding="utf-8")


class VolumeBackend(ClassicalSVDBackend):
    def measure_entanglement(self, matrix, chi_max: int = 4, tau: float = 0.15):
        return EntanglementReport(4.0, 0.8, 8, True, 3.0, 0.2)


def all_opcodes(ir) -> set[str]:
    found: set[str] = set()

    def visit(instructions) -> None:
        for instruction in instructions:
            found.add(instruction.op)
            visit(instruction.body)

    visit(ir.instructions())
    return found


def test_example_parses_with_spans_and_declarations() -> None:
    program = parse_gauge(SOURCE)
    assert len(program.declarations) == 8
    assert program.span.line >= 1
    assert program.span.end_line >= 40
    assert [type(decl).__name__ for decl in program.declarations[:4]] == [
        "ScaleDecl",
        "ManifoldDecl",
        "GateDecl",
        "GateDecl",
    ]


def test_static_checker_enforces_algebra_reversibility_and_no_leak() -> None:
    report = check_program(SOURCE)
    assert report.ok
    assert not report.errors

    after = check_program(SOURCE.replace("after(capture_baseline)", "after(missing_op)"))
    assert not after.ok
    assert any("after" in issue.message for issue in after.errors)

    irreversible = check_program(
        SOURCE.replace("reencode(p, basis = dependency_cut)", "write_file(p, basis = dependency_cut)")
    )
    assert not irreversible.ok
    assert any("irreversible" in issue.message.casefold() for issue in irreversible.errors)

    leak = "\n".join(line for line in SOURCE.splitlines() if "collapse(" not in line)
    assert not check_program(leak).ok


def test_compiler_emits_required_hybrid_vm_opcodes() -> None:
    ir = compile_gauge(SOURCE)
    opcodes = all_opcodes(ir)
    required = {
        "SCALE_INIT",
        "MANIFOLD_INIT",
        "TRANSFORM_REV",
        "GATE_PROJECT",
        "OP_APPLY",
        "ENT_MEASURE",
        "GOVERN_CHECK",
        "COLLAPSE_SINK",
        "UNCALL_TRANSFORM",
        "RECALL_SINK",
        "SYNTHESIZE_OP",
    }
    assert required <= opcodes
    assert ir.metadata["op_order"] == ["capture_baseline", "apply_candidate"]
    assert ir.metadata["scales"] == ["L0", "L2"]


def test_vm_runs_example_against_real_subsystems(tmp_path: Path) -> None:
    settings = Settings.from_env({}, load_dotenv=False)
    result = run_gauge(SOURCE, tmp_path, settings=settings, name="netron_solve.gauge")

    assert result.status == "halted"
    assert result.ok
    assert not result.violations
    assert "patch_ensemble" in result.collapses
    assert result.manifolds["patch_ensemble"]["collapsed"] is True
    assert "EVOLVE" in {event["op"] for event in result.trace}
    assert result.tick["hash"]


def test_drag_trap_synthesis_requires_positive_efficiency_improvement(tmp_path: Path) -> None:
    settings = Settings.from_env({}, load_dotenv=False)
    vm = GaugeVM(tmp_path, settings=settings, backend=VolumeBackend())
    result = vm.run(SOURCE, name="volume-law.gauge")

    assert result.ok
    assert {"volume_law", "drag"} <= set(result.traps)
    assert len(result.synthesis) == 1
    synthesis = result.synthesis[0]
    assert synthesis["accepted"] is True
    assert synthesis["efficiency_improvement"] > 0.0
    assert synthesis["origin"] == "deterministic"
    assert vm.registry.operator("disentangle_cut").description.startswith("synthesized")


def test_cli_check_returns_machine_readable_report() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(part for part in [str(REPO_ROOT), env.get("PYTHONPATH", "")] if part)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "early_agi.gauge",
            "check",
            "examples/netron_solve.gauge",
            "--compact",
            "--no-dotenv",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["declarations"] == 8
