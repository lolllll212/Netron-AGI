"""Static Gauge checks: ordering, no-leak, reversibility, and CLI report."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from _gauge_fixtures import REPO_ROOT, SOURCE
from early_agi.gauge import check_program

MINIMAL = """scale L0 -> L2
manifold patch_ensemble : PatchSpace {}
gate syntax_valid(p) : ast_check(p.files) == true
extern op capture_baseline(m) deforms(repo_state) commutes(none)
reversible transform module_cut(p : PatchSpace) -> CutGraph {
    reencode(p, basis = dependency_cut)
}
govern eta_min = 0.25, chi_max = 4 {
    evolve patch_ensemble under [syntax_valid]
    resolved_patch = collapse(patch_ensemble) -> sink
}
"""


def _messages(report) -> str:
    return "\n".join(issue.message.casefold() for issue in report.errors)


def test_canonical_example_is_statically_valid() -> None:
    report = check_program(SOURCE)
    assert report.ok
    assert not report.errors


def test_after_graph_cycles_are_rejected() -> None:
    cycle = MINIMAL.replace(
        "extern op capture_baseline(m) deforms(repo_state) commutes(none)",
        "extern op a(m) deforms(repo_state) after(b)\nextern op b(m) deforms(repo_state) after(a)",
    )
    report = check_program(cycle)
    assert not report.ok
    assert "cycle" in _messages(report)


def test_uncollapsed_manifold_leaks_are_rejected() -> None:
    leak = "\n".join(line for line in SOURCE.splitlines() if "collapse(" not in line)
    report = check_program(leak)
    assert not report.ok
    assert "leaks uncollapsed" in _messages(report)


def test_destructive_statements_inside_reversible_transform_are_rejected() -> None:
    irreversible_call = MINIMAL.replace(
        "reencode(p, basis = dependency_cut)",
        "write_file(p, basis = dependency_cut)",
    )
    call_report = check_program(irreversible_call)
    assert not call_report.ok
    assert "irreversible operation" in _messages(call_report)

    irreversible_stmt = MINIMAL.replace(
        "reencode(p, basis = dependency_cut)",
        "recall sink where scale == L0",
    )
    stmt_report = check_program(irreversible_stmt)
    assert not stmt_report.ok
    assert "irreversible statement" in _messages(stmt_report)

    assignment = MINIMAL.replace("reencode(p, basis = dependency_cut)", "q = p")
    assign_report = check_program(assignment)
    assert not assign_report.ok
    assert "cannot assign" in _messages(assign_report)


def test_cli_check_returns_machine_readable_report() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(REPO_ROOT), *sys.path, env.get("PYTHONPATH", "")] if part
    )
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
    assert payload["errors"] == []
