"""Shared source fixtures for the split Gauge test modules."""

from __future__ import annotations

from pathlib import Path

from early_agi.backends.classical_svd import ClassicalSVDBackend
from early_agi.core.entropy_sensor import EntanglementReport

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (REPO_ROOT / "examples" / "netron_solve.gauge").read_text(encoding="utf-8")


class VolumeBackend(ClassicalSVDBackend):
    """Force volume-law coupling and drag for deterministic trap tests."""

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
