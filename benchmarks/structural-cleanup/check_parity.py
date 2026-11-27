#!/usr/bin/env python3
"""Verify an offline A/B report against the structural-cleanup baseline.

Usage::

    python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --compact \\
      --output /tmp/after.json
    python benchmarks/structural-cleanup/check_parity.py /tmp/after.json

The comparison is exact: the full JSON report must be byte-for-report equal to
the captured baseline, and the locked acceptance metrics are checked again by
name so a renamed or reshaped curve cannot silently pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASELINE = Path(__file__).with_name("baseline-report.json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(report: dict[str, Any], baseline: dict[str, Any]) -> None:
    assert report == baseline, "the full offline A/B report differs from the baseline"
    assert report["offline"] is True
    assert report["network_calls"] == 0
    assert report["summary"]["ok"] is True

    checks = {
        "eta_cog": (
            report["arms"]["netron"]["eta_cog_curve"],
            report["arms"]["plain_retry"]["eta_cog_curve"],
            report["curves"]["discovery@k"]["netron_eta_cog"],
            report["curves"]["discovery@k"]["plain_retry_eta_cog"],
            report["curves"]["repeat_task_reflex_speedup"]["netron_eta_cog"],
            report["curves"]["repeat_task_reflex_speedup"]["plain_retry_eta_cog"],
        ),
        "success@k": (
            report["curves"]["discovery@k"]["attempt_grid"],
            report["curves"]["discovery@k"]["netron_cumulative_success"],
            report["curves"]["discovery@k"]["plain_retry_cumulative_success"],
        ),
        "rollback counts": (
            report["arms"]["netron"]["rollbacks"],
            report["arms"]["plain_retry"]["rollbacks"],
            report["curves"]["repeat_task_reflex_speedup"]["netron_rollbacks"],
            report["curves"]["repeat_task_reflex_speedup"]["plain_retry_rollbacks"],
            report["curves"]["repeat_task_reflex_speedup"]["rollbacks_avoided"],
        ),
        "repeat-task reflex speedups": (
            report["curves"]["repeat_task_reflex_speedup"]["attempt_speedup"],
            report["curves"]["repeat_task_reflex_speedup"]["token_speedup"],
            report["summary"]["repeat_token_speedup"],
            report["summary"]["reflex_hits"],
        ),
    }
    for name, actual in checks.items():
        expected = {
            "eta_cog": (
                baseline["arms"]["netron"]["eta_cog_curve"],
                baseline["arms"]["plain_retry"]["eta_cog_curve"],
                baseline["curves"]["discovery@k"]["netron_eta_cog"],
                baseline["curves"]["discovery@k"]["plain_retry_eta_cog"],
                baseline["curves"]["repeat_task_reflex_speedup"]["netron_eta_cog"],
                baseline["curves"]["repeat_task_reflex_speedup"]["plain_retry_eta_cog"],
            ),
            "success@k": (
                baseline["curves"]["discovery@k"]["attempt_grid"],
                baseline["curves"]["discovery@k"]["netron_cumulative_success"],
                baseline["curves"]["discovery@k"]["plain_retry_cumulative_success"],
            ),
            "rollback counts": (
                baseline["arms"]["netron"]["rollbacks"],
                baseline["arms"]["plain_retry"]["rollbacks"],
                baseline["curves"]["repeat_task_reflex_speedup"]["netron_rollbacks"],
                baseline["curves"]["repeat_task_reflex_speedup"]["plain_retry_rollbacks"],
                baseline["curves"]["repeat_task_reflex_speedup"]["rollbacks_avoided"],
            ),
            "repeat-task reflex speedups": (
                baseline["curves"]["repeat_task_reflex_speedup"]["attempt_speedup"],
                baseline["curves"]["repeat_task_reflex_speedup"]["token_speedup"],
                baseline["summary"]["repeat_token_speedup"],
                baseline["summary"]["reflex_hits"],
            ),
        }[name]
        assert actual == expected, f"{name} changed after refactoring"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="offline A/B JSON report to compare")
    args = parser.parse_args()
    report = _load(args.report)
    baseline = _load(BASELINE)
    _check(report, baseline)
    summary = report["summary"]
    print(
        "offline A/B parity OK: "
        f"eta_cog curves identical, success@k identical, rollbacks identical, "
        f"repeat-token speedup {summary['repeat_token_speedup']}, "
        f"reflex hits {summary['reflex_hits']}, network calls 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
