"""Re-score an existing live ledger offline, preserving original governance evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .metrics import LIVE_SCORING, live_efficiency_summary, score_live_task


def rescore_report(report):
    if report.get("offline") is not False or report.get("schema_version") not in (1, 2, 3):
        raise ValueError("expected a schema 1/2/3 live evaluation report")
    result = copy.deepcopy(report)
    result.setdefault(
        "scoring_provenance",
        {
            "source_schema_version": report["schema_version"],
            "method": "offline reanalysis of recorded costs and gates; no new model calls or verification",
        },
    )
    result["schema_version"] = 3
    result["scoring"] = LIVE_SCORING
    for arm in result["arms"].values():
        arm["tasks"] = [score_live_task(task) for task in arm["tasks"]]
        arm.update(live_efficiency_summary(arm["tasks"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.resolve() == args.output.resolve():
        parser.error("use a different output path to preserve the original ledger")
    result = rescore_report(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
