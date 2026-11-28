"""Native RACE_BRANCHES opcode handler for thermodynamic counterfactual races."""

from __future__ import annotations

from typing import Any, cast

from early_agi.core.branch_race import RaceConfig, run_branch_race
from early_agi.gauge.ir import Instruction
from early_agi.gauge.results import GaugeRaceOutcome

__all__ = ["_RaceOps"]


class _RaceOps:
    """Fork scoped WAL branches when a volume-law/frustration trap fires."""

    def _op_race_branches(self, args: dict[str, Any], _instruction: Instruction) -> dict[str, Any]:
        branches = tuple(str(item) for item in args.get("branches", ()))
        gates = tuple(str(item) for item in args.get("gates", ()))
        winner = str(args.get("winner") or "winner")
        volume_due = bool(self.vm.traps_runner.is_due("volume_law"))
        frustration_due = bool(self.vm.traps_runner.is_due("frustration"))
        if not (volume_due or frustration_due):
            return {
                "skipped": True,
                "reason": "race requires volume_law or frustration trigger",
                "branches": list(branches),
                "gates": list(gates),
            }
        config: RaceConfig | None = getattr(self.vm, "race_config", None)
        if config is None:
            return {
                "skipped": True,
                "reason": "no offline race fixture configured",
                "branches": list(branches),
                "gates": list(gates),
            }
        trigger = "volume_law" if volume_due else "frustration"
        run_id = f"gauge-race-{self.execution.run_index}-{trigger}"
        parent_hash = str(getattr(self.vm.kernel, "last_hash", "") or "")
        result = run_branch_race(
            source_root=self.vm.root,
            config=config,
            sink=self.vm.sink,
            sink_path=str(self.vm.sink.db_path),
            run_id=run_id,
            parent_hash=parent_hash,
            trigger=trigger,
            branches=branches,
            gates=gates,
        )
        detail = result.as_dict()
        self.vm.result.races.append(cast(GaugeRaceOutcome, detail))
        self.execution.plasticity["branch_race"] = detail
        self.vm.result.plasticity = dict(self.execution.plasticity)
        self.execution.cost.discarded_tokens += int(result.total_tokens)
        self.execution.cost.pruned_branches += max(0, len(result.branches) - 1)
        if result.winner_hash:
            self.execution.variables[winner] = result.winner_hash
        return detail
