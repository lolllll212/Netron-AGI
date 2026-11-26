"""Trap evaluation, uncall dispatch, and inverse-stack rollback.

Trap bodies are ordinary bytecode, but deciding whether an event is due and
restoring reversible frames are isolated here so the VM dispatcher never owns
rollback state directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from .expressions import evaluate
from .ir import Instruction
from .runtime import ManifoldState

__all__ = ["TrapPolicy", "TrapRunner"]


class TrapPolicy:
    """Pure event aliasing for thermodynamic traps."""

    _ALIASES: ClassVar[dict[str, frozenset[str]]] = {
        "volume_law": frozenset({"volume_law", "frustration"}),
        "drag": frozenset({"drag", "stall"}),
        "frustration": frozenset({"frustration", "recall"}),
        "stall": frozenset({"stall", "drag"}),
        "shift": frozenset({"shift", "volume_law"}),
        "recall": frozenset({"recall", "frustration"}),
    }

    @classmethod
    def due(cls, event: str, events: set[str]) -> bool:
        return bool(events & cls._ALIASES.get(event, frozenset({event})))


class TrapRunner:
    """Execute trap instructions and centralise recall/uncall recovery."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    def is_due(self, event: str) -> bool:
        return TrapPolicy.due(event, self.vm.execution.events)

    def execute(self, instruction: Instruction, executor: Callable[[Instruction], None]) -> None:
        op = instruction.op
        event = str(instruction.args.get("event", op.removeprefix("TRAP_").casefold()))
        if not self.is_due(event):
            self.vm.emit(op, {"event": event, "skipped": True})
            return
        self.vm.traps.append(event)
        self.vm.emit(op, {"event": event, "skipped": False})
        for child in instruction.body:
            executor(child)

    def uncall_transform(self, args: Mapping[str, Any]) -> dict[str, Any]:
        name = str(args["transform"])
        transform = self.vm.execution.transforms.get(name)
        context = self.vm.execution.evaluation_context()
        evaluated_args = [evaluate(item, context) for item in args.get("args", ())]
        manifold = next(
            (item for item in evaluated_args if isinstance(item, ManifoldState)),
            None,
        )
        restored = transform.uncall(manifold) if transform is not None else None
        return {"transform": name, "restored": restored.name if restored else ""}

    def recall_sink(self, condition: Any | None) -> dict[str, Any]:
        execution = self.vm.execution
        verdict = evaluate(condition, execution.evaluation_context()) if condition is not None else True
        if verdict is not True:
            return {"recalled": 0, "condition": verdict}
        records = self.vm.sink.records(
            kinds=("gauge_collapse", "pruned_branch", "uv_state"),
            limit=8,
            oldest_first=False,
        )
        execution.recalled = [record.as_dict() for record in records]
        self.vm.result.recalled = list(execution.recalled)
        frames = self.rollback_inverse_stack()
        self.vm.kernel.restore_checkpoint(execution.kernel_checkpoint)
        self.vm.kernel.reconcile_gates(execution.plain_state())
        execution.gate_state_committed = True
        return {
            "recalled": len(execution.recalled),
            "scale": execution.state.scale,
            "inverse_frames": frames,
        }

    def rollback_inverse_stack(self) -> int:
        return self.vm.execution.restore_inverse_stack()

    def recover(self) -> int:
        """Restore reversible frames and kernel bookkeeping after a failed tick."""
        frames = self.rollback_inverse_stack()
        execution = self.vm.execution
        self.vm.kernel.restore_checkpoint(execution.kernel_checkpoint)
        self.vm.kernel.reconcile_gates(execution.plain_state())
        execution.gate_state_committed = True
        return frames
