"""Compact Gauge bytecode IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ast import node_to_dict
from .span import Span


@dataclass(frozen=True, slots=True)
class Instruction:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    body: tuple[Instruction, ...] = ()
    span: Span = field(default_factory=Span)

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": node_to_dict(self.args),
            "body": [instruction.as_dict() for instruction in self.body],
            "span": self.span.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProgramIR:
    setup: tuple[Instruction, ...]
    body: tuple[Instruction, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    span: Span = field(default_factory=Span)

    def instructions(self) -> tuple[Instruction, ...]:
        return self.setup + self.body

    def as_dict(self) -> dict[str, Any]:
        return {
            "setup": [instruction.as_dict() for instruction in self.setup],
            "body": [instruction.as_dict() for instruction in self.body],
            "metadata": node_to_dict(self.metadata),
            "span": self.span.as_dict(),
        }


__all__ = ["Instruction", "ProgramIR"]
