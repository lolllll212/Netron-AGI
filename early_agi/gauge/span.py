"""Source spans and language errors for Gauge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """1-based source span. ``end_col`` is exclusive."""

    line: int = 1
    col: int = 1
    end_line: int = 1
    end_col: int = 1

    def merge(self, other: Span) -> Span:
        return Span(self.line, self.col, other.end_line, other.end_col)

    def as_dict(self) -> dict[str, int]:
        return {
            "line": self.line,
            "col": self.col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }

    def __str__(self) -> str:
        if self.line == self.end_line:
            return f"{self.line}:{self.col}-{self.end_col}"
        return f"{self.line}:{self.col}-{self.end_line}:{self.end_col}"


class GaugeSyntaxError(SyntaxError):
    def __init__(self, message: str, span: Span) -> None:
        super().__init__(f"{message} at {span}")
        self.message = message
        self.span = span


class GaugeSemanticError(ValueError):
    def __init__(self, message: str, span: Span | None = None) -> None:
        detail = f"{message} at {span}" if span is not None else message
        super().__init__(detail)
        self.message = message
        self.span = span


__all__ = ["GaugeSemanticError", "GaugeSyntaxError", "Span"]
