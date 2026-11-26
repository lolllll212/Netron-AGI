"""Gauge lexer with exact line/column spans."""

from __future__ import annotations

import ast as pyast
import re
from dataclasses import dataclass

from .span import GaugeSyntaxError, Span

TOKEN_SPEC: list[tuple[str, str]] = [
    ("COMMENT", r"//[^\n]*"),
    ("ARROW", r"->"),
    ("OP", r"==|!=|<=|>=|[=<>+\-*/]"),
    ("FLOAT", r"\d+\.\d+"),
    ("INT", r"\d+"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("PUNCT", r"[{}()\[\],:.]"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("MISMATCH", r"."),
]
MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str
    span: Span


def _decode_string(raw: str, span: Span) -> str:
    try:
        value = pyast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise GaugeSyntaxError(f"invalid string literal: {exc}", span) from exc
    if not isinstance(value, str):
        raise GaugeSyntaxError("string literal did not decode to text", span)
    return value


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    line_start = 0

    for match in MASTER_RE.finditer(source):
        kind = match.lastgroup or "MISMATCH"
        value = match.group()
        start, end = match.start(), match.end()
        col = start - line_start + 1
        end_col = end - line_start + 1
        span = Span(line, col, line, end_col)

        if kind == "NEWLINE":
            line += 1
            line_start = end
            continue
        if kind in ("SKIP", "COMMENT"):
            continue
        if kind == "MISMATCH":
            raise GaugeSyntaxError(f"Unexpected character {value!r}", span)
        if kind == "STRING":
            value = _decode_string(value, span)
            kind = "STRING"
        tokens.append(Token(kind, value, span))

    eof_col = len(source) - line_start + 1
    tokens.append(Token("EOF", "", Span(line, eof_col, line, eof_col)))
    return tokens


__all__ = ["MASTER_RE", "TOKEN_SPEC", "Token", "tokenize"]
