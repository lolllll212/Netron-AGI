"""Frontend parsing: precedence, spans, decoded literals, and malformed syntax."""

from __future__ import annotations

import pytest

from _gauge_fixtures import SOURCE
from early_agi.gauge import ast as g
from early_agi.gauge import parse_gauge
from early_agi.gauge.span import GaugeSyntaxError


def test_example_parses_with_spans_and_declarations() -> None:
    program = parse_gauge(SOURCE)
    assert len(program.declarations) == 8
    assert program.span.line >= 1
    assert program.span.end_line >= 40
    assert program.span.end_col > program.span.col
    assert [type(decl).__name__ for decl in program.declarations[:4]] == [
        "ScaleDecl",
        "ManifoldDecl",
        "GateDecl",
        "GateDecl",
    ]
    scale = program.declarations[0]
    assert isinstance(scale, g.ScaleDecl)
    assert (scale.uv_scale, scale.ir_scale) == ("L0", "L2")
    assert scale.span.line == 2


def test_expression_precedence_is_span_tagged() -> None:
    program = parse_gauge("gate logic(p) : true or false and not true\ngate arithmetic(p) : 1 + 2 * 3 == 7\n")
    logic = program.declarations[0].predicate
    assert isinstance(logic, g.BinaryExpr)
    assert logic.op == "or"
    assert isinstance(logic.right, g.BinaryExpr)
    assert logic.right.op == "and"
    assert isinstance(logic.right.right, g.UnaryExpr)
    assert logic.right.right.op == "not"
    assert logic.span.end_col > logic.span.col

    arithmetic = program.declarations[1].predicate
    assert isinstance(arithmetic, g.BinaryExpr)
    assert arithmetic.op == "=="
    assert isinstance(arithmetic.left, g.BinaryExpr)
    assert arithmetic.left.op == "+"
    assert isinstance(arithmetic.left.right, g.BinaryExpr)
    assert arithmetic.left.right.op == "*"


def test_string_literals_are_decoded_not_left_as_raw_tokens() -> None:
    program = parse_gauge(r'gate strings(p) : "a\n\"b" == "a\n\"b"')
    predicate = program.declarations[0].predicate
    assert isinstance(predicate, g.BinaryExpr)
    assert isinstance(predicate.left, g.LiteralExpr)
    assert isinstance(predicate.right, g.LiteralExpr)
    assert predicate.left.value == 'a\n"b'
    assert predicate.right.value == 'a\n"b'


@pytest.mark.parametrize(
    "source",
    [
        "unknown x",
        "manifold m : PatchSpace {",
        "gate bad(p) : 1 +",
        "scale L0 L2",
    ],
)
def test_malformed_syntax_raises_span_tagged_error(source: str) -> None:
    with pytest.raises(GaugeSyntaxError) as excinfo:
        parse_gauge(source)
    assert excinfo.value.span.line >= 1
    assert excinfo.value.span.col >= 1
    assert str(excinfo.value)
