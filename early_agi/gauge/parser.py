"""Recursive-descent parser for Gauge Step 1: frontend only, no semantic checks."""

from __future__ import annotations

from . import ast as g
from .lexer import Token, tokenize
from .span import GaugeSyntaxError, Span


class GaugeParser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[min(self.pos, len(self.tokens) - 1)]
        self.pos = min(self.pos + 1, len(self.tokens))
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            wanted = f"{kind}({value!r})" if value is not None else kind
            raise GaugeSyntaxError(f"Expected {wanted}, got {tok.kind}({tok.value!r})", tok.span)
        return self.advance()

    def match_word(self, word: str) -> Token | None:
        tok = self.peek()
        if tok.kind == "IDENT" and tok.value == word:
            return self.advance()
        return None

    def match_op(self, *ops: str) -> Token | None:
        tok = self.peek()
        if tok.kind == "OP" and tok.value in ops:
            return self.advance()
        return None

    def parse_program(self) -> g.Program:
        start = self.peek().span
        decls: list[g.Declaration] = []
        while self.peek().kind != "EOF":
            decls.append(self.parse_decl())
        end = decls[-1].span if decls else start
        return g.Program(decls, start.merge(end))

    def parse_decl(self) -> g.Declaration:
        tok = self.peek()
        if tok.kind != "IDENT":
            raise GaugeSyntaxError(f"Expected declaration, got {tok.value!r}", tok.span)
        dispatch = {
            "scale": self.parse_scale,
            "manifold": self.parse_manifold,
            "gate": self.parse_gate,
            "extern": self.parse_extern_op,
            "reversible": self.parse_transform,
            "govern": self.parse_govern,
            "plastic": self.parse_plastic_govern,
        }
        handler = dispatch.get(tok.value)
        if handler is None:
            raise GaugeSyntaxError(f"Unknown declaration {tok.value!r}", tok.span)
        return handler()

    def parse_scale(self) -> g.ScaleDecl:
        kw = self.expect("IDENT", "scale")
        uv = self.expect("IDENT")
        self.expect("ARROW")
        ir = self.expect("IDENT")
        return g.ScaleDecl(uv.value, ir.value, kw.span.merge(ir.span))

    def parse_manifold(self) -> g.ManifoldDecl:
        kw = self.expect("IDENT", "manifold")
        name = self.expect("IDENT").value
        self.expect("PUNCT", ":")
        space = self.expect("IDENT").value
        self.expect("PUNCT", "{")
        constraints: list[g.Expr] = []
        while self.peek().value != "}":
            constraints.append(self.parse_expr())
        end = self.expect("PUNCT", "}")
        return g.ManifoldDecl(name, space, constraints, kw.span.merge(end.span))

    def parse_gate(self) -> g.GateDecl:
        kw = self.expect("IDENT", "gate")
        name = self.expect("IDENT").value
        params, _ = self.parse_ident_parens()
        self.expect("PUNCT", ":")
        pred = self.parse_expr()
        return g.GateDecl(name, params, pred, kw.span.merge(pred.span))

    def parse_extern_op(self) -> g.OpDecl:
        kw = self.expect("IDENT", "extern")
        self.expect("IDENT", "op")
        name = self.expect("IDENT").value
        params, last_span = self.parse_ident_parens()
        mods: dict[str, list[str]] = {"deforms": [], "commutes": [], "after": []}
        aliases = {"communes": "commutes"}
        while self.peek().kind == "IDENT" and (self.peek().value in mods or self.peek().value in aliases):
            mod_name = self.advance().value
            target = aliases.get(mod_name, mod_name)
            items, last_span = self.parse_ident_parens()
            mods[target].extend(items)
        return g.OpDecl(
            name=name,
            params=params,
            deforms=mods["deforms"],
            commutes=mods["commutes"],
            after=mods["after"],
            span=kw.span.merge(last_span),
        )

    def parse_transform(self) -> g.TransformDecl:
        kw = self.expect("IDENT", "reversible")
        self.expect("IDENT", "transform")
        name = self.expect("IDENT").value
        self.expect("PUNCT", "(")
        params: list[tuple[str, str]] = []
        if self.peek().value != ")":
            while True:
                p_name = self.expect("IDENT").value
                self.expect("PUNCT", ":")
                p_type = self.expect("IDENT").value
                params.append((p_name, p_type))
                if self.peek().value == ",":
                    self.advance()
                else:
                    break
        self.expect("PUNCT", ")")
        self.expect("ARROW")
        ret_type = self.expect("IDENT").value
        body, end_span = self.parse_block()
        return g.TransformDecl(name, params, ret_type, body, True, kw.span.merge(end_span))

    def parse_govern(self) -> g.GovernDecl:
        kw = self.expect("IDENT", "govern")
        return self._parse_govern(kw, plastic=False)

    def parse_plastic_govern(self) -> g.GovernDecl:
        kw = self.expect("IDENT", "plastic")
        self.expect("IDENT", "govern")
        return self._parse_govern(kw, plastic=True)

    def _parse_govern(self, kw: Token, *, plastic: bool) -> g.GovernDecl:
        config: dict[str, g.Expr] = {}
        while True:
            key = self.expect("IDENT").value
            self.expect("OP", "=")
            config[key] = self.parse_expr()
            if self.peek().value == ",":
                self.advance()
            else:
                break
        body, end_span = self.parse_block()
        return g.GovernDecl(config, body, kw.span.merge(end_span), plastic=plastic)

    def parse_block(self) -> tuple[list[g.Statement], Span]:
        self.expect("PUNCT", "{")
        stmts: list[g.Statement] = []
        while self.peek().value != "}":
            stmts.append(self.parse_stmt())
        end = self.expect("PUNCT", "}")
        return stmts, end.span

    def parse_stmt(self) -> g.Statement:
        if kw := self.match_word("evolve"):
            manifold = self.expect("IDENT").value
            self.expect("IDENT", "under")
            self.expect("PUNCT", "[")
            gates = self.parse_ident_list("]")
            end = self.expect("PUNCT", "]")
            return g.EvolveStmt(manifold, gates, kw.span.merge(end.span))

        if kw := self.match_word("on"):
            event = self.expect("IDENT").value
            binding: str | None = None
            if self.peek().value == "(":
                self.advance()
                binding = self.expect("IDENT").value
                self.expect("PUNCT", ")")
            body, end_span = self.parse_block()
            return g.TrapHandler(event, binding, body, kw.span.merge(end_span))

        if kw := self.match_word("synthesize"):
            self.expect("IDENT", "op")
            op_name = self.expect("IDENT").value
            self.expect("IDENT", "reducing")
            metric = self.expect("IDENT")
            return g.SynthesizeStmt(op_name, metric.value, kw.span.merge(metric.span))

        if kw := self.match_word("probe"):
            manifold = self.expect("IDENT").value
            self.expect("IDENT", "with")
            self.expect("PUNCT", "[")
            ops = self.parse_ident_list("]")
            end = self.expect("PUNCT", "]")
            return g.ProbeStmt(manifold, ops, kw.span.merge(end.span))

        if kw := self.match_word("race"):
            self.expect("PUNCT", "[")
            branches = self.parse_ident_list("]")
            self.expect("PUNCT", "]")
            self.expect("IDENT", "under")
            self.expect("PUNCT", "[")
            gates = self.parse_ident_list("]")
            self.expect("PUNCT", "]")
            self.expect("ARROW")
            winner = self.expect("IDENT")
            return g.RaceStmt(branches, gates, winner.value, kw.span.merge(winner.span))

        if kw := self.match_word("recall"):
            if self.match_word("analogy"):
                self.expect("IDENT", "where")
                cond = self.parse_expr()
                return g.RecallStmt(cond, kw.span.merge(cond.span), kind="analogy")
            self.expect("IDENT", "sink")
            self.expect("IDENT", "where")
            cond = self.parse_expr()
            return g.RecallStmt(cond, kw.span.merge(cond.span), kind="sink")

        if kw := self.match_word("uncall"):
            name_tok = self.expect("IDENT")
            args: list[g.Expr] = []
            end = name_tok.span
            if self.peek().value == "(":
                args, kwargs, end = self.parse_call_args()
                if kwargs:
                    raise GaugeSyntaxError("uncall does not accept keyword arguments", end)
            return g.UncallStmt(name_tok.value, args, kw.span.merge(end))

        if self.peek().kind == "IDENT" and self.peek().value == "collapse" and self.peek(1).value == "(":
            start = self.peek().span
            return self._parse_collapse(None, start)

        if self.peek().kind == "IDENT" and self.peek(1).kind == "OP" and self.peek(1).value == "=":
            target_tok = self.advance()
            self.expect("OP", "=")
            if self.peek().kind == "IDENT" and self.peek().value == "collapse" and self.peek(1).value == "(":
                return self._parse_collapse(target_tok.value, target_tok.span)
            val = self.parse_expr()
            return g.AssignStmt(target_tok.value, val, target_tok.span.merge(val.span))

        expr = self.parse_expr()
        return g.ExprStmt(expr, expr.span)

    def _parse_collapse(self, target: str | None, start: Span) -> g.CollapseStmt:
        self.expect("IDENT", "collapse")
        self.expect("PUNCT", "(")
        manifold = self.expect("IDENT").value
        self.expect("PUNCT", ")")
        self.expect("ARROW")
        sink_tok = self.expect("IDENT", "sink")
        return g.CollapseStmt(target, manifold, sink_tok.value, start.merge(sink_tok.span))

    def parse_expr(self) -> g.Expr:
        return self.parse_or()

    def parse_or(self) -> g.Expr:
        node = self.parse_and()
        while self.match_word("or"):
            right = self.parse_and()
            node = g.BinaryExpr("or", node, right, node.span.merge(right.span))
        return node

    def parse_and(self) -> g.Expr:
        node = self.parse_not()
        while self.match_word("and"):
            right = self.parse_not()
            node = g.BinaryExpr("and", node, right, node.span.merge(right.span))
        return node

    def parse_not(self) -> g.Expr:
        if kw := self.match_word("not"):
            operand = self.parse_not()
            return g.UnaryExpr("not", operand, kw.span.merge(operand.span))
        return self.parse_cmp()

    def parse_cmp(self) -> g.Expr:
        left = self.parse_add()
        tok = self.peek()
        if (tok.kind == "OP" and tok.value in ("==", "!=", "<=", ">=", "<", ">")) or (
            tok.kind == "IDENT" and tok.value == "in"
        ):
            op = self.advance().value
            right = self.parse_add()
            return g.BinaryExpr(op, left, right, left.span.merge(right.span))
        return left

    def parse_add(self) -> g.Expr:
        node = self.parse_mul()
        while tok := self.match_op("+", "-"):
            right = self.parse_mul()
            node = g.BinaryExpr(tok.value, node, right, node.span.merge(right.span))
        return node

    def parse_mul(self) -> g.Expr:
        node = self.parse_unary()
        while tok := self.match_op("*", "/"):
            right = self.parse_unary()
            node = g.BinaryExpr(tok.value, node, right, node.span.merge(right.span))
        return node

    def parse_unary(self) -> g.Expr:
        if tok := self.match_op("-"):
            operand = self.parse_unary()
            return g.UnaryExpr("-", operand, tok.span.merge(operand.span))
        return self.parse_postfix()

    def parse_postfix(self) -> g.Expr:
        node = self.parse_primary()
        while True:
            if self.peek().value == ".":
                self.advance()
                attr = self.expect("IDENT")
                node = g.MemberExpr(node, attr.value, node.span.merge(attr.span))
            elif self.peek().value == "(":
                args, kwargs, end_span = self.parse_call_args()
                node = g.CallExpr(node, args, kwargs, node.span.merge(end_span))
            else:
                break
        return node

    def parse_primary(self) -> g.Expr:
        tok = self.peek()
        if tok.kind == "INT":
            t = self.advance()
            return g.LiteralExpr(int(t.value), t.span)
        if tok.kind == "FLOAT":
            t = self.advance()
            return g.LiteralExpr(float(t.value), t.span)
        if tok.kind == "STRING":
            t = self.advance()
            return g.LiteralExpr(t.value, t.span)
        if tok.kind == "IDENT":
            t = self.advance()
            if t.value == "true":
                return g.LiteralExpr(True, t.span)
            if t.value == "false":
                return g.LiteralExpr(False, t.span)
            if t.value == "none":
                return g.LiteralExpr(None, t.span)
            return g.IdentExpr(t.value, t.span)
        if tok.value == "[":
            start = self.advance().span
            elems: list[g.Expr] = []
            if self.peek().value != "]":
                while True:
                    elems.append(self.parse_expr())
                    if self.peek().value == ",":
                        self.advance()
                    else:
                        break
            end = self.expect("PUNCT", "]").span
            return g.ListExpr(elems, start.merge(end))
        if tok.value == "(":
            self.advance()
            expr = self.parse_expr()
            self.expect("PUNCT", ")")
            return expr
        raise GaugeSyntaxError(f"Unexpected token {tok.kind}({tok.value!r})", tok.span)

    def parse_call_args(self) -> tuple[list[g.Expr], dict[str, g.Expr], Span]:
        self.expect("PUNCT", "(")
        args: list[g.Expr] = []
        kwargs: dict[str, g.Expr] = {}
        if self.peek().value != ")":
            while True:
                if self.peek().kind == "IDENT" and self.peek(1).kind == "OP" and self.peek(1).value == "=":
                    key = self.advance().value
                    self.expect("OP", "=")
                    kwargs[key] = self.parse_expr()
                else:
                    args.append(self.parse_expr())
                if self.peek().value == ",":
                    self.advance()
                else:
                    break
        end = self.expect("PUNCT", ")")
        return args, kwargs, end.span

    def parse_ident_parens(self) -> tuple[list[str], Span]:
        self.expect("PUNCT", "(")
        items = self.parse_ident_list(")")
        end = self.expect("PUNCT", ")")
        return items, end.span

    def parse_ident_list(self, closing: str) -> list[str]:
        items: list[str] = []
        if self.peek().value != closing:
            while True:
                items.append(self.expect("IDENT").value)
                if self.peek().value == ",":
                    self.advance()
                else:
                    break
        return items


def parse_program(source: str) -> g.Program:
    return GaugeParser(tokenize(source)).parse_program()


parse_gauge = parse_program

__all__ = ["GaugeParser", "parse_gauge", "parse_program"]
