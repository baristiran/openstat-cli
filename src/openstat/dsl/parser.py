"""Safe recursive-descent parser: expression string -> Polars expression.

Grammar:
    expr     -> or_expr
    or_expr  -> and_expr ('or' and_expr)*
    and_expr -> not_expr ('and' not_expr)*
    not_expr -> 'not' not_expr | compare
    compare  -> add (comp_op add)?
    add      -> mul (('+' | '-') mul)*
    mul      -> power (('*' | '/' | '%') power)*
    power    -> unary ('**' unary)?
    unary    -> '-' unary | atom
    atom     -> NUMBER | STRING | func_call | IDENT | '(' expr ')'
    func_call -> IDENT '(' args? ')'
    args     -> expr (',' expr)*

Produces a polars.Expr.  No Python eval is ever used.

Supported functions (whitelisted):
    Math:    log, sqrt, abs, round, exp
    String:  upper, lower, len_chars
    Null:    is_null, is_not_null, fill_null
    Type:    cast_float, cast_int, cast_str
"""

from __future__ import annotations

import math

import polars as pl

from openstat.dsl.tokenizer import TT, Token, tokenize


class ParseError(Exception):
    pass


# ── Whitelisted functions ────────────────────────────────────────────

def _apply_function(name: str, args: list[pl.Expr]) -> pl.Expr:
    """Apply a whitelisted function to Polars expressions."""
    # Math functions (1 argument)
    if name == "log" and len(args) == 1:
        return args[0].log(math.e)
    if name == "log10" and len(args) == 1:
        return args[0].log(10)
    if name == "sqrt" and len(args) == 1:
        return args[0].sqrt()
    if name == "abs" and len(args) == 1:
        return args[0].abs()
    if name == "exp" and len(args) == 1:
        return args[0].exp()
    if name == "round" and len(args) in (1, 2):
        decimals = 0
        if len(args) == 2:
            # Extract literal integer from the expression
            try:
                # Evaluate the literal expression to get the integer value
                decimals = int(pl.select(args[1]).item())
            except Exception:
                raise ParseError("round() second argument must be a literal integer")
        return args[0].round(decimals)

    # String functions (1 argument, operate on the column)
    if name == "upper" and len(args) == 1:
        return args[0].str.to_uppercase()
    if name == "lower" and len(args) == 1:
        return args[0].str.to_lowercase()
    if name == "len_chars" and len(args) == 1:
        return args[0].str.len_chars()
    if name == "strip" and len(args) == 1:
        return args[0].str.strip_chars()
    if name == "contains" and len(args) == 2:
        return args[0].str.contains(args[1])

    # Null functions
    if name == "is_null" and len(args) == 1:
        return args[0].is_null()
    if name == "is_not_null" and len(args) == 1:
        return args[0].is_not_null()
    if name == "fill_null" and len(args) == 2:
        return args[0].fill_null(args[1])

    # Cast functions
    if name == "cast_float" and len(args) == 1:
        return args[0].cast(pl.Float64)
    if name == "cast_int" and len(args) == 1:
        return args[0].cast(pl.Int64)
    if name == "cast_str" and len(args) == 1:
        return args[0].cast(pl.Utf8)

    available = (
        "log, log10, sqrt, abs, exp, round, "
        "upper, lower, len_chars, strip, contains, "
        "is_null, is_not_null, fill_null, "
        "cast_float, cast_int, cast_str"
    )
    raise ParseError(
        f"Unknown function '{name}' with {len(args)} argument(s). "
        f"Available: {available}"
    )


# ── Parser ───────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- helpers ---------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TT, value: str | None = None) -> Token:
        tok = self._advance()
        if tok.type != tt or (value is not None and tok.value != value):
            raise ParseError(f"Expected {tt.name} {value!r}, got {tok}")
        return tok

    def _match_op(self, *ops: str) -> str | None:
        tok = self._peek()
        if tok.type == TT.OP and tok.value in ops:
            self._advance()
            return tok.value
        return None

    # -- grammar ---------------------------------------------------------

    def parse(self) -> pl.Expr:
        expr = self._or_expr()
        if self._peek().type != TT.EOF:
            raise ParseError(f"Unexpected token: {self._peek()}")
        return expr

    def _or_expr(self) -> pl.Expr:
        left = self._and_expr()
        while self._peek().type == TT.OR:
            self._advance()
            right = self._and_expr()
            left = left | right
        return left

    def _and_expr(self) -> pl.Expr:
        left = self._not_expr()
        while self._peek().type == TT.AND:
            self._advance()
            right = self._not_expr()
            left = left & right
        return left

    def _not_expr(self) -> pl.Expr:
        if self._peek().type == TT.NOT:
            self._advance()
            return ~self._not_expr()
        return self._compare()

    def _compare(self) -> pl.Expr:
        left = self._add()
        op = self._match_op(">", "<", ">=", "<=", "==", "!=")
        if op is None:
            return left
        right = self._add()
        ops = {
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
            "==": left == right,
            "!=": left != right,
        }
        return ops[op]

    def _add(self) -> pl.Expr:
        left = self._mul()
        while True:
            op = self._match_op("+", "-")
            if op is None:
                break
            right = self._mul()
            left = left + right if op == "+" else left - right
        return left

    def _mul(self) -> pl.Expr:
        left = self._power()
        while True:
            op = self._match_op("*", "/", "%")
            if op is None:
                break
            right = self._power()
            if op == "*":
                left = left * right
            elif op == "/":
                left = left / right
            else:
                left = left % right
        return left

    def _power(self) -> pl.Expr:
        base = self._unary()
        if self._match_op("**"):
            exp = self._unary()
            return base.pow(exp)
        return base

    def _unary(self) -> pl.Expr:
        if self._match_op("-"):
            return -self._unary()
        return self._atom()

    def _atom(self) -> pl.Expr:
        tok = self._peek()

        if tok.type == TT.NUMBER:
            self._advance()
            val = float(tok.value) if "." in tok.value else int(tok.value)
            return pl.lit(val)

        if tok.type == TT.STRING:
            self._advance()
            return pl.lit(tok.value)

        if tok.type == TT.IDENT:
            # Check if it's a function call: IDENT '('
            next_pos = self.pos + 1
            if next_pos < len(self.tokens) and self.tokens[next_pos].type == TT.LPAREN:
                return self._func_call()
            self._advance()
            return pl.col(tok.value)

        if tok.type == TT.LPAREN:
            self._advance()
            expr = self._or_expr()
            self._expect(TT.RPAREN)
            return expr

        raise ParseError(f"Unexpected token: {tok}")

    def _func_call(self) -> pl.Expr:
        """Parse function_name(arg1, arg2, ...)."""
        name_tok = self._advance()  # IDENT
        self._expect(TT.LPAREN)

        args: list[pl.Expr] = []
        if self._peek().type != TT.RPAREN:
            args.append(self._or_expr())
            while self._peek().type == TT.COMMA:
                self._advance()  # skip comma
                args.append(self._or_expr())

        self._expect(TT.RPAREN)
        return _apply_function(name_tok.value, args)


def parse_expression(text: str) -> pl.Expr:
    """Parse an expression string into a Polars Expr (safe, no eval)."""
    tokens = tokenize(text)
    return _Parser(tokens).parse()


def parse_formula(text: str) -> tuple[str, list[str]]:
    """Parse 'y ~ x1 + x2 + x3' into (dep_var, [indep_vars]).

    The '+' here means 'include predictor', not arithmetic addition.

    Interaction syntax:
    - x1:x2  → interaction only (product term)
    - x1*x2  → full factorial = x1 + x2 + x1:x2
    """
    text = text.strip()
    if "~" not in text:
        raise ParseError("Formula must contain '~', e.g. y ~ x1 + x2")
    left, right = text.split("~", 1)
    dep = left.strip()
    if not dep:
        raise ParseError("Missing dependent variable before '~'")

    # Expand x1*x2 → x1 + x2 + x1:x2 before splitting on +
    right = _expand_star_interactions(right)

    indeps = [v.strip() for v in right.split("+")]
    indeps = [v for v in indeps if v]
    if not indeps:
        raise ParseError("Missing independent variables after '~'")

    # Normalize interaction terms: strip whitespace around ':'
    indeps = [
        ":".join(p.strip() for p in v.split(":")) if ":" in v else v
        for v in indeps
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in indeps:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return dep, unique


def _expand_star_interactions(rhs: str) -> str:
    """Expand full-factorial ``*`` terms in a formula RHS string.

    - ``x1*x2``       → ``x1 + x2 + x1:x2``
    - ``x1*x2*x3``    → ``x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + x1:x2:x3``
    """
    from itertools import combinations

    terms = [t.strip() for t in rhs.split("+")]
    expanded: list[str] = []
    for term in terms:
        if "*" in term and ":" not in term:
            parts = [p.strip() for p in term.split("*")]
            # Generate all subsets of size 1..len(parts)
            for r in range(1, len(parts) + 1):
                for combo in combinations(parts, r):
                    if r == 1:
                        expanded.append(combo[0])
                    else:
                        expanded.append(":".join(combo))
        else:
            expanded.append(term)
    return " + ".join(expanded)
