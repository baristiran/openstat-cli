"""Tests for the expression parser."""

import polars as pl
import pytest

from openstat.dsl.parser import parse_expression, parse_formula, ParseError
from openstat.dsl.tokenizer import tokenize, TT


# ── Tokenizer ────────────────────────────────────────────────────────


class TestTokenizer:
    def test_simple_comparison(self):
        tokens = tokenize("age > 30")
        types = [t.type for t in tokens if t.type != TT.EOF]
        assert types == [TT.IDENT, TT.OP, TT.NUMBER]

    def test_string_literal(self):
        tokens = tokenize('region == "North"')
        vals = [t.value for t in tokens if t.type != TT.EOF]
        assert vals == ["region", "==", "North"]

    def test_boolean_keywords(self):
        tokens = tokenize("age > 30 and income < 50000")
        types = [t.type for t in tokens if t.type != TT.EOF]
        assert TT.AND in types

    def test_parentheses(self):
        tokens = tokenize("(a + b) * c")
        types = [t.type for t in tokens if t.type != TT.EOF]
        assert types == [TT.LPAREN, TT.IDENT, TT.OP, TT.IDENT, TT.RPAREN, TT.OP, TT.IDENT]

    def test_backtick_identifier(self):
        tokens = tokenize("`First Name` == \"John\"")
        vals = [t.value for t in tokens if t.type != TT.EOF]
        assert vals == ["First Name", "==", "John"]

    def test_comma_token(self):
        tokens = tokenize("round(x, 2)")
        types = [t.type for t in tokens if t.type != TT.EOF]
        assert TT.COMMA in types

    def test_modulo(self):
        tokens = tokenize("x % 2")
        ops = [t.value for t in tokens if t.type == TT.OP]
        assert ops == ["%"]


# ── Expression Parser ────────────────────────────────────────────────


class TestExpressionParser:
    @pytest.fixture
    def df(self):
        return pl.DataFrame({
            "age": [25, 35, 45],
            "income": [30000, 50000, 70000],
            "region": ["North", "South", "East"],
        })

    def test_simple_comparison(self, df):
        expr = parse_expression("age > 30")
        result = df.filter(expr)
        assert result.height == 2

    def test_and_expression(self, df):
        expr = parse_expression("age > 30 and income < 60000")
        result = df.filter(expr)
        assert result.height == 1

    def test_or_expression(self, df):
        expr = parse_expression("age < 30 or age > 40")
        result = df.filter(expr)
        assert result.height == 2

    def test_string_comparison(self, df):
        expr = parse_expression('region == "North"')
        result = df.filter(expr)
        assert result.height == 1

    def test_arithmetic(self, df):
        expr = parse_expression("income / 1000")
        result = df.with_columns(expr.alias("inc_k"))
        assert result["inc_k"].to_list() == [30.0, 50.0, 70.0]

    def test_not_expression(self, df):
        expr = parse_expression("not age > 30")
        result = df.filter(expr)
        assert result.height == 1

    def test_parentheses(self, df):
        expr = parse_expression("(age + 5) * 2")
        result = df.with_columns(expr.alias("calc"))
        assert result["calc"][0] == 60  # (25+5)*2

    def test_power(self, df):
        expr = parse_expression("age ** 2")
        result = df.with_columns(expr.alias("sq"))
        assert result["sq"][0] == 625  # 25^2

    def test_modulo(self, df):
        expr = parse_expression("age % 10")
        result = df.with_columns(expr.alias("mod"))
        assert result["mod"][0] == 5  # 25 % 10


# ── Built-in Functions ───────────────────────────────────────────────


class TestBuiltinFunctions:
    @pytest.fixture
    def df(self):
        return pl.DataFrame({
            "x": [1.0, 4.0, 9.0, 16.0],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "val": [10.0, None, 30.0, None],
        })

    def test_sqrt(self, df):
        expr = parse_expression("sqrt(x)")
        result = df.with_columns(expr.alias("s"))
        assert result["s"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_abs(self):
        df = pl.DataFrame({"x": [-5, 3, -1]})
        expr = parse_expression("abs(x)")
        result = df.with_columns(expr.alias("a"))
        assert result["a"].to_list() == [5, 3, 1]

    def test_log(self):
        import math
        df = pl.DataFrame({"x": [1.0, math.e, math.e**2]})
        expr = parse_expression("log(x)")
        result = df.with_columns(expr.alias("l"))
        assert abs(result["l"][0] - 0.0) < 0.001
        assert abs(result["l"][1] - 1.0) < 0.001

    def test_upper(self, df):
        expr = parse_expression("upper(name)")
        result = df.with_columns(expr.alias("u"))
        assert result["u"][0] == "ALICE"

    def test_lower(self, df):
        expr = parse_expression("lower(name)")
        result = df.with_columns(expr.alias("l"))
        assert result["l"][0] == "alice"

    def test_is_null(self, df):
        expr = parse_expression("is_null(val)")
        result = df.filter(expr)
        assert result.height == 2

    def test_is_not_null(self, df):
        expr = parse_expression("is_not_null(val)")
        result = df.filter(expr)
        assert result.height == 2

    def test_unknown_function_raises(self):
        with pytest.raises(ParseError, match="Unknown function"):
            parse_expression("evil_exec(x)")

    def test_nested_functions(self):
        df = pl.DataFrame({"x": [4.0, 16.0]})
        expr = parse_expression("log(sqrt(x))")
        result = df.with_columns(expr.alias("r"))
        # log(sqrt(4)) = log(2) ≈ 0.693
        assert abs(result["r"][0] - 0.693) < 0.01

    def test_backtick_column(self):
        df = pl.DataFrame({"First Name": ["Alice", "Bob"]})
        expr = parse_expression('`First Name` == "Alice"')
        result = df.filter(expr)
        assert result.height == 1


# ── Formula Parser ───────────────────────────────────────────────────


class TestFormulaParser:
    def test_basic_formula(self):
        dep, indeps = parse_formula("y ~ x1 + x2")
        assert dep == "y"
        assert indeps == ["x1", "x2"]

    def test_single_predictor(self):
        dep, indeps = parse_formula("score ~ age")
        assert dep == "score"
        assert indeps == ["age"]

    def test_missing_tilde(self):
        with pytest.raises(ParseError):
            parse_formula("y x1 x2")

    def test_missing_dep_var(self):
        with pytest.raises(ParseError):
            parse_formula("~ x1 + x2")

    def test_many_predictors(self):
        dep, indeps = parse_formula("y ~ a + b + c + d + e")
        assert dep == "y"
        assert indeps == ["a", "b", "c", "d", "e"]
