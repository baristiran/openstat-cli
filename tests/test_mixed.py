"""Tests for mixed/hierarchical models (F4)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.mixed_cmds import cmd_mixed, _parse_mixed_formula
from openstat.commands.stat_cmds import cmd_estat
from openstat.dsl.parser import ParseError


class TestMixedFormulaParsing:
    def test_random_intercept(self):
        dep, fixed, group, re_vars = _parse_mixed_formula("y ~ x1 || school:")
        assert dep == "y"
        assert fixed == ["x1"]
        assert group == "school"
        assert re_vars == []

    def test_random_slope(self):
        dep, fixed, group, re_vars = _parse_mixed_formula("y ~ x1 || school: x1")
        assert dep == "y"
        assert group == "school"
        assert re_vars == ["x1"]

    def test_no_pipe(self):
        with pytest.raises(ParseError, match="\\|\\|"):
            _parse_mixed_formula("y ~ x1")


@pytest.fixture
def mixed_session(tmp_path):
    np.random.seed(42)
    n_groups = 20
    n_per_group = 10
    n = n_groups * n_per_group

    group = np.repeat(np.arange(n_groups), n_per_group)
    group_effect = np.repeat(np.random.randn(n_groups) * 2, n_per_group)
    x1 = np.random.randn(n)
    y = 5.0 + 1.5 * x1 + group_effect + np.random.randn(n) * 0.5

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y": y, "x1": x1, "school": group,
    })
    return s


class TestMixedCommand:
    def test_random_intercept(self, mixed_session):
        result = cmd_mixed(mixed_session, "y ~ x1 || school:")
        assert "Mixed LM" in result or "Coef" in result

    def test_random_slope(self, mixed_session):
        result = cmd_mixed(mixed_session, "y ~ x1 || school: x1")
        assert "Mixed LM" in result or "Coef" in result

    def test_missing_column(self, mixed_session):
        result = cmd_mixed(mixed_session, "y ~ x1 || nonexistent:")
        assert "not found" in result

    def test_icc_in_output(self, mixed_session):
        result = cmd_mixed(mixed_session, "y ~ x1 || school:")
        assert "ICC" in result

    def test_estat_icc(self, mixed_session):
        cmd_mixed(mixed_session, "y ~ x1 || school:")
        result = cmd_estat(mixed_session, "icc")
        assert "ICC" in result
