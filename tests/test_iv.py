"""Tests for instrumental variables (F5)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.iv_cmds import cmd_ivregress, _parse_iv_formula
from openstat.dsl.parser import ParseError

try:
    import linearmodels
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False


class TestIVFormulaParsing:
    def test_basic_iv_formula(self):
        dep, exog, endog, instruments = _parse_iv_formula("y ~ x1 (x_endog = z1 z2)")
        assert dep == "y"
        assert exog == ["x1"]
        assert endog == ["x_endog"]
        assert instruments == ["z1", "z2"]

    def test_no_parentheses(self):
        with pytest.raises(ParseError, match="parenthesized"):
            _parse_iv_formula("y ~ x1 x_endog = z1 z2")

    def test_no_equals(self):
        with pytest.raises(ParseError, match="="):
            _parse_iv_formula("y ~ x1 (x_endog z1 z2)")

    def test_multiple_exog(self):
        dep, exog, endog, instruments = _parse_iv_formula("y ~ x1 + x2 (x_endog = z1)")
        assert dep == "y"
        assert "x1" in exog
        assert "x2" in exog


@pytest.fixture
def iv_session(tmp_path):
    np.random.seed(42)
    n = 200
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    x1 = np.random.randn(n)
    # x_endog is correlated with z1, z2 (instruments) and error
    e = np.random.randn(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.5 * e  # endogenous
    y = 2.0 + 1.0 * x1 + 0.8 * x_endog + e

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y": y, "x1": x1, "x_endog": x_endog, "z1": z1, "z2": z2,
    })
    return s


class TestIVCommand:
    def test_usage(self, iv_session):
        result = cmd_ivregress(iv_session, "")
        assert "Usage" in result

    @pytest.mark.skipif(not HAS_LINEARMODELS, reason="linearmodels not installed")
    def test_ivregress_2sls(self, iv_session):
        result = cmd_ivregress(iv_session, "2sls y ~ x1 (x_endog = z1 z2)")
        assert "IV-2SLS" in result or "Coef" in result

    @pytest.mark.skipif(not HAS_LINEARMODELS, reason="linearmodels not installed")
    def test_ivregress_robust(self, iv_session):
        result = cmd_ivregress(iv_session, "2sls y ~ x1 (x_endog = z1 z2) --robust")
        assert "Coef" in result or "IV-2SLS" in result
