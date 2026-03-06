"""Tests for multiple imputation (F11)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.mi_cmds import cmd_mi


@pytest.fixture
def mi_session(tmp_path):
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + np.random.randn(n) * 0.5

    # Introduce MCAR missing data (~15%)
    x1_miss = x1.copy()
    x2_miss = x2.copy()
    mask1 = np.random.rand(n) < 0.15
    mask2 = np.random.rand(n) < 0.15
    x1_miss[mask1] = None
    x2_miss[mask2] = None

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y": y,
        "x1": x1_miss.tolist(),
        "x2": x2_miss.tolist(),
    })
    return s


class TestMIImpute:
    def test_mi_impute_chained(self, mi_session):
        result = cmd_mi(mi_session, "impute chained (regress) x1 (regress) x2, add(3)")
        assert "imputed datasets" in result or "Created" in result
        assert mi_session._imputed_datasets is not None
        assert len(mi_session._imputed_datasets) == 3

    def test_mi_impute_pmm(self, mi_session):
        result = cmd_mi(mi_session, "impute pmm x1 x2, add(3)")
        assert "imputed datasets" in result or "Created" in result

    def test_mi_impute_no_missing(self, tmp_path):
        s = Session(output_dir=tmp_path / "out")
        s.df = pl.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        result = cmd_mi(s, "impute chained (regress) x1, add(5)")
        assert "No missing" in result

    def test_mi_impute_missing_col(self, mi_session):
        result = cmd_mi(mi_session, "impute chained (regress) nonexistent, add(5)")
        assert "not found" in result


class TestMIEstimate:
    def test_mi_estimate_no_imputation(self, mi_session):
        result = cmd_mi(mi_session, "estimate: ols y ~ x1 + x2")
        assert "No imputed" in result

    def test_mi_estimate_ols(self, mi_session):
        cmd_mi(mi_session, "impute chained (regress) x1 (regress) x2, add(5)")
        result = cmd_mi(mi_session, "estimate: ols y ~ x1 + x2")
        assert "MI" in result
        assert "Coef" in result
        assert "FMI" in result


class TestMIDescribe:
    def test_mi_describe_no_data(self, mi_session):
        result = cmd_mi(mi_session, "describe")
        assert "No imputed" in result

    def test_mi_describe(self, mi_session):
        cmd_mi(mi_session, "impute chained (regress) x1 (regress) x2, add(3)")
        result = cmd_mi(mi_session, "describe")
        assert "3" in result


class TestMIUsage:
    def test_mi_usage(self, mi_session):
        result = cmd_mi(mi_session, "")
        assert "Usage" in result
