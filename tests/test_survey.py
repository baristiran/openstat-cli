"""Tests for survey weighting (F12)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.survey_cmds import cmd_svyset, cmd_svy
from openstat.commands.stat_cmds import cmd_estat


@pytest.fixture
def svy_session(tmp_path):
    np.random.seed(42)
    n = 300
    # Create stratified sample
    strata = np.repeat([1, 2, 3], n // 3)
    psu = np.tile(np.arange(10), n // 10)
    weight = np.where(strata == 1, 2.0, np.where(strata == 2, 1.5, 1.0))
    x1 = np.random.randn(n) + strata * 0.5
    x2 = np.random.randn(n)
    y = 5.0 + 1.5 * x1 - 0.5 * x2 + np.random.randn(n) * 0.5

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "weight": weight, "strata": strata, "psu": psu,
    })
    return s


class TestSvyset:
    def test_svyset_basic(self, svy_session):
        result = cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        assert "PSU: psu" in result
        assert "Weight: weight" in result
        assert "Strata: strata" in result
        assert svy_session._svy_weight_var == "weight"

    def test_svyset_weight_only(self, svy_session):
        result = cmd_svyset(svy_session, "psu [pw=weight]")
        assert "Weight: weight" in result

    def test_svyset_missing_col(self, svy_session):
        result = cmd_svyset(svy_session, "psu [pw=nonexistent]")
        assert "not found" in result


class TestSvySummarize:
    def test_svy_summarize(self, svy_session):
        cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        result = cmd_svy(svy_session, "summarize y x1")
        assert "Wt.Mean" in result

    def test_svy_no_svyset(self, svy_session):
        result = cmd_svy(svy_session, "summarize y")
        assert "not set" in result


class TestSvyOLS:
    def test_svy_ols(self, svy_session):
        cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        result = cmd_svy(svy_session, "ols y ~ x1 + x2")
        assert "Svy: OLS" in result or "Coef" in result

    def test_svy_ols_stores_result(self, svy_session):
        cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        cmd_svy(svy_session, "ols y ~ x1 + x2")
        assert len(svy_session.results) > 0


class TestEstatDeff:
    def test_estat_deff(self, svy_session):
        cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        cmd_svy(svy_session, "ols y ~ x1 + x2")
        result = cmd_estat(svy_session, "deff")
        assert "DEFF" in result

    def test_estat_deff_no_svyset(self, svy_session):
        # Run an OLS without svyset to trigger "not set"
        from openstat.commands.stat_cmds import cmd_ols
        cmd_ols(svy_session, "y ~ x1 + x2")
        result = cmd_estat(svy_session, "deff")
        assert "not set" in result


class TestSvyLogit:
    def test_svy_logit(self, svy_session):
        # Create binary outcome
        svy_session.df = svy_session.df.with_columns(
            (pl.col("y") > pl.col("y").median()).cast(pl.Float64).alias("y_binary")
        )
        cmd_svyset(svy_session, "psu [pw=weight], strata(strata)")
        result = cmd_svy(svy_session, "logit y_binary ~ x1 + x2")
        assert "Logit" in result or "Coef" in result
