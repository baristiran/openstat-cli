"""Tests for advanced time-series commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.ts_advanced import granger_causality, johansen_test, stl_decompose, tssmooth


@pytest.fixture()
def ts_df():
    rng = np.random.default_rng(42)
    n = 100
    x = np.cumsum(rng.normal(0, 1, n))
    y = 0.5 * x + np.cumsum(rng.normal(0, 0.5, n))
    return pl.DataFrame({"x": x, "y": y})


@pytest.fixture()
def seasonal_df():
    n = 120
    t = np.arange(n)
    y = 2 * np.sin(2 * np.pi * t / 12) + 0.05 * t + np.random.default_rng(0).normal(0, 0.3, n)
    return pl.DataFrame({"y": y})


class TestGranger:
    def test_keys(self, ts_df):
        r = granger_causality(ts_df, "y", "x", maxlag=2)
        assert "lag_pvalues" in r and "min_pvalue" in r

    def test_lag_count(self, ts_df):
        r = granger_causality(ts_df, "y", "x", maxlag=3)
        assert len(r["lag_pvalues"]) == 3

    def test_reject_null(self, ts_df):
        r = granger_causality(ts_df, "y", "x", maxlag=2)
        assert isinstance(r["reject_null_5pct"], bool)

    def test_cmd(self, ts_df):
        from openstat.commands.ts_adv_cmds import cmd_granger
        s = Session()
        s.df = ts_df
        out = cmd_granger(s, "y x maxlag(2)")
        assert "Granger" in out

    def test_cmd_missing_var(self, ts_df):
        from openstat.commands.ts_adv_cmds import cmd_granger
        s = Session()
        s.df = ts_df
        out = cmd_granger(s, "y")
        assert "Usage" in out


class TestJohansen:
    def test_keys(self, ts_df):
        r = johansen_test(ts_df, ["x", "y"], k_ar_diff=1)
        assert "n_cointegrating_vectors" in r
        assert "trace_statistics" in r

    def test_cmd(self, ts_df):
        from openstat.commands.ts_adv_cmds import cmd_johansen
        s = Session()
        s.df = ts_df
        out = cmd_johansen(s, "x y lags(1)")
        assert "Johansen" in out


class TestSTL:
    def test_keys(self, seasonal_df):
        r = stl_decompose(seasonal_df, "y", period=12)
        assert "trend" in r and "seasonal" in r and "resid" in r

    def test_lengths(self, seasonal_df):
        r = stl_decompose(seasonal_df, "y", period=12)
        assert len(r["trend"]) == len(seasonal_df)

    def test_strength(self, seasonal_df):
        r = stl_decompose(seasonal_df, "y", period=12)
        assert 0 <= r["strength_seasonal"] <= 1.1

    def test_cmd(self, seasonal_df):
        from openstat.commands.ts_adv_cmds import cmd_stl
        s = Session()
        s.df = seasonal_df
        out = cmd_stl(s, "y period(12)")
        assert "STL" in out


class TestTssmooth:
    def test_ma(self, seasonal_df):
        result = tssmooth(seasonal_df, "y", method="ma", window=3)
        assert "y_smooth" in result.columns

    def test_exp(self, seasonal_df):
        result = tssmooth(seasonal_df, "y", method="exp", alpha=0.3)
        assert "y_smooth" in result.columns

    def test_cmd_ma(self, seasonal_df):
        from openstat.commands.ts_adv_cmds import cmd_tssmooth
        s = Session()
        s.df = seasonal_df
        out = cmd_tssmooth(s, "y method(ma) window(5)")
        assert "y_smooth" in s.df.columns

    def test_cmd_exp(self, seasonal_df):
        from openstat.commands.ts_adv_cmds import cmd_tssmooth
        s = Session()
        s.df = seasonal_df
        out = cmd_tssmooth(s, "y method(exp) alpha(0.2)")
        assert "y_smooth" in s.df.columns

    def test_invalid_method(self, seasonal_df):
        with pytest.raises(ValueError):
            tssmooth(seasonal_df, "y", method="unknown")
