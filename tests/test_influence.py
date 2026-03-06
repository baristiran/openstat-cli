"""Tests for influence diagnostics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.influence import compute_influence, detect_outliers


@pytest.fixture()
def ols_df():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    # Add one outlier
    y[0] = 100.0
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestComputeInfluence:
    def test_keys(self, ols_df):
        r = compute_influence(ols_df, "y", ["x1", "x2"])
        assert "leverage" in r and "cooks_d" in r and "dfbetas" in r

    def test_lengths(self, ols_df):
        r = compute_influence(ols_df, "y", ["x1", "x2"])
        assert len(r["leverage"]) == len(ols_df)
        assert len(r["cooks_d"]) == len(ols_df)

    def test_leverage_range(self, ols_df):
        r = compute_influence(ols_df, "y", ["x1", "x2"])
        lev = np.array(r["leverage"])
        assert (lev >= 0).all() and (lev <= 1).all()

    def test_cooks_d_positive(self, ols_df):
        r = compute_influence(ols_df, "y", ["x1", "x2"])
        assert all(v >= 0 for v in r["cooks_d"])

    def test_dfbetas_keys(self, ols_df):
        r = compute_influence(ols_df, "y", ["x1", "x2"])
        assert "x1" in r["dfbetas"] and "x2" in r["dfbetas"]

    def test_detects_outlier(self, ols_df):
        r = detect_outliers(ols_df, "y", ["x1", "x2"], threshold=3.0)
        assert r["n_outliers"] >= 1
        assert 0 in r["outlier_indices"]


class TestInfluenceCmds:
    def test_dfbeta_cmd(self, ols_df):
        from openstat.commands.influence_cmds import cmd_dfbeta
        s = Session()
        s.df = ols_df
        out = cmd_dfbeta(s, "y x1 x2")
        assert "DFBETA" in out

    def test_leverage_cmd(self, ols_df):
        from openstat.commands.influence_cmds import cmd_leverage
        s = Session()
        s.df = ols_df
        out = cmd_leverage(s, "y x1 x2")
        assert "Leverage" in out

    def test_cooksd_cmd(self, ols_df):
        from openstat.commands.influence_cmds import cmd_cooksd
        s = Session()
        s.df = ols_df
        out = cmd_cooksd(s, "y x1 x2")
        assert "Cook" in out

    def test_outlier_cmd(self, ols_df):
        from openstat.commands.influence_cmds import cmd_outlier
        s = Session()
        s.df = ols_df
        out = cmd_outlier(s, "y x1 x2 threshold(3.0)")
        assert "Outlier" in out
