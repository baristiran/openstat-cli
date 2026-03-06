"""Tests for equivalence tests (TOST) and Tobit."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.equiv_tobit import tost_onemean, tost_twomeans, fit_tobit


@pytest.fixture()
def equiv_df():
    rng = np.random.default_rng(42)
    n = 100
    # Two groups with almost the same mean
    vals = np.concatenate([rng.normal(5.0, 1.0, n // 2), rng.normal(5.1, 1.0, n // 2)])
    groups = ["A"] * (n // 2) + ["B"] * (n // 2)
    return pl.DataFrame({"val": vals, "group": groups})


@pytest.fixture()
def censored_df():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    y_latent = 1.5 * x + rng.normal(0, 1, n)
    y = np.maximum(y_latent, 0)  # left-censored at 0
    return pl.DataFrame({"y": y, "x": x})


class TestTOST:
    def test_onemean_keys(self, equiv_df):
        r = tost_onemean(equiv_df, "val", mu=5.0, delta=0.5)
        assert "p_tost" in r and "equivalent_at_alpha" in r

    def test_onemean_close_to_mu(self, equiv_df):
        # Mean is ~5.05, mu=5.0, delta=0.5 → should be equivalent
        r = tost_onemean(equiv_df, "val", mu=5.0, delta=0.5)
        assert r["equivalent_at_alpha"] is True

    def test_twomeans_keys(self, equiv_df):
        r = tost_twomeans(equiv_df, "val", "group", delta=1.0)
        assert "p_tost" in r and "mean_diff" in r

    def test_twomeans_equivalent(self, equiv_df):
        r = tost_twomeans(equiv_df, "val", "group", delta=1.0)
        assert r["equivalent_at_alpha"] is True

    def test_cmd_onemean(self, equiv_df):
        from openstat.commands.equiv_tobit_cmds import cmd_tost
        s = Session()
        s.df = equiv_df
        out = cmd_tost(s, "val mu(5.0) delta(0.5)")
        assert "TOST" in out

    def test_cmd_twomeans(self, equiv_df):
        from openstat.commands.equiv_tobit_cmds import cmd_tost
        s = Session()
        s.df = equiv_df
        out = cmd_tost(s, "val by(group) delta(1.0)")
        assert "TOST" in out


class TestTobit:
    def test_keys(self, censored_df):
        r = fit_tobit(censored_df, "y", ["x"])
        assert "params" in r and "sigma" in r and "log_likelihood" in r

    def test_positive_sigma(self, censored_df):
        r = fit_tobit(censored_df, "y", ["x"])
        assert r["sigma"] > 0

    def test_censored_count(self, censored_df):
        r = fit_tobit(censored_df, "y", ["x"])
        assert r["n_censored_left"] > 0

    def test_coef_positive(self, censored_df):
        r = fit_tobit(censored_df, "y", ["x"])
        # x has positive effect
        assert r["params"]["x"] > 0

    def test_cmd(self, censored_df):
        from openstat.commands.equiv_tobit_cmds import cmd_tobit
        s = Session()
        s.df = censored_df
        out = cmd_tobit(s, "y x ll(0)")
        assert "Tobit" in out
        assert "x" in out
