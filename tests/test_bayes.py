"""Tests for Bayesian OLS."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.bayesian import bayes_ols


@pytest.fixture()
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestBayesOLS:
    def test_keys(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        assert {"model", "coefficients", "r_squared", "sigma_mean"}.issubset(r)

    def test_coef_keys(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        for name in ["_cons", "x1", "x2"]:
            assert name in r["coefficients"]

    def test_coef_stats(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        for name, stats in r["coefficients"].items():
            assert "mean" in stats and "std" in stats

    def test_true_coefs_recovered(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        assert abs(r["coefficients"]["x1"]["mean"] - 2.0) < 0.3
        assert abs(r["coefficients"]["x2"]["mean"] + 1.5) < 0.3

    def test_r_squared(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        assert r["r_squared"] > 0.8

    def test_credible_interval_coverage(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"], credible_interval=0.95)
        x1_stats = r["coefficients"]["x1"]
        assert x1_stats["ci_95_lo"] < 2.0 < x1_stats["ci_95_hi"]

    def test_n_samples(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"], n_samples=1000)
        assert r["n_samples"] == 1000
        assert len(r["_beta_draws"]) == 1000

    def test_prob_positive(self, reg_df):
        r = bayes_ols(reg_df, "y", ["x1", "x2"])
        # x1 has positive coefficient → high P(β>0)
        assert r["coefficients"]["x1"]["prob_positive"] > 0.9
        # x2 has negative → low P(β>0)
        assert r["coefficients"]["x2"]["prob_positive"] < 0.1


class TestBayesCommand:
    def test_cmd_basic(self, reg_df):
        from openstat.commands.bayes_cmds import cmd_bayes
        s = Session()
        s.df = reg_df
        out = cmd_bayes(s, ": ols y x1 x2")
        assert "Bayesian" in out
        assert "x1" in out
        assert "Post. Mean" in out

    def test_cmd_no_ols(self, reg_df):
        from openstat.commands.bayes_cmds import cmd_bayes
        s = Session()
        s.df = reg_df
        out = cmd_bayes(s, "ols y x1 x2")
        assert "Bayesian" in out

    def test_cmd_no_args(self, reg_df):
        from openstat.commands.bayes_cmds import cmd_bayes
        s = Session()
        s.df = reg_df
        out = cmd_bayes(s, ": ols")
        assert "Usage" in out or "error" in out.lower()

    def test_cmd_stores_model(self, reg_df):
        from openstat.commands.bayes_cmds import cmd_bayes
        s = Session()
        s.df = reg_df
        cmd_bayes(s, ": ols y x1 x2")
        assert s._last_model is not None

    def test_cmd_ci_option(self, reg_df):
        from openstat.commands.bayes_cmds import cmd_bayes
        s = Session()
        s.df = reg_df
        out = cmd_bayes(s, ": ols y x1 x2 ci(0.90)")
        assert "90" in out
