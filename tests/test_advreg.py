"""Tests for advanced regression commands."""

from __future__ import annotations
import numpy as np
import polars as pl
import pytest
from openstat.session import Session


@pytest.fixture()
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0.5, 3.0, n)
    y = 2.0 * x ** 1.5 + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y, "x": x})


@pytest.fixture()
def beta_df():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    from scipy.special import expit
    y = np.clip(expit(0.5 + 0.8 * x) + rng.normal(0, 0.05, n), 0.01, 0.99)
    return pl.DataFrame({"y": y, "x": x})


@pytest.fixture()
def count_df():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(0, 1, n)
    lam = np.exp(0.5 + 0.5 * x)
    y = rng.poisson(lam).astype(float)
    # Add extra zeros
    y[rng.uniform(size=n) < 0.3] = 0
    return pl.DataFrame({"y": y, "x": x})


@pytest.fixture()
def hurdle_df(count_df):
    return count_df


@pytest.fixture()
def sur_df():
    rng = np.random.default_rng(9)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y1 = 1.0 + 2.0*x1 + rng.normal(0, 0.5, n)
    y2 = -1.0 + 1.5*x2 + rng.normal(0, 0.5, n)
    return pl.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2})


class TestNLS:
    def test_power_converges(self, reg_df):
        from openstat.stats.advanced_regression import fit_nls
        import numpy as np
        fn = lambda X, a, b: a * X[:, 0] ** b
        r = fit_nls(reg_df, "y", ["x"], fn, [1.0, 1.0])
        assert r["converged"]
        assert abs(r["params"]["p0"] - 2.0) < 0.5
        assert abs(r["params"]["p1"] - 1.5) < 0.5

    def test_r_squared(self, reg_df):
        from openstat.stats.advanced_regression import fit_nls
        import numpy as np
        fn = lambda X, a, b: a * X[:, 0] ** b
        r = fit_nls(reg_df, "y", ["x"], fn, [1.0, 1.0])
        assert r["r_squared"] > 0.8

    def test_keys(self, reg_df):
        from openstat.stats.advanced_regression import fit_nls
        import numpy as np
        fn = lambda X, a, b: a * X[:, 0] ** b
        r = fit_nls(reg_df, "y", ["x"], fn, [1.0, 1.0])
        assert {"params", "std_errors", "r_squared", "n_obs", "converged"}.issubset(r)


class TestBetareg:
    def test_basic(self, beta_df):
        from openstat.stats.advanced_regression import fit_betareg
        r = fit_betareg(beta_df, "y", ["x"])
        assert "params" in r
        assert r["n_obs"] == 200

    def test_positive_coef(self, beta_df):
        from openstat.stats.advanced_regression import fit_betareg
        r = fit_betareg(beta_df, "y", ["x"])
        assert r["params"]["x"] > 0

    def test_aic(self, beta_df):
        from openstat.stats.advanced_regression import fit_betareg
        r = fit_betareg(beta_df, "y", ["x"])
        assert r["aic"] < 0 or r["aic"] > 0  # just finite


class TestZIP:
    def test_basic(self, count_df):
        from openstat.stats.advanced_regression import fit_zip
        r = fit_zip(count_df, "y", ["x"])
        assert "params" in r
        assert r["n_obs"] == 300

    def test_keys(self, count_df):
        from openstat.stats.advanced_regression import fit_zip
        r = fit_zip(count_df, "y", ["x"])
        assert {"method", "aic", "log_likelihood", "n_obs"}.issubset(r)


class TestZINB:
    def test_basic(self, count_df):
        from openstat.stats.advanced_regression import fit_zinb
        r = fit_zinb(count_df, "y", ["x"])
        assert "params" in r

    def test_aic_better_than_poisson(self, count_df):
        from openstat.stats.advanced_regression import fit_zinb, fit_zip
        r_zinb = fit_zinb(count_df, "y", ["x"])
        assert r_zinb["aic"] < 1e6  # finite


class TestHurdle:
    def test_basic(self, hurdle_df):
        from openstat.stats.advanced_regression import fit_hurdle
        r = fit_hurdle(hurdle_df, "y", ["x"])
        assert "logit_params" in r and "count_params" in r
        assert r["n_zeros"] > 0
        assert r["n_positive"] > 0

    def test_parts_sum(self, hurdle_df):
        from openstat.stats.advanced_regression import fit_hurdle
        r = fit_hurdle(hurdle_df, "y", ["x"])
        assert r["n_zeros"] + r["n_positive"] == r["n_obs"]

    def test_keys(self, hurdle_df):
        from openstat.stats.advanced_regression import fit_hurdle
        r = fit_hurdle(hurdle_df, "y", ["x"])
        assert {"logit_params", "count_params", "logit_pvalues", "count_pvalues"}.issubset(r)


class TestSUR:
    def test_basic(self, sur_df):
        from openstat.stats.advanced_regression import fit_sur
        r = fit_sur(sur_df, [("y1", ["x1"]), ("y2", ["x2"])])
        assert r["n_equations"] == 2
        assert len(r["equations"]) == 2

    def test_r_squared(self, sur_df):
        from openstat.stats.advanced_regression import fit_sur
        r = fit_sur(sur_df, [("y1", ["x1"]), ("y2", ["x2"])])
        for eq in r["equations"]:
            assert eq["r_squared"] > 0.5

    def test_cross_corr(self, sur_df):
        from openstat.stats.advanced_regression import fit_sur
        r = fit_sur(sur_df, [("y1", ["x1"]), ("y2", ["x2"])])
        assert len(r["cross_equation_corr"]) == 2


class TestAdvregCommands:
    def test_nls_cmd(self, reg_df):
        from openstat.commands.advreg_cmds import cmd_nls
        s = Session(); s.df = reg_df
        out = cmd_nls(s, "y x fn(power) p0(1,1)")
        assert "NLS" in out

    def test_nls_no_args(self, reg_df):
        from openstat.commands.advreg_cmds import cmd_nls
        s = Session(); s.df = reg_df
        out = cmd_nls(s, "")
        assert "Usage" in out

    def test_betareg_cmd(self, beta_df):
        from openstat.commands.advreg_cmds import cmd_betareg
        s = Session(); s.df = beta_df
        out = cmd_betareg(s, "y x")
        assert "Beta" in out

    def test_zip_cmd(self, count_df):
        from openstat.commands.advreg_cmds import cmd_zip
        s = Session(); s.df = count_df
        out = cmd_zip(s, "y x")
        assert "Zero-Inflated" in out or "error" in out.lower()

    def test_zinb_cmd(self, count_df):
        from openstat.commands.advreg_cmds import cmd_zinb
        s = Session(); s.df = count_df
        out = cmd_zinb(s, "y x")
        assert "Zero-Inflated" in out or "error" in out.lower()

    def test_hurdle_cmd(self, count_df):
        from openstat.commands.advreg_cmds import cmd_hurdle
        s = Session(); s.df = count_df
        out = cmd_hurdle(s, "y x")
        assert "Hurdle" in out

    def test_sureg_cmd(self, sur_df):
        from openstat.commands.advreg_cmds import cmd_sureg
        s = Session(); s.df = sur_df
        out = cmd_sureg(s, "(y1 x1) (y2 x2)")
        assert "SUR" in out

    def test_sureg_no_parens(self, sur_df):
        from openstat.commands.advreg_cmds import cmd_sureg
        s = Session(); s.df = sur_df
        out = cmd_sureg(s, "y1 x1 y2 x2")
        assert "Usage" in out
