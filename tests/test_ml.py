"""Tests for ML commands (lasso, ridge, elasticnet, cart, crossval)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session

_sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


@pytest.fixture()
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)
    y = 2 * X1 - 1.5 * X2 + 0.5 * X3 + rng.normal(0, 0.5, n)
    return pl.DataFrame({"y": y, "x1": X1, "x2": X2, "x3": X3})


@pytest.fixture()
def clf_df():
    rng = np.random.default_rng(7)
    n = 120
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    label = (X1 + X2 > 0).astype(str)
    return pl.DataFrame({"label": label.tolist(), "x1": X1, "x2": X2})


@pytest.fixture()
def session_reg(reg_df):
    s = Session()
    s.df = reg_df
    return s


class TestLasso:
    def test_basic(self, reg_df):
        from openstat.stats.ml import fit_lasso
        r = fit_lasso(reg_df, "y", ["x1", "x2", "x3"])
        assert "coefficients" in r
        assert r["r_squared"] > 0.5
        assert r["n_obs"] == 200

    def test_fixed_alpha(self, reg_df):
        from openstat.stats.ml import fit_lasso
        r = fit_lasso(reg_df, "y", ["x1", "x2", "x3"], alpha=0.1)
        assert r["alpha"] == pytest.approx(0.1)

    def test_sparsity(self, reg_df):
        from openstat.stats.ml import fit_lasso
        # high alpha → more zero coefficients
        r = fit_lasso(reg_df, "y", ["x1", "x2", "x3"], alpha=100.0)
        assert r["n_nonzero"] < 3


class TestRidge:
    def test_basic(self, reg_df):
        from openstat.stats.ml import fit_ridge
        r = fit_ridge(reg_df, "y", ["x1", "x2", "x3"])
        assert r["r_squared"] > 0.5

    def test_no_zeros(self, reg_df):
        from openstat.stats.ml import fit_ridge
        r = fit_ridge(reg_df, "y", ["x1", "x2", "x3"], alpha=1.0)
        # ridge shrinks but doesn't zero out
        assert all(v != 0 for v in r["coefficients"].values())

    def test_keys(self, reg_df):
        from openstat.stats.ml import fit_ridge
        r = fit_ridge(reg_df, "y", ["x1", "x2", "x3"])
        assert {"method", "r_squared", "mse", "rmse", "n_obs"}.issubset(r)


class TestElasticNet:
    def test_basic(self, reg_df):
        from openstat.stats.ml import fit_elasticnet
        r = fit_elasticnet(reg_df, "y", ["x1", "x2", "x3"])
        assert r["r_squared"] > 0.3

    def test_l1_ratio(self, reg_df):
        from openstat.stats.ml import fit_elasticnet
        r = fit_elasticnet(reg_df, "y", ["x1", "x2", "x3"], alpha=0.1, l1_ratio=0.7)
        assert r["l1_ratio"] == pytest.approx(0.7)


class TestCART:
    def test_regression(self, reg_df):
        from openstat.stats.ml import fit_cart
        r = fit_cart(reg_df, "y", ["x1", "x2", "x3"], task="regression")
        assert r["r_squared"] > 0
        assert "feature_importances" in r
        assert r["n_leaves"] >= 1

    def test_classification(self, clf_df):
        from openstat.stats.ml import fit_cart
        r = fit_cart(clf_df, "label", ["x1", "x2"], task="classification")
        assert r["accuracy"] > 0.5
        assert r["task"] == "classification"

    def test_max_depth(self, reg_df):
        from openstat.stats.ml import fit_cart
        r = fit_cart(reg_df, "y", ["x1", "x2", "x3"], task="regression", max_depth=2)
        assert r["max_depth"] == 2

    def test_importances_sum_one(self, reg_df):
        from openstat.stats.ml import fit_cart
        r = fit_cart(reg_df, "y", ["x1", "x2", "x3"])
        total = sum(r["feature_importances"].values())
        assert abs(total - 1.0) < 1e-9


class TestCrossVal:
    def test_ols(self, reg_df):
        from openstat.stats.ml import cross_validate_model
        r = cross_validate_model(reg_df, "y", ["x1", "x2", "x3"], method="ols", k=5)
        assert r["k_folds"] == 5
        assert r["mean_score"] > 0.5
        assert len(r["scores"]) == 5

    def test_lasso(self, reg_df):
        from openstat.stats.ml import cross_validate_model
        r = cross_validate_model(reg_df, "y", ["x1", "x2", "x3"], method="lasso", k=3)
        assert r["mean_score"] > 0

    def test_keys(self, reg_df):
        from openstat.stats.ml import cross_validate_model
        r = cross_validate_model(reg_df, "y", ["x1", "x2"], method="ridge", k=3)
        assert {"mean_score", "std_score", "min_score", "max_score", "scores"}.issubset(r)


class TestMLCommands:
    def test_lasso_cmd(self, session_reg):
        from openstat.commands.ml_cmds import cmd_lasso
        out = cmd_lasso(session_reg, "y x1 x2 x3")
        assert "Lasso" in out
        assert "r_squared" in out

    def test_ridge_cmd(self, session_reg):
        from openstat.commands.ml_cmds import cmd_ridge
        out = cmd_ridge(session_reg, "y x1 x2 x3")
        assert "Ridge" in out

    def test_elasticnet_cmd(self, session_reg):
        from openstat.commands.ml_cmds import cmd_elasticnet
        out = cmd_elasticnet(session_reg, "y x1 x2 x3")
        assert "ElasticNet" in out

    def test_cart_cmd(self, session_reg):
        from openstat.commands.ml_cmds import cmd_cart
        out = cmd_cart(session_reg, "y x1 x2 x3 depth(3)")
        assert "CART" in out
        assert "r_squared" in out or "Feature" in out

    def test_crossval_cmd(self, session_reg):
        from openstat.commands.ml_cmds import cmd_crossval
        out = cmd_crossval(session_reg, "y x1 x2 x3 method(ols) k(3)")
        assert "Cross-Validation" in out
        assert "Mean score" in out

    def test_lasso_no_args(self, session_reg):
        from openstat.commands.ml_cmds import cmd_lasso
        out = cmd_lasso(session_reg, "")
        assert "Usage" in out

    def test_stores_model(self, session_reg):
        from openstat.commands.ml_cmds import cmd_lasso
        cmd_lasso(session_reg, "y x1 x2 x3")
        assert session_reg._last_model is not None
