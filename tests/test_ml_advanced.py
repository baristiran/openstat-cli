"""Tests for advanced ML commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")

from openstat.session import Session
from openstat.stats.ml_advanced import fit_random_forest, fit_gradient_boosting, fit_svm, fit_tsne


@pytest.fixture()
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture()
def cls_df():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (x1 + x2 > 0).astype(int)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestRandomForest:
    def test_regression_keys(self, reg_df):
        r = fit_random_forest(reg_df, "y", ["x1", "x2"])
        assert "r_squared" in r and "feature_importances" in r

    def test_classification_keys(self, cls_df):
        r = fit_random_forest(cls_df, "y", ["x1", "x2"], task="classification")
        assert "accuracy" in r

    def test_r2_positive(self, reg_df):
        r = fit_random_forest(reg_df, "y", ["x1", "x2"])
        assert r["r_squared"] > 0.5

    def test_cmd(self, reg_df):
        from openstat.commands.ml_adv_cmds import cmd_randomforest
        s = Session()
        s.df = reg_df
        out = cmd_randomforest(s, "y x1 x2 n(50)")
        assert "Random Forest" in out


class TestGBM:
    def test_keys(self, reg_df):
        r = fit_gradient_boosting(reg_df, "y", ["x1", "x2"])
        assert "r_squared" in r and "feature_importances" in r

    def test_r2_positive(self, reg_df):
        r = fit_gradient_boosting(reg_df, "y", ["x1", "x2"], n_estimators=50)
        assert r["r_squared"] > 0.5

    def test_cmd(self, reg_df):
        from openstat.commands.ml_adv_cmds import cmd_gbm
        s = Session()
        s.df = reg_df
        out = cmd_gbm(s, "y x1 x2 n(50)")
        assert "Gradient Boosting" in out


class TestSVM:
    def test_regression_keys(self, reg_df):
        r = fit_svm(reg_df, "y", ["x1", "x2"])
        assert "r_squared" in r

    def test_classification_keys(self, cls_df):
        r = fit_svm(cls_df, "y", ["x1", "x2"], task="classification")
        assert "accuracy" in r

    def test_cmd(self, reg_df):
        from openstat.commands.ml_adv_cmds import cmd_svm
        s = Session()
        s.df = reg_df
        out = cmd_svm(s, "y x1 x2 kernel(rbf)")
        assert "SVM" in out


class TestTSNE:
    def test_keys(self, reg_df):
        r = fit_tsne(reg_df, ["x1", "x2"], n_components=2)
        assert "embedding" in r
        assert len(r["embedding"]) == len(reg_df)

    def test_cmd(self, reg_df):
        from openstat.commands.ml_adv_cmds import cmd_tsne
        s = Session()
        s.df = reg_df
        out = cmd_tsne(s, "x1 x2 components(2)")
        assert "tsne1" in s.df.columns
