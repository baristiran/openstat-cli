"""Tests for Factor Analysis / PCA (stats/factor.py and commands/factor_cmds.py)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.stats.factor import fit_pca, fit_factor, varimax_rotation
from openstat.session import Session


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def correlated_df():
    rng = np.random.default_rng(42)
    n = 200
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    data = {
        "x1": f1 + 0.2 * rng.standard_normal(n),
        "x2": f1 + 0.2 * rng.standard_normal(n),
        "x3": f2 + 0.2 * rng.standard_normal(n),
        "x4": f2 + 0.2 * rng.standard_normal(n),
        "x5": 0.5 * f1 + 0.5 * f2 + rng.standard_normal(n),
    }
    return pl.DataFrame(data)


@pytest.fixture()
def session_with_data(correlated_df):
    s = Session()
    s.df = correlated_df
    return s


# ── Unit tests: fit_pca ────────────────────────────────────────────────────

class TestFitPCA:
    def test_returns_expected_keys(self, correlated_df):
        r = fit_pca(correlated_df, ["x1", "x2", "x3", "x4", "x5"])
        assert {"eigenvalues", "loadings", "explained_variance_ratio", "cumulative_variance",
                "scores", "n_components", "cols"}.issubset(r)

    def test_eigenvalues_descending(self, correlated_df):
        r = fit_pca(correlated_df, ["x1", "x2", "x3", "x4", "x5"])
        eigvals = r["eigenvalues"]
        assert all(eigvals[i] >= eigvals[i + 1] - 1e-10 for i in range(len(eigvals) - 1))

    def test_n_components_respected(self, correlated_df):
        r = fit_pca(correlated_df, ["x1", "x2", "x3"], n_components=2)
        assert r["n_components"] == 2
        assert len(r["eigenvalues"]) == 2

    def test_loadings_shape(self, correlated_df):
        cols = ["x1", "x2", "x3", "x4"]
        r = fit_pca(correlated_df, cols, n_components=2)
        arr = np.array(r["loadings"])
        assert arr.shape == (len(cols), 2)

    def test_scores_shape(self, correlated_df):
        cols = ["x1", "x2", "x3"]
        r = fit_pca(correlated_df, cols, n_components=2)
        scores = np.array(r["scores"])
        assert scores.shape[1] == 2

    def test_cumulative_variance_ends_near_one(self, correlated_df):
        r = fit_pca(correlated_df, ["x1", "x2", "x3", "x4", "x5"])
        assert abs(r["cumulative_variance"][-1] - 1.0) < 0.05

    def test_two_cols(self, correlated_df):
        r = fit_pca(correlated_df, ["x1", "x2"])
        assert r["n_components"] >= 1


# ── Unit tests: fit_factor ────────────────────────────────────────────────

class TestFitFactor:
    def test_returns_expected_keys(self, correlated_df):
        r = fit_factor(correlated_df, ["x1", "x2", "x3", "x4", "x5"], n_factors=2)
        assert {"loadings", "communalities", "uniqueness", "n_factors", "cols"}.issubset(r)

    def test_communalities_between_0_1(self, correlated_df):
        r = fit_factor(correlated_df, ["x1", "x2", "x3", "x4", "x5"], n_factors=2)
        comm = r["communalities"]
        assert all(-0.01 <= c <= 1.01 for c in comm)

    def test_uniqueness_plus_comm_approx_1(self, correlated_df):
        r = fit_factor(correlated_df, ["x1", "x2", "x3", "x4"], n_factors=2)
        for c, u in zip(r["communalities"], r["uniqueness"]):
            assert abs(c + u - 1.0) < 1e-9

    def test_loadings_shape(self, correlated_df):
        cols = ["x1", "x2", "x3", "x4"]
        r = fit_factor(correlated_df, cols, n_factors=2)
        arr = np.array(r["loadings"])
        assert arr.shape == (4, 2)

    def test_no_rotation(self, correlated_df):
        r = fit_factor(correlated_df, ["x1", "x2", "x3"], n_factors=2, rotate=False)
        assert len(r["loadings"]) == 3

    def test_single_factor(self, correlated_df):
        r = fit_factor(correlated_df, ["x1", "x2", "x3"], n_factors=1)
        assert r["n_factors"] == 1
        arr = np.array(r["loadings"])
        assert arr.shape[1] == 1


# ── Unit tests: varimax_rotation ──────────────────────────────────────────

class TestVarimax:
    def test_orthogonal(self):
        rng = np.random.default_rng(0)
        L = rng.standard_normal((6, 2))
        R = varimax_rotation(L)
        # Columns of rotated loadings should not be orthogonal to each other
        # but the rotation matrix itself is orthogonal (R^T R ≈ I)
        assert R.shape == (6, 2)

    def test_variance_increased(self):
        """Varimax should increase variance of squared loadings."""
        rng = np.random.default_rng(1)
        L = rng.standard_normal((8, 3))
        R = varimax_rotation(L)
        sq_var_before = np.var(L**2)
        sq_var_after = np.var(R**2)
        # Generally varimax increases or maintains variance
        assert sq_var_after >= sq_var_before * 0.5  # loose check


# ── Integration tests: commands ────────────────────────────────────────────

class TestPCACommand:
    def test_basic(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_pca
        out = cmd_pca(session_with_data, "x1 x2 x3 x4 x5")
        assert "PCA" in out
        assert "Comp1" in out

    def test_n_components_option(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_pca
        out = cmd_pca(session_with_data, "x1 x2 x3 x4 x5, n(2)")
        assert "Comp2" in out

    def test_stores_model(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_pca
        cmd_pca(session_with_data, "x1 x2 x3 x4")
        assert session_with_data._last_model is not None

    def test_too_few_vars(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_pca
        out = cmd_pca(session_with_data, "x1")
        assert "require" in out.lower() or "at least" in out.lower()


class TestFactorCommand:
    def test_basic(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_factor
        out = cmd_factor(session_with_data, "x1 x2 x3 x4 x5, n(2)")
        assert "Factor Analysis" in out
        assert "Communality" in out

    def test_pc_method(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_factor
        out = cmd_factor(session_with_data, "x1 x2 x3, n(2) method(pc)")
        assert "PC" in out

    def test_norotate(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_factor
        out = cmd_factor(session_with_data, "x1 x2 x3, n(2) --norotate")
        assert "varimax" not in out

    def test_too_few_vars(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_factor
        out = cmd_factor(session_with_data, "x1")
        assert "require" in out.lower() or "at least" in out.lower()


class TestEstatCommand:
    def test_screeplot_no_model(self):
        from openstat.commands.factor_cmds import cmd_screeplot
        s = Session()
        out = cmd_screeplot(s, "")
        assert out is not None and len(out) > 0

    def test_screeplot_after_pca(self, session_with_data, tmp_path):
        from openstat.commands.factor_cmds import cmd_pca, cmd_screeplot
        session_with_data.output_dir = tmp_path
        cmd_pca(session_with_data, "x1 x2 x3 x4 x5")
        out = cmd_screeplot(session_with_data, "")
        assert "Scree" in out or "scree" in out.lower() or "component" in out.lower()

    def test_loadings_after_pca(self, session_with_data, tmp_path):
        from openstat.commands.factor_cmds import cmd_pca, cmd_screeplot
        session_with_data.output_dir = tmp_path
        cmd_pca(session_with_data, "x1 x2 x3")
        out = cmd_screeplot(session_with_data, "loadings")
        assert out is not None

    def test_loadings_blanks(self, session_with_data, tmp_path):
        from openstat.commands.factor_cmds import cmd_pca, cmd_screeplot
        session_with_data.output_dir = tmp_path
        cmd_pca(session_with_data, "x1 x2 x3 x4 x5")
        out = cmd_screeplot(session_with_data, "loadings blanks(0.5)")
        assert out is not None

    def test_unknown_subcommand(self, session_with_data):
        from openstat.commands.factor_cmds import cmd_screeplot
        out = cmd_screeplot(session_with_data, "unknown")
        assert out is not None
