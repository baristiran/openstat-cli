"""Statistical correctness tests.

These tests verify that models produce CORRECT numerical results,
not just that they run without errors.
"""

import math

import numpy as np
import polars as pl
import pytest

from openstat.stats.models import fit_ols, fit_logit


class TestOLSCorrectness:
    """Verify OLS produces known-correct results."""

    def test_perfect_linear(self):
        """y = 2x + 1 → slope=2, intercept=1, R²=1."""
        df = pl.DataFrame({
            "y": [3.0, 5.0, 7.0, 9.0, 11.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result, _ = fit_ols(df, "y", ["x"])
        assert abs(result.params["x"] - 2.0) < 0.001
        assert abs(result.params["_cons"] - 1.0) < 0.001
        assert result.r_squared is not None
        assert result.r_squared > 0.999

    def test_zero_slope(self):
        """y = 5 (constant) → slope≈0, intercept≈5."""
        df = pl.DataFrame({
            "y": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        result, _ = fit_ols(df, "y", ["x"])
        assert abs(result.params["x"]) < 0.001
        assert abs(result.params["_cons"] - 5.0) < 0.001

    def test_multiple_regression(self):
        """y = 1 + 2*x1 + 3*x2."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n) * 0.01  # tiny noise
        df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
        result, _ = fit_ols(df, "y", ["x1", "x2"])
        assert abs(result.params["x1"] - 2.0) < 0.1
        assert abs(result.params["x2"] - 3.0) < 0.1
        assert abs(result.params["_cons"] - 1.0) < 0.1
        assert result.r_squared > 0.99

    def test_significance_direction(self):
        """Strong predictor should have p < 0.05."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 * x + np.random.randn(n) * 0.5
        df = pl.DataFrame({"y": y, "x": x})
        result, _ = fit_ols(df, "y", ["x"])
        assert result.p_values["x"] < 0.001
        assert result.params["x"] > 0

    def test_robust_changes_se(self):
        """Robust SEs should differ from regular SEs."""
        np.random.seed(42)
        n = 50
        x = np.linspace(1, 10, n)
        # Heteroscedastic errors: variance increases with x
        y = 2 * x + np.random.randn(n) * x
        df = pl.DataFrame({"y": y, "x": x})
        regular, _ = fit_ols(df, "y", ["x"], robust=False)
        robust, _ = fit_ols(df, "y", ["x"], robust=True)
        # Coefficients should be the same
        assert abs(regular.params["x"] - robust.params["x"]) < 0.001
        # But standard errors should differ
        assert regular.std_errors["x"] != robust.std_errors["x"]

    def test_nobs_correct(self):
        df = pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result, _ = fit_ols(df, "y", ["x"])
        assert result.n_obs == 5

    def test_conf_intervals_contain_true_value(self):
        """95% CI should contain the true parameter value."""
        np.random.seed(42)
        x = np.linspace(0, 10, 200)
        y = 3.0 * x + 5.0 + np.random.randn(200) * 2
        df = pl.DataFrame({"y": y, "x": x})
        result, _ = fit_ols(df, "y", ["x"])
        # True slope = 3.0 should be in CI
        assert result.conf_int_low["x"] < 3.0 < result.conf_int_high["x"]
        # True intercept = 5.0 should be in CI
        assert result.conf_int_low["_cons"] < 5.0 < result.conf_int_high["_cons"]


class TestLogitCorrectness:
    """Verify logistic regression produces correct results."""

    def test_clear_relationship(self):
        """Clear (but not perfect) relationship → positive coef, low p-value."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        prob = 1 / (1 + np.exp(-3 * x))  # strong relationship
        y = (np.random.rand(n) < prob).astype(float)
        df = pl.DataFrame({"y": y, "x": x})
        result, _ = fit_logit(df, "y", ["x"])
        # x coefficient should be positive (higher x → y=1)
        assert result.params["x"] > 0
        assert result.p_values["x"] < 0.05

    def test_non_binary_raises(self):
        df = pl.DataFrame({
            "y": [0, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        with pytest.raises(ValueError, match="binary"):
            fit_logit(df, "y", ["x"])

    def test_pseudo_r2_range(self):
        """Pseudo R² should be between 0 and 1."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        prob = 1 / (1 + np.exp(-2 * x))
        y = (np.random.rand(n) < prob).astype(float)
        df = pl.DataFrame({"y": y, "x": x})
        result, _ = fit_logit(df, "y", ["x"])
        assert 0 < result.pseudo_r2 < 1


class TestDiagnostics:
    """Test that warnings are correctly generated."""

    def test_multicollinearity_warning(self):
        """Perfectly correlated predictors should trigger warning."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 * 2 + 0.001 * np.random.randn(n)  # near-perfect correlation
        y = x1 + np.random.randn(n)
        df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
        result, _ = fit_ols(df, "y", ["x1", "x2"])
        assert any("multicollinearity" in w.lower() for w in result.warnings)

    def test_dropped_missing_note(self):
        """Missing values should generate a note."""
        df = pl.DataFrame({
            "y": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        result, _ = fit_ols(df, "y", ["x"])
        assert any("dropped" in w.lower() for w in result.warnings)
        assert result.n_obs == 7

    def test_too_few_observations(self):
        """Should raise error with too few observations."""
        df = pl.DataFrame({
            "y": [1.0, 2.0, 3.0],
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
        })
        with pytest.raises(ValueError, match="Too few"):
            fit_ols(df, "y", ["x1", "x2"])

    def test_missing_column_error(self):
        df = pl.DataFrame({"y": [1.0, 2.0], "x": [3.0, 4.0]})
        with pytest.raises(ValueError, match="not found"):
            fit_ols(df, "y", ["nonexistent"])


class TestEdgeCases:
    """Edge cases that should be handled gracefully."""

    def test_all_same_y(self):
        """Constant dependent variable — should still run (R²=0 or error)."""
        df = pl.DataFrame({
            "y": [5.0] * 10,
            "x": list(range(10)),
        })
        # This should either work (with R²≈0) or give a clear error
        try:
            result, _ = fit_ols(df, "y", ["x"])
            # If it works, R² should be ~0 and slope should be ~0
            assert abs(result.params["x"]) < 0.001
        except ValueError:
            pass  # Also acceptable to raise

    def test_single_predictor_with_noise(self):
        """Basic sanity: noisy linear relationship."""
        np.random.seed(123)
        n = 30
        x = np.random.randn(n)
        y = 0.5 * x + np.random.randn(n) * 0.1
        df = pl.DataFrame({"y": y, "x": x})
        result, _ = fit_ols(df, "y", ["x"])
        assert 0.3 < result.params["x"] < 0.7

    def test_unicode_column_names(self):
        """Unicode column names should work."""
        df = pl.DataFrame({
            "gelir": [100.0, 200.0, 300.0, 400.0, 500.0],
            "puan": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result, _ = fit_ols(df, "puan", ["gelir"])
        assert "gelir" in result.params

    def test_empty_after_dropna(self):
        """All-null column should raise clear error."""
        df = pl.DataFrame({
            "y": [None, None, None],
            "x": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="No observations"):
            fit_ols(df, "y", ["x"])
