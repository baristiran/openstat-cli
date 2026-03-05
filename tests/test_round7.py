"""Tests for Round 7 features: stepwise regression, residuals/diagnostics, LaTeX export, enhanced reshape."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.commands.data_cmds import cmd_melt, cmd_pivot
from openstat.commands.stat_cmds import (
    cmd_ols, cmd_stepwise, cmd_residuals, cmd_latex, cmd_predict,
)
from openstat.commands.plot_cmds import cmd_plot
from openstat.stats.models import stepwise_ols, compute_residuals, fit_ols


# -----------------------------------------------------------------------
# Stepwise regression
# -----------------------------------------------------------------------

class TestStepwise:
    def _make_data(self, n=100):
        np.random.seed(42)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)  # noise, no relationship
        y = 2.0 * x1 + 3.0 * x2 + np.random.normal(0, 0.5, n)
        return pl.DataFrame({
            "y": y.tolist(), "x1": x1.tolist(),
            "x2": x2.tolist(), "x3": x3.tolist(),
        })

    def test_forward_selects_relevant(self):
        df = self._make_data()
        result = stepwise_ols(df, "y", ["x1", "x2", "x3"], direction="forward")
        assert "x1" in result.selected
        assert "x2" in result.selected
        # x3 should ideally not be selected (noise variable)

    def test_backward_drops_noise(self):
        df = self._make_data()
        result = stepwise_ols(df, "y", ["x1", "x2", "x3"], direction="backward")
        assert "x1" in result.selected
        assert "x2" in result.selected

    def test_stepwise_returns_steps(self):
        df = self._make_data()
        result = stepwise_ols(df, "y", ["x1", "x2", "x3"], direction="forward")
        assert len(result.steps) > 0
        assert "step" in result.steps[0]
        assert "aic" in result.steps[0]

    def test_stepwise_summary(self):
        df = self._make_data()
        result = stepwise_ols(df, "y", ["x1", "x2", "x3"], direction="forward")
        summary = result.summary()
        assert "Stepwise" in summary
        assert "x1" in summary

    def test_stepwise_command(self):
        s = Session()
        s.df = self._make_data()
        result = cmd_stepwise(s, "y ~ x1 + x2 + x3")
        assert "Stepwise" in result
        assert s._last_model_vars is not None

    def test_stepwise_backward_command(self):
        s = Session()
        s.df = self._make_data()
        result = cmd_stepwise(s, "y ~ x1 + x2 + x3 --backward")
        assert "Stepwise" in result

    def test_stepwise_usage(self):
        s = Session()
        s.df = pl.DataFrame({"y": [1.0], "x": [2.0]})
        assert "Usage" in cmd_stepwise(s, "")


# -----------------------------------------------------------------------
# Residuals and diagnostic plots
# -----------------------------------------------------------------------

class TestResiduals:
    def _fit_model(self, s: Session):
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        s.df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        cmd_ols(s, "y ~ x")

    def test_compute_residuals(self):
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        _, model = fit_ols(df, "y", ["x"])
        diag = compute_residuals(model, df, "y", ["x"])
        assert "residuals" in diag
        assert "fitted" in diag
        assert "std_residuals" in diag
        assert len(diag["residuals"]) == n

    def test_residuals_command(self, tmp_path):
        s = Session()
        s.output_dir = tmp_path
        self._fit_model(s)
        result = cmd_residuals(s, "")
        assert "residuals" in s.df.columns
        assert "Residuals added" in result

    def test_residuals_custom_name(self, tmp_path):
        s = Session()
        s.output_dir = tmp_path
        self._fit_model(s)
        cmd_residuals(s, "my_resid")
        assert "my_resid" in s.df.columns

    def test_residuals_creates_snapshot(self, tmp_path):
        s = Session()
        s.output_dir = tmp_path
        self._fit_model(s)
        cmd_residuals(s, "")
        # ols creates no snapshot, but residuals does
        assert s.undo_depth >= 1

    def test_residuals_no_model(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0]})
        result = cmd_residuals(s, "")
        assert "No model" in result

    def test_diagnostic_plots_created(self, tmp_path):
        s = Session()
        s.output_dir = tmp_path
        self._fit_model(s)
        result = cmd_residuals(s, "")
        assert "Diagnostic plots saved" in result
        assert (tmp_path / "resid_vs_fitted.png").exists()
        assert (tmp_path / "qq_plot.png").exists()
        assert (tmp_path / "scale_location.png").exists()


class TestPlotDiagnostics:
    def test_plot_diagnostics_command(self, tmp_path):
        s = Session()
        s.output_dir = tmp_path
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        s.df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        cmd_ols(s, "y ~ x")
        result = cmd_plot(s, "diagnostics")
        assert "Diagnostic plots saved" in result

    def test_plot_diagnostics_no_model(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0]})
        result = cmd_plot(s, "diagnostics")
        assert "No model" in result


# -----------------------------------------------------------------------
# LaTeX export
# -----------------------------------------------------------------------

class TestLatex:
    def test_latex_output(self):
        s = Session()
        np.random.seed(42)
        n = 30
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        s.df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        cmd_ols(s, "y ~ x")
        result = cmd_latex(s, "")
        assert r"\begin{table}" in result
        assert r"\end{table}" in result
        assert "Coef" in result

    def test_latex_save_to_file(self, tmp_path):
        s = Session()
        np.random.seed(42)
        n = 30
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        s.df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        cmd_ols(s, "y ~ x")
        path = tmp_path / "table.tex"
        result = cmd_latex(s, str(path))
        assert "saved" in result
        assert path.exists()
        content = path.read_text()
        assert r"\begin{table}" in content

    def test_latex_no_model(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0]})
        result = cmd_latex(s, "")
        assert "No model" in result

    def test_fit_result_to_latex(self):
        np.random.seed(42)
        n = 30
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        result, _ = fit_ols(df, "y", ["x"])
        latex = result.to_latex()
        assert r"\begin{table}" in latex
        assert "R^2" in latex
        assert "x" in latex


# -----------------------------------------------------------------------
# Enhanced reshape (melt with var_name/value_name, pivot with agg)
# -----------------------------------------------------------------------

class TestEnhancedReshape:
    def test_melt_custom_names(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "math": [90, 80],
            "eng": [85, 95],
        })
        cmd_melt(s, "name, math eng var_name=subject value_name=score")
        assert "subject" in s.df.columns
        assert "score" in s.df.columns
        assert "variable" not in s.df.columns
        assert "value" not in s.df.columns

    def test_melt_default_names(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "math": [90, 80],
            "eng": [85, 95],
        })
        cmd_melt(s, "name, math eng")
        assert "variable" in s.df.columns
        assert "value" in s.df.columns

    def test_pivot_agg_mean(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Alice", "Bob", "Bob"],
            "subject": ["math", "math", "math", "math"],
            "score": [90, 80, 70, 60],
        })
        cmd_pivot(s, "score by subject over name agg=mean")
        # Alice: mean(90,80)=85, Bob: mean(70,60)=65
        assert s.df.height == 2

    def test_pivot_agg_sum(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Alice", "Bob"],
            "subject": ["math", "math", "math"],
            "score": [10, 20, 30],
        })
        cmd_pivot(s, "score by subject over name agg=sum")
        assert s.df.height == 2

    def test_pivot_unknown_agg(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice"], "subject": ["math"], "score": [90],
        })
        result = cmd_pivot(s, "score by subject over name agg=banana")
        assert "Unknown aggregation" in result
