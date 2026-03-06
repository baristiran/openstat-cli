"""Tests for Jupyter integration (F9)."""

import pytest

from openstat.session import Session


class TestJupyterModule:
    def test_import(self):
        """Jupyter module can be imported."""
        from openstat.jupyter import load_ipython_extension
        assert callable(load_ipython_extension)

    def test_magic_class_exists(self):
        from openstat.jupyter.magic import OpenStatMagics
        assert OpenStatMagics is not None

    def test_display_helpers(self):
        from openstat.jupyter.display import fit_result_to_html, dataframe_to_html
        assert callable(fit_result_to_html)
        assert callable(dataframe_to_html)


class TestFitResultHTML:
    def test_to_html(self):
        """FitResult.to_html() returns HTML string."""
        from openstat.stats.models import FitResult
        result = FitResult(
            model_type="OLS",
            formula="y ~ x1",
            dep_var="y",
            indep_vars=["const", "x1"],
            n_obs=100,
            params={"const": 1.0, "x1": 2.0},
            std_errors={"const": 0.1, "x1": 0.2},
            t_values={"const": 10.0, "x1": 10.0},
            p_values={"const": 0.0001, "x1": 0.0001},
            conf_int_low={"const": 0.8, "x1": 1.6},
            conf_int_high={"const": 1.2, "x1": 2.4},
            r_squared=0.95,
        )
        html = result.to_html()
        assert "<" in html  # contains HTML tags
        assert "OLS" in html
