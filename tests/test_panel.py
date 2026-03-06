"""Tests for panel data models (F1)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.panel_cmds import cmd_xtset, cmd_xtreg, cmd_hausman

# Check if linearmodels is available
try:
    import linearmodels
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False


@pytest.fixture
def panel_session(tmp_path):
    """Create a session with synthetic panel data."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n = n_entities * n_periods

    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    entity_effect = np.repeat(np.random.randn(n_entities) * 2, n_periods)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + entity_effect + np.random.randn(n) * 0.5

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "firm": entity,
        "year": time,
        "y": y,
        "x1": x1,
        "x2": x2,
    })
    return s


class TestXtset:
    def test_xtset_basic(self, panel_session):
        result = cmd_xtset(panel_session, "firm year")
        assert "firm" in result
        assert "year" in result
        assert panel_session._panel_var == "firm"
        assert panel_session._time_var == "year"

    def test_xtset_missing_col(self, panel_session):
        result = cmd_xtset(panel_session, "nonexistent year")
        assert "not found" in result

    def test_xtset_usage(self, panel_session):
        result = cmd_xtset(panel_session, "")
        assert "Usage" in result


@pytest.mark.skipif(not HAS_LINEARMODELS, reason="linearmodels not installed")
class TestXtreg:
    def test_xtreg_fe(self, panel_session):
        cmd_xtset(panel_session, "firm year")
        result = cmd_xtreg(panel_session, "y ~ x1 + x2, fe")
        assert "Panel FE" in result or "Coef" in result

    def test_xtreg_re(self, panel_session):
        cmd_xtset(panel_session, "firm year")
        result = cmd_xtreg(panel_session, "y ~ x1 + x2, re")
        assert "Panel RE" in result or "Coef" in result

    def test_xtreg_no_xtset(self, panel_session):
        result = cmd_xtreg(panel_session, "y ~ x1 + x2, fe")
        assert "not set" in result

    def test_xtreg_no_estimator(self, panel_session):
        cmd_xtset(panel_session, "firm year")
        result = cmd_xtreg(panel_session, "y ~ x1 + x2")
        assert "Usage" in result or "Specify estimator" in result

    def test_hausman(self, panel_session):
        cmd_xtset(panel_session, "firm year")
        cmd_xtreg(panel_session, "y ~ x1 + x2, fe")
        cmd_xtreg(panel_session, "y ~ x1 + x2, re")
        result = cmd_hausman(panel_session, "")
        assert "Hausman" in result
        assert "chi2" in result
        assert "p-value" in result

    def test_hausman_no_models(self, panel_session):
        result = cmd_hausman(panel_session, "")
        assert "Run both" in result
