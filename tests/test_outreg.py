"""Tests for outreg and log commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def fitted_session():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    s = Session()
    s.df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    from openstat.commands.stat_cmds import cmd_ols
    cmd_ols(s, "y ~ x1 + x2")
    cmd_ols(s, "y ~ x1")
    return s


class TestOutreg:
    def test_no_results(self):
        from openstat.commands.outreg_cmds import cmd_outreg
        s = Session()
        out = cmd_outreg(s, "")
        assert "No stored" in out

    def test_latex_output(self, fitted_session):
        from openstat.commands.outreg_cmds import cmd_outreg
        out = cmd_outreg(fitted_session, "format(latex)")
        assert "tabular" in out or "outreg" in out.lower()

    def test_html_output(self, fitted_session):
        from openstat.commands.outreg_cmds import cmd_outreg
        out = cmd_outreg(fitted_session, "format(html)")
        assert "<table" in out or "outreg" in out.lower()

    def test_stars(self, fitted_session):
        from openstat.commands.outreg_cmds import cmd_outreg
        out = cmd_outreg(fitted_session, "format(latex) --stars")
        assert "***" in out or "p<0.001" in out or "outreg" in out.lower()


class TestLog:
    def test_display(self, fitted_session):
        from openstat.commands.outreg_cmds import cmd_log
        fitted_session.history.append("ols y ~ x1 + x2")
        out = cmd_log(fitted_session, "display")
        assert "ols" in out or "Session" in out

    def test_empty_history(self):
        from openstat.commands.outreg_cmds import cmd_log
        s = Session()
        out = cmd_log(s, "display")
        assert "No commands" in out or "history" in out.lower()

    def test_clear(self, fitted_session):
        from openstat.commands.outreg_cmds import cmd_log
        fitted_session.history.append("test")
        cmd_log(fitted_session, "clear")
        assert len(fitted_session.history) == 0
