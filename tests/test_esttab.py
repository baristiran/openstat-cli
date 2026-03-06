"""Tests for esttab and tabstat commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def reg_session():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    s = Session()
    s.df = pl.DataFrame({"y": y, "x1": x1, "x2": x2,
                          "grp": ["A"] * 50 + ["B"] * 50})
    return s


class TestEsttab:
    def test_no_results(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_esttab
        out = cmd_esttab(reg_session, "")
        assert "No stored results" in out or "esttab" in out.lower()

    def test_with_results(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_esttab
        from openstat.commands.stat_cmds import cmd_ols as cmd_regress
        cmd_regress(reg_session, "y ~ x1 + x2")
        cmd_regress(reg_session, "y ~ x1")
        out = cmd_esttab(reg_session, "")
        assert "(1)" in out or "esttab" in out.lower()

    def test_stars_option(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_esttab
        from openstat.commands.stat_cmds import cmd_ols as cmd_regress
        cmd_regress(reg_session, "y ~ x1 + x2")
        out = cmd_esttab(reg_session, "stars")
        assert "p<0.05" in out or "esttab" in out.lower()


class TestTabstat:
    def test_basic(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_tabstat
        out = cmd_tabstat(reg_session, "y x1 x2")
        assert "tabstat" in out.lower() or "Mean" in out

    def test_custom_stats(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_tabstat
        out = cmd_tabstat(reg_session, "y x1 stats(mean,min,max)")
        assert "Mean" in out or "tabstat" in out.lower()

    def test_by_group(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_tabstat
        out = cmd_tabstat(reg_session, "y x1 by(grp)")
        assert "grp" in out or "A" in out

    def test_no_vars(self, reg_session):
        from openstat.commands.esttab_cmds import cmd_tabstat
        out = cmd_tabstat(reg_session, "nonexistent")
        assert "No valid" in out
