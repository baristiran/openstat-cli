"""Tests for MANOVA and two-way ANOVA."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def twoway_df():
    rng = np.random.default_rng(42)
    n = 120
    f1 = (["A"] * 60 + ["B"] * 60)
    f2 = (["X", "Y", "Z"] * 40)
    y = (
        rng.normal(0, 1, n)
        + np.array([2 if g == "B" else 0 for g in f1])
        + np.array([0 if g == "X" else (1 if g == "Y" else 2) for g in f2])
    )
    return pl.DataFrame({"y": y.tolist(), "f1": f1, "f2": f2})


@pytest.fixture()
def manova_df():
    rng = np.random.default_rng(7)
    n = 90
    groups = (["G1"] * 30 + ["G2"] * 30 + ["G3"] * 30)
    y1 = rng.normal(0, 1, n) + np.array([0] * 30 + [2] * 30 + [4] * 30)
    y2 = rng.normal(0, 1, n) + np.array([0] * 30 + [1] * 30 + [2] * 30)
    return pl.DataFrame({"group": groups, "y1": y1.tolist(), "y2": y2.tolist()})


class TestTwowayAnova:
    def test_basic(self, twoway_df):
        from openstat.stats.manova import twoway_anova
        result = twoway_anova(twoway_df, "y", "f1", "f2")
        assert "table" in result
        assert result["n_obs"] == 120

    def test_returns_rows(self, twoway_df):
        from openstat.stats.manova import twoway_anova
        result = twoway_anova(twoway_df, "y", "f1", "f2")
        assert len(result["table"]) > 0

    def test_no_interaction(self, twoway_df):
        from openstat.stats.manova import twoway_anova
        result = twoway_anova(twoway_df, "y", "f1", "f2", interaction=False)
        assert result["interaction"] is False

    def test_r_squared(self, twoway_df):
        from openstat.stats.manova import twoway_anova
        result = twoway_anova(twoway_df, "y", "f1", "f2")
        assert 0 <= result["r_squared"] <= 1


class TestMANOVA:
    def test_basic(self, manova_df):
        from openstat.stats.manova import fit_manova
        result = fit_manova(manova_df, ["y1", "y2"], "group")
        assert "effects" in result
        assert result["n_groups"] == 3

    def test_effects_list(self, manova_df):
        from openstat.stats.manova import fit_manova
        result = fit_manova(manova_df, ["y1", "y2"], "group")
        assert len(result["effects"]) > 0

    def test_wilks_present(self, manova_df):
        from openstat.stats.manova import fit_manova
        result = fit_manova(manova_df, ["y1", "y2"], "group")
        tests = [e["test"] for e in result["effects"]]
        assert any("Wilks" in t for t in tests)


class TestManovaCommands:
    def test_anova2_cmd(self, twoway_df):
        from openstat.commands.manova_cmds import cmd_anova2
        s = Session()
        s.df = twoway_df
        out = cmd_anova2(s, "y f1 f2")
        assert "Two-way ANOVA" in out
        assert "R²" in out

    def test_anova2_no_args(self, twoway_df):
        from openstat.commands.manova_cmds import cmd_anova2
        s = Session()
        s.df = twoway_df
        out = cmd_anova2(s, "y f1")
        assert "Usage" in out

    def test_manova_cmd(self, manova_df):
        from openstat.commands.manova_cmds import cmd_manova
        s = Session()
        s.df = manova_df
        out = cmd_manova(s, "y1 y2 = group")
        assert "MANOVA" in out

    def test_manova_no_eq(self, manova_df):
        from openstat.commands.manova_cmds import cmd_manova
        s = Session()
        s.df = manova_df
        out = cmd_manova(s, "y1 y2 group")
        assert "=" in out or "Usage" in out
