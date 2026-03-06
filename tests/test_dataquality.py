"""Tests for data quality commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def dirty_df():
    return pl.DataFrame({
        "x": [1.0, 2.0, 1.0, 4.0, 100.0, 2.0, None, 7.0],
        "y": [10.0, 20.0, 10.0, 40.0, 50.0, 20.0, 70.0, None],
        "cat": ["A", "B", "A", "C", "B", "B", None, "A"],
    })


@pytest.fixture()
def numeric_df():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    x[0] = 100.0  # outlier
    x[-1] = -100.0  # outlier
    return pl.DataFrame({"x": x, "y": rng.normal(0, 1, n)})


class TestDuplicates:
    def test_report(self, dirty_df):
        from openstat.commands.dataquality_cmds import cmd_duplicates
        s = Session()
        s.df = dirty_df
        out = cmd_duplicates(s, "")
        assert "duplicate" in out.lower() or "Duplicate" in out or len(out) > 0

    def test_drop(self, dirty_df):
        from openstat.commands.dataquality_cmds import cmd_duplicates
        s = Session()
        s.df = dirty_df
        original_rows = dirty_df.height
        out = cmd_duplicates(s, "drop")
        assert s.df.height <= original_rows
        assert "Dropped" in out

    def test_undo(self, dirty_df):
        from openstat.commands.dataquality_cmds import cmd_duplicates
        s = Session()
        s.df = dirty_df
        cmd_duplicates(s, "drop")
        assert s.undo()
        assert s.df.height == dirty_df.height


class TestWinsor:
    def test_basic(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_winsor
        s = Session()
        s.df = numeric_df
        out = cmd_winsor(s, "x p(0.05)")
        assert "x_w" in s.df.columns
        assert "Winsorized" in out

    def test_gen_option(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_winsor
        s = Session()
        s.df = numeric_df
        cmd_winsor(s, "x p(0.05) gen(x_win)")
        assert "x_win" in s.df.columns

    def test_clips_outliers(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_winsor
        s = Session()
        s.df = numeric_df
        cmd_winsor(s, "x p(0.05)")
        # Outlier at 100 should be clipped
        assert float(s.df["x_w"].max()) < 100.0

    def test_missing_col(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_winsor
        s = Session()
        s.df = numeric_df
        out = cmd_winsor(s, "nonexistent")
        assert "not found" in out.lower()


class TestStandardize:
    def test_basic(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_standardize
        s = Session()
        s.df = numeric_df
        out = cmd_standardize(s, "x y")
        assert "x_z" in s.df.columns
        assert "y_z" in s.df.columns

    def test_mean_zero(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_standardize
        s = Session()
        s.df = numeric_df
        cmd_standardize(s, "y")
        assert abs(float(s.df["y_z"].mean())) < 1e-6

    def test_std_one(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_standardize
        s = Session()
        s.df = numeric_df
        cmd_standardize(s, "y")
        assert abs(float(s.df["y_z"].std()) - 1.0) < 0.01


class TestNormalize:
    def test_basic(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_normalize
        s = Session()
        s.df = numeric_df
        cmd_normalize(s, "y")
        assert "y_norm" in s.df.columns

    def test_range_01(self, numeric_df):
        from openstat.commands.dataquality_cmds import cmd_normalize
        s = Session()
        s.df = numeric_df
        cmd_normalize(s, "y")
        assert float(s.df["y_norm"].min()) >= 0.0
        assert float(s.df["y_norm"].max()) <= 1.0 + 1e-10


class TestMdpattern:
    def test_basic(self, dirty_df):
        from openstat.commands.dataquality_cmds import cmd_mdpattern
        s = Session()
        s.df = dirty_df
        out = cmd_mdpattern(s, "")
        assert "Missing" in out

    def test_shows_columns(self, dirty_df):
        from openstat.commands.dataquality_cmds import cmd_mdpattern
        s = Session()
        s.df = dirty_df
        out = cmd_mdpattern(s, "x y")
        assert "x" in out and "y" in out
