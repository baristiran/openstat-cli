"""Tests for Round 5 features: bar/heatmap plots, duplicates, unique, encode, recode, crosstab."""

from __future__ import annotations

import polars as pl
import pytest

from openstat.session import Session
from openstat.commands.data_cmds import (
    cmd_duplicates, cmd_unique, cmd_encode, cmd_recode,
)
from openstat.commands.stat_cmds import cmd_crosstab
from openstat.commands.plot_cmds import cmd_plot


# -----------------------------------------------------------------------
# Bar Chart / Heatmap
# -----------------------------------------------------------------------

class TestBarPlot:
    @pytest.fixture
    def session(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "region": ["North", "South", "East", "North", "West"],
            "income": [30000.0, 50000.0, 70000.0, 40000.0, 80000.0],
        })
        return s

    def test_bar_categorical(self, session):
        result = cmd_plot(session, "bar region")
        assert "saved" in result.lower()

    def test_bar_grouped(self, session):
        result = cmd_plot(session, "bar income by region")
        assert "saved" in result.lower()

    def test_bar_adds_plot_path(self, session):
        cmd_plot(session, "bar region")
        assert len(session.plot_paths) == 1


class TestHeatmapPlot:
    @pytest.fixture
    def session(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "z": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        return s

    def test_heatmap_all_cols(self, session):
        result = cmd_plot(session, "heatmap")
        assert "saved" in result.lower()

    def test_heatmap_specific_cols(self, session):
        result = cmd_plot(session, "heatmap x y")
        assert "saved" in result.lower()

    def test_heatmap_adds_plot_path(self, session):
        cmd_plot(session, "heatmap")
        assert len(session.plot_paths) == 1


# -----------------------------------------------------------------------
# Duplicates
# -----------------------------------------------------------------------

class TestDuplicates:
    def test_find_duplicates(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 2, 3], "y": ["a", "b", "b", "c"]})
        result = cmd_duplicates(s, "")
        assert "1" in result  # 1 duplicate

    def test_no_duplicates(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        result = cmd_duplicates(s, "")
        assert "No duplicates" in result

    def test_drop_duplicates(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 2, 3], "y": ["a", "b", "b", "c"]})
        result = cmd_duplicates(s, "drop")
        assert s.df.height == 3  # one duplicate removed
        assert "Dropped" in result

    def test_duplicates_by_column(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 1, 2], "y": ["a", "b", "c"]})
        result = cmd_duplicates(s, "x")
        assert "1" in result  # 1 duplicate on col x

    def test_drop_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 1, 2]})
        cmd_duplicates(s, "drop")
        assert s.undo_depth == 1


# -----------------------------------------------------------------------
# Unique
# -----------------------------------------------------------------------

class TestUnique:
    def test_unique_values(self):
        s = Session()
        s.df = pl.DataFrame({"color": ["red", "blue", "red", "green"]})
        result = cmd_unique(s, "color")
        assert "3" in result
        assert "red" in result
        assert "blue" in result

    def test_unique_missing_col(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_unique(s, "nope")
        assert "not found" in result

    def test_unique_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_unique(s, "")


# -----------------------------------------------------------------------
# Encode
# -----------------------------------------------------------------------

class TestEncode:
    def test_encode_creates_code_column(self):
        s = Session()
        s.df = pl.DataFrame({"color": ["red", "blue", "green", "red"]})
        result = cmd_encode(s, "color")
        assert "color_code" in s.df.columns
        # blue=0, green=1, red=2 (alphabetical)
        assert s.df["color_code"].to_list() == [2, 0, 1, 2]

    def test_encode_custom_name(self):
        s = Session()
        s.df = pl.DataFrame({"size": ["S", "M", "L"]})
        result = cmd_encode(s, "size as size_num")
        assert "size_num" in s.df.columns

    def test_encode_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": ["a", "b"]})
        cmd_encode(s, "x")
        assert s.undo_depth == 1

    def test_encode_missing_col(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_encode(s, "nope")
        assert "not found" in result


# -----------------------------------------------------------------------
# Recode
# -----------------------------------------------------------------------

class TestRecode:
    def test_recode_values(self):
        s = Session()
        s.df = pl.DataFrame({"region": ["North", "South", "East"]})
        result = cmd_recode(s, 'region North=N South=S East=E')
        assert s.df["region"].to_list() == ["N", "S", "E"]

    def test_recode_partial(self):
        s = Session()
        s.df = pl.DataFrame({"x": ["a", "b", "c"]})
        result = cmd_recode(s, "x a=A")
        assert s.df["x"].to_list() == ["A", "b", "c"]

    def test_recode_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": ["a"]})
        cmd_recode(s, "x a=b")
        assert s.undo_depth == 1

    def test_recode_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_recode(s, "x noequals")


# -----------------------------------------------------------------------
# Crosstab
# -----------------------------------------------------------------------

class TestCrosstab:
    def test_crosstab_output(self):
        s = Session()
        s.df = pl.DataFrame({
            "gender": ["M", "F", "M", "F", "M"],
            "status": ["A", "A", "B", "B", "A"],
        })
        result = cmd_crosstab(s, "gender status")
        assert "Cross-tabulation" in result
        assert "M" in result
        assert "F" in result
        assert "Total" in result

    def test_crosstab_percentages(self):
        s = Session()
        s.df = pl.DataFrame({
            "x": ["A", "A", "B", "B"],
            "y": ["Y", "N", "Y", "N"],
        })
        result = cmd_crosstab(s, "x y")
        assert "50%" in result  # each cell is 50% of row

    def test_crosstab_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_crosstab(s, "single")

    def test_crosstab_missing_col(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_crosstab(s, "x nope")
        assert "not found" in result
