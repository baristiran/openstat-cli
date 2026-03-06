"""Tests for string and date manipulation commands."""

from __future__ import annotations

import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def str_df():
    return pl.DataFrame({
        "name": ["  Alice  ", "  Bob ", " Carol"],
        "city": ["New York", "Los Angeles", "Chicago"],
        "val": [1.0, 2.0, 3.0],
    })


@pytest.fixture()
def ts_df():
    return pl.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


class TestSplit:
    def test_basic(self, str_df):
        from openstat.commands.string_cmds import cmd_split
        s = Session()
        s.df = str_df
        out = cmd_split(s, "city sep( )")
        assert "city1" in s.df.columns

    def test_gen_prefix(self, str_df):
        from openstat.commands.string_cmds import cmd_split
        s = Session()
        s.df = str_df
        cmd_split(s, "city sep( ) gen(part)")
        assert "part1" in s.df.columns

    def test_missing_col(self, str_df):
        from openstat.commands.string_cmds import cmd_split
        s = Session()
        s.df = str_df
        out = cmd_split(s, "nonexistent")
        assert "not found" in out.lower()


class TestStrTrim:
    def test_trims(self, str_df):
        from openstat.commands.string_cmds import cmd_strtrim
        s = Session()
        s.df = str_df
        cmd_strtrim(s, "name gen(name_trimmed)")
        assert s.df["name_trimmed"][0] == "Alice"

    def test_inplace(self, str_df):
        from openstat.commands.string_cmds import cmd_strtrim
        s = Session()
        s.df = str_df
        cmd_strtrim(s, "name")
        assert "Alice" in s.df["name"].to_list()


class TestStrCase:
    def test_upper(self, str_df):
        from openstat.commands.string_cmds import cmd_strupper
        s = Session()
        s.df = str_df
        cmd_strupper(s, "city gen(city_up)")
        assert s.df["city_up"][0] == "NEW YORK"

    def test_lower(self, str_df):
        from openstat.commands.string_cmds import cmd_strlower
        s = Session()
        s.df = str_df
        cmd_strlower(s, "city gen(city_lo)")
        assert s.df["city_lo"][0] == "new york"


class TestStrReplace:
    def test_basic(self, str_df):
        from openstat.commands.string_cmds import cmd_strreplace
        s = Session()
        s.df = str_df
        cmd_strreplace(s, "city New Old gen(city2)")
        assert "Old York" in s.df["city2"].to_list()


class TestLagLead:
    def test_lag(self, ts_df):
        from openstat.commands.data_cmds import cmd_lag
        s = Session()
        s.df = ts_df
        cmd_lag(s, "y 1 as y_lag1")
        assert "y_lag1" in s.df.columns

    def test_lead(self, ts_df):
        from openstat.commands.data_cmds import cmd_lead
        s = Session()
        s.df = ts_df
        cmd_lead(s, "y 1 as y_lead1")
        assert "y_lead1" in s.df.columns
