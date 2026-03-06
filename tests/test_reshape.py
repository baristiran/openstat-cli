"""Tests for reshape, collapse, encode, decode commands."""

from __future__ import annotations

import polars as pl
import pytest

from openstat.session import Session


@pytest.fixture()
def wide_df():
    return pl.DataFrame({
        "id": [1, 2, 3],
        "score2020": [80.0, 90.0, 85.0],
        "score2021": [82.0, 88.0, 91.0],
        "score2022": [78.0, 95.0, 87.0],
    })


@pytest.fixture()
def long_df():
    return pl.DataFrame({
        "id":   [1, 1, 2, 2, 3, 3],
        "year": ["2020", "2021", "2020", "2021", "2020", "2021"],
        "score": [80.0, 82.0, 90.0, 88.0, 85.0, 91.0],
    })


@pytest.fixture()
def group_df():
    return pl.DataFrame({
        "region": ["North", "North", "South", "South", "East"],
        "income": [50000.0, 60000.0, 45000.0, 55000.0, 70000.0],
        "age":    [30, 35, 28, 40, 45],
    })


@pytest.fixture()
def str_df():
    return pl.DataFrame({
        "name":  ["Alice", "Bob", "Carol", "Alice", "Bob"],
        "score": [85.0, 90.0, 78.0, 88.0, 92.0],
    })


class TestReshapeLong:
    def test_wide_to_long(self, wide_df):
        from openstat.commands.reshape_cmds import cmd_reshape
        s = Session()
        s.df = wide_df
        out = cmd_reshape(s, "long score i(id) j(year)")
        assert "wide→long" in out or "Reshaped" in out
        assert s.df is not None
        assert s.df.shape[0] > wide_df.shape[0]

    def test_undo_after_reshape(self, wide_df):
        from openstat.commands.reshape_cmds import cmd_reshape
        s = Session()
        s.df = wide_df
        original_shape = wide_df.shape
        cmd_reshape(s, "long score i(id) j(year)")
        assert s.undo()
        assert s.df.shape == original_shape


class TestReshapeWide:
    def test_long_to_wide(self, long_df):
        from openstat.commands.reshape_cmds import cmd_reshape
        s = Session()
        s.df = long_df
        out = cmd_reshape(s, "wide score i(id) j(year)")
        assert "Reshaped" in out or "wide" in out.lower()
        assert s.df is not None
        assert s.df.shape[0] == 3  # 3 unique ids


class TestCollapse:
    def test_mean_by_group(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        out = cmd_collapse(s, "(mean) income by(region)")
        assert "Collapsed" in out
        assert s.df.shape[0] == 3  # 3 regions

    def test_sum(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        old_sum = group_df["income"].sum()
        out = cmd_collapse(s, "(sum) income")
        assert s.df["income"][0] == pytest.approx(old_sum)

    def test_count(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        cmd_collapse(s, "(count) income by(region)")
        assert s.df.shape[0] == 3

    def test_unknown_stat(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        out = cmd_collapse(s, "(geometric) income by(region)")
        assert "Unknown" in out

    def test_undo(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        original_shape = group_df.shape
        cmd_collapse(s, "(mean) income by(region)")
        assert s.undo()
        assert s.df.shape == original_shape

    def test_no_by(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        cmd_collapse(s, "(mean) income")
        assert s.df.shape[0] == 1

    def test_multiple_vars(self, group_df):
        from openstat.commands.reshape_cmds import cmd_collapse
        s = Session()
        s.df = group_df
        out = cmd_collapse(s, "(mean) income age by(region)")
        assert "income" in out or "Collapsed" in out


class TestEncode:
    def test_basic(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode
        s = Session()
        s.df = str_df
        out = cmd_encode(s, "name")
        assert "Encoded" in out or "categories" in out
        assert "name_encoded" in s.df.columns

    def test_gen_option(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode
        s = Session()
        s.df = str_df
        cmd_encode(s, "name gen(name_id)")
        assert "name_id" in s.df.columns

    def test_integer_output(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode
        s = Session()
        s.df = str_df
        cmd_encode(s, "name")
        assert s.df["name_encoded"].dtype in (pl.Int64, pl.Int32)

    def test_missing_col(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode
        s = Session()
        s.df = str_df
        out = cmd_encode(s, "nonexistent")
        assert "not found" in out.lower()

    def test_roundtrip_count(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode
        s = Session()
        s.df = str_df
        cmd_encode(s, "name gen(name_code)")
        unique_codes = s.df["name_code"].unique().len()
        unique_names = str_df["name"].unique().len()
        assert unique_codes == unique_names


class TestDecode:
    def test_basic(self, str_df):
        from openstat.commands.reshape_cmds import cmd_encode, cmd_decode
        s = Session()
        s.df = str_df
        cmd_encode(s, "name gen(name_code)")
        out = cmd_decode(s, "name_code name gen(name_back)")
        assert "Decoded" in out
        assert "name_back" in s.df.columns

    def test_missing_col(self, str_df):
        from openstat.commands.reshape_cmds import cmd_decode
        s = Session()
        s.df = str_df
        out = cmd_decode(s, "missing name gen(x)")
        assert "not found" in out.lower()

    def test_no_args(self, str_df):
        from openstat.commands.reshape_cmds import cmd_decode
        s = Session()
        s.df = str_df
        out = cmd_decode(s, "only_one")
        assert "Usage" in out
