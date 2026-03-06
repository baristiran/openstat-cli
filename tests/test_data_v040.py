"""Tests for v0.4.0 data commands: append, egen."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.data_cmds import cmd_append, cmd_egen


@pytest.fixture
def data_session(tmp_path):
    """Session with sample data for testing append/egen."""
    np.random.seed(42)
    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "group": ["A", "A", "B", "B", "C", "C"],
        "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "z": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    })
    return s


class TestAppend:
    def test_append_basic(self, data_session, tmp_path):
        # Create a CSV file to append
        new_df = pl.DataFrame({
            "id": [7, 8],
            "group": ["D", "D"],
            "x": [70.0, 80.0],
            "y": [7.0, 8.0],
            "z": [700.0, 800.0],
        })
        csv_path = tmp_path / "new_data.csv"
        new_df.write_csv(str(csv_path))

        result = cmd_append(data_session, f'using {csv_path}')
        assert "Appended 2 rows" in result
        assert data_session.df.height == 8

    def test_append_column_mismatch_no_force(self, data_session, tmp_path):
        new_df = pl.DataFrame({
            "id": [7],
            "group": ["D"],
            "x": [70.0],
            "extra_col": [999.0],
        })
        csv_path = tmp_path / "mismatch.csv"
        new_df.write_csv(str(csv_path))

        result = cmd_append(data_session, f'using {csv_path}')
        assert "Column mismatch" in result
        assert data_session.df.height == 6  # unchanged

    def test_append_column_mismatch_force(self, data_session, tmp_path):
        new_df = pl.DataFrame({
            "id": [7],
            "group": ["D"],
            "x": [70.0],
        })
        csv_path = tmp_path / "partial.csv"
        new_df.write_csv(str(csv_path))

        result = cmd_append(data_session, f'using {csv_path}, force')
        assert "Appended 1 rows" in result
        assert data_session.df.height == 7

    def test_append_usage(self, data_session):
        result = cmd_append(data_session, "")
        assert "Usage" in result

    def test_append_file_not_found(self, data_session):
        result = cmd_append(data_session, "using /nonexistent/file.csv")
        assert "Error" in result


class TestEgen:
    def test_egen_mean(self, data_session):
        result = cmd_egen(data_session, "x_mean = mean(x)")
        assert "Created" in result
        assert "x_mean" in data_session.df.columns
        assert abs(data_session.df["x_mean"][0] - 35.0) < 0.01  # global mean

    def test_egen_mean_by_group(self, data_session):
        result = cmd_egen(data_session, "x_gmean = mean(x), by(group)")
        assert "Created" in result
        assert "x_gmean" in data_session.df.columns
        # Group A mean = (10+20)/2 = 15
        assert abs(data_session.df.filter(pl.col("group") == "A")["x_gmean"][0] - 15.0) < 0.01

    def test_egen_sum(self, data_session):
        result = cmd_egen(data_session, "x_sum = sum(x)")
        assert "Created" in result
        assert abs(data_session.df["x_sum"][0] - 210.0) < 0.01

    def test_egen_sum_by_group(self, data_session):
        result = cmd_egen(data_session, "x_gsum = sum(x), by(group)")
        assert "Created" in result
        # Group A sum = 10 + 20 = 30
        assert abs(data_session.df.filter(pl.col("group") == "A")["x_gsum"][0] - 30.0) < 0.01

    def test_egen_min_max(self, data_session):
        cmd_egen(data_session, "x_min = min(x)")
        cmd_egen(data_session, "x_max = max(x)")
        assert abs(data_session.df["x_min"][0] - 10.0) < 0.01
        assert abs(data_session.df["x_max"][0] - 60.0) < 0.01

    def test_egen_median(self, data_session):
        result = cmd_egen(data_session, "x_med = median(x)")
        assert "Created" in result

    def test_egen_count(self, data_session):
        result = cmd_egen(data_session, "x_cnt = count(x)")
        assert "Created" in result
        assert data_session.df["x_cnt"][0] == 6

    def test_egen_count_by_group(self, data_session):
        result = cmd_egen(data_session, "g_cnt = count(x), by(group)")
        assert "Created" in result
        assert data_session.df.filter(pl.col("group") == "A")["g_cnt"][0] == 2

    def test_egen_rank(self, data_session):
        result = cmd_egen(data_session, "x_rank = rank(x)")
        assert "Created" in result
        assert "x_rank" in data_session.df.columns

    def test_egen_group(self, data_session):
        result = cmd_egen(data_session, "g_id = group(group)")
        assert "Created" in result
        assert "g_id" in data_session.df.columns

    def test_egen_rowtotal(self, data_session):
        result = cmd_egen(data_session, "rtotal = rowtotal(x y z)")
        assert "Created" in result
        # Row 0: 10 + 1 + 100 = 111
        assert abs(data_session.df["rtotal"][0] - 111.0) < 0.01

    def test_egen_rowmean(self, data_session):
        result = cmd_egen(data_session, "rmean = rowmean(x y z)")
        assert "Created" in result
        # Row 0: (10 + 1 + 100) / 3 = 37.0
        assert abs(data_session.df["rmean"][0] - 37.0) < 0.01

    def test_egen_usage(self, data_session):
        result = cmd_egen(data_session, "")
        assert "Usage" in result

    def test_egen_unknown_func(self, data_session):
        result = cmd_egen(data_session, "x_bad = blah(x)")
        assert "Unknown function" in result

    def test_egen_missing_col(self, data_session):
        result = cmd_egen(data_session, "x_m = mean(nonexist)")
        assert "not found" in result

    def test_egen_missing_by_col(self, data_session):
        result = cmd_egen(data_session, "x_m = mean(x), by(nonexist)")
        assert "not found" in result
