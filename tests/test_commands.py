"""Tests for command handlers."""

import polars as pl
import pytest
from pathlib import Path

from openstat.session import Session
from openstat.commands.data_cmds import (
    cmd_load, cmd_describe, cmd_head, cmd_filter, cmd_select,
    cmd_derive, cmd_dropna, cmd_undo, cmd_save,
)
from openstat.commands.stat_cmds import (
    cmd_summarize, cmd_tabulate, cmd_ols, cmd_logit,
)
from openstat.commands.groupby_cmds import cmd_groupby
from openstat.commands.report_cmds import cmd_report


@pytest.fixture
def session(tmp_path):
    s = Session(output_dir=tmp_path / "outputs")
    s.df = pl.DataFrame({
        "age": [25, 35, 45, 30, 50],
        "income": [30000.0, 50000.0, 70000.0, 40000.0, 80000.0],
        "region": ["North", "South", "East", "North", "West"],
        "score": [6.0, 7.5, 9.0, 7.0, 8.5],
        "employed": [1, 1, 1, 0, 1],
    })
    s.dataset_name = "test.csv"
    return s


@pytest.fixture
def csv_path(tmp_path):
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = tmp_path / "test.csv"
    df.write_csv(path)
    return path


class TestLoad:
    def test_load_csv(self, csv_path):
        s = Session()
        result = cmd_load(s, str(csv_path))
        assert "3 rows" in result
        assert s.df is not None

    def test_load_missing_file(self):
        s = Session()
        with pytest.raises(FileNotFoundError):
            cmd_load(s, "/nonexistent/file.csv")

    def test_load_no_args(self):
        s = Session()
        assert "Usage" in cmd_load(s, "")

    def test_load_clears_undo_stack(self, csv_path):
        s = Session()
        s._undo_stack.append(pl.DataFrame({"old": [1]}))
        cmd_load(s, str(csv_path))
        assert len(s._undo_stack) == 0


class TestDescribe:
    def test_describe(self, session):
        result = cmd_describe(session, "")
        assert "age" in result
        assert "income" in result

    def test_describe_no_data(self):
        s = Session()
        with pytest.raises(RuntimeError):
            cmd_describe(s, "")


class TestSummarize:
    def test_summarize_all(self, session):
        result = cmd_summarize(session, "")
        assert "Mean" in result
        assert "age" in result

    def test_summarize_specific(self, session):
        result = cmd_summarize(session, "age income")
        assert "age" in result
        assert "income" in result

    def test_summarize_missing_col(self, session):
        result = cmd_summarize(session, "nonexistent")
        assert "not found" in result

    def test_summarize_labels_sd_as_sample(self, session):
        result = cmd_summarize(session, "age")
        assert "sample" in result.lower()


class TestTabulate:
    def test_tabulate(self, session):
        result = cmd_tabulate(session, "region")
        assert "North" in result
        assert "Count" in result

    def test_tabulate_no_col(self, session):
        assert "Usage" in cmd_tabulate(session, "")

    def test_tabulate_truncates_large_cardinality(self, tmp_path):
        """Tabulate should truncate at 50 unique values."""
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "id": [f"item_{i}" for i in range(200)]
        })
        result = cmd_tabulate(s, "id")
        assert "top 50" in result.lower() or "200" in result


class TestFilter:
    def test_filter_comparison(self, session):
        result = cmd_filter(session, "age > 30")
        assert "dropped" in result
        assert session.df.height < 5

    def test_filter_no_expr(self, session):
        assert "Usage" in cmd_filter(session, "")

    def test_filter_creates_snapshot(self, session):
        assert session.undo_depth == 0
        cmd_filter(session, "age > 30")
        assert session.undo_depth == 1


class TestUndo:
    def test_undo_after_filter(self, session):
        original_height = session.df.height
        cmd_filter(session, "age > 30")
        assert session.df.height < original_height
        result = cmd_undo(session, "")
        assert "restored" in result.lower() or "Undone" in result
        assert session.df.height == original_height

    def test_undo_after_select(self, session):
        original_cols = session.df.columns
        cmd_select(session, "age income")
        assert len(session.df.columns) == 2
        cmd_undo(session, "")
        assert session.df.columns == original_cols

    def test_undo_after_derive(self, session):
        original_cols = session.df.columns
        cmd_derive(session, "x = age * 2")
        assert "x" in session.df.columns
        cmd_undo(session, "")
        assert session.df.columns == original_cols

    def test_undo_empty_stack(self, session):
        result = cmd_undo(session, "")
        assert "Nothing" in result

    def test_multiple_undos(self, session):
        h0 = session.df.height
        cmd_filter(session, "age > 25")
        h1 = session.df.height
        cmd_filter(session, "age > 35")
        h2 = session.df.height
        assert h0 > h1 > h2
        cmd_undo(session, "")
        assert session.df.height == h1
        cmd_undo(session, "")
        assert session.df.height == h0


class TestSelect:
    def test_select_columns(self, session):
        cmd_select(session, "age income")
        assert session.df.columns == ["age", "income"]

    def test_select_missing(self, session):
        result = cmd_select(session, "nonexistent")
        assert "not found" in result


class TestDerive:
    def test_derive_new_column(self, session):
        result = cmd_derive(session, "income_k = income / 1000")
        assert "income_k" in result
        assert "income_k" in session.df.columns

    def test_derive_no_equals(self, session):
        assert "Usage" in cmd_derive(session, "foo bar")

    def test_derive_with_function(self, session):
        result = cmd_derive(session, "log_income = log(income)")
        assert "log_income" in session.df.columns
        assert session.df["log_income"][0] > 0


class TestDropna:
    def test_dropna(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, None, 3], "y": [4, 5, None]})
        result = cmd_dropna(s, "")
        assert s.df.height == 1
        assert "removed" in result

    def test_dropna_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, None, 3]})
        cmd_dropna(s, "")
        assert s.undo_depth == 1


class TestModels:
    def test_ols(self, session):
        result = cmd_ols(session, "score ~ age + income")
        assert "OLS" in result
        assert "Coef" in result

    def test_ols_robust(self, session):
        result = cmd_ols(session, "score ~ age + income --robust")
        assert "robust" in result.lower() or "Coef" in result

    def test_logit(self, session):
        result = cmd_logit(session, "employed ~ age + income")
        assert "Logit" in result or "Coef" in result

    def test_ols_no_formula(self, session):
        assert "Usage" in cmd_ols(session, "")


class TestPlot:
    def test_hist(self, session):
        from openstat.commands.plot_cmds import cmd_plot
        result = cmd_plot(session, "hist age")
        assert "saved" in result.lower() or "Histogram" in result
        assert len(session.plot_paths) == 1

    def test_scatter(self, session):
        from openstat.commands.plot_cmds import cmd_plot
        result = cmd_plot(session, "scatter score income")
        assert "saved" in result.lower() or "Scatter" in result


class TestGroupby:
    def test_groupby_summarize(self, session):
        result = cmd_groupby(session, "region agg income:mean")
        assert "Group-by" in result or "rows" in result or "North" in result or "income" in result


class TestSaveAndReport:
    def test_save_csv(self, session, tmp_path):
        path = tmp_path / "out.csv"
        result = cmd_save(session, str(path))
        assert path.exists()

    def test_report(self, session, tmp_path):
        session.history = ["load data.csv", "summarize"]
        path = tmp_path / "report.md"
        result = cmd_report(session, str(path))
        assert path.exists()
        content = path.read_text()
        assert "OpenStat" in content


class TestHead:
    def test_head_default(self, session):
        result = cmd_head(session, "")
        assert "25" in result

    def test_head_n(self, session):
        result = cmd_head(session, "2")
        assert "25" in result
