"""Integration tests: run demo script via subprocess, test new commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from openstat.session import Session
from openstat.commands.data_cmds import (
    cmd_load, cmd_tail, cmd_count, cmd_sort, cmd_rename,
)
from openstat.commands.stat_cmds import cmd_corr
from openstat.commands.plot_cmds import cmd_plot


ROOT = Path(__file__).resolve().parent.parent


class TestNewCommands:
    """Test commands added in Round 3: tail, count, sort, rename, corr."""

    @pytest.fixture
    def session(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "age": [25, 35, 45, 30, 50],
            "income": [30000.0, 50000.0, 70000.0, 40000.0, 80000.0],
            "score": [6.0, 7.5, 9.0, 7.0, 8.5],
            "region": ["North", "South", "East", "North", "West"],
        })
        s.dataset_name = "test.csv"
        return s

    def test_tail_default(self, session):
        result = cmd_tail(session, "")
        assert "50" in result  # last row age=50

    def test_tail_n(self, session):
        result = cmd_tail(session, "2")
        assert "50" in result
        assert "30" in result

    def test_count(self, session):
        result = cmd_count(session, "")
        assert "5" in result
        assert "4" in result  # 4 columns

    def test_sort_ascending(self, session):
        cmd_sort(session, "age")
        assert session.df["age"].to_list() == [25, 30, 35, 45, 50]

    def test_sort_descending(self, session):
        cmd_sort(session, "age --desc")
        assert session.df["age"].to_list() == [50, 45, 35, 30, 25]

    def test_sort_creates_snapshot(self, session):
        cmd_sort(session, "age")
        assert session.undo_depth == 1

    def test_sort_missing_col(self, session):
        result = cmd_sort(session, "nonexistent")
        assert "not found" in result

    def test_rename(self, session):
        result = cmd_rename(session, "age years")
        assert "years" in result
        assert "years" in session.df.columns
        assert "age" not in session.df.columns

    def test_rename_creates_snapshot(self, session):
        cmd_rename(session, "age years")
        assert session.undo_depth == 1

    def test_rename_missing_col(self, session):
        result = cmd_rename(session, "nonexistent newname")
        assert "not found" in result

    def test_rename_conflict(self, session):
        result = cmd_rename(session, "age income")
        assert "already exists" in result

    def test_corr(self, session):
        result = cmd_corr(session, "age income score")
        assert "Correlation" in result
        assert "age" in result
        assert "income" in result

    def test_corr_too_few_cols(self, session):
        result = cmd_corr(session, "age")
        assert "at least 2" in result


class TestNewPlots:
    """Test plot types added in Round 3: line, box."""

    @pytest.fixture
    def session(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "group": ["A", "A", "B", "B", "B"],
        })
        return s

    def test_line_plot(self, session):
        result = cmd_plot(session, "line y x")
        assert "saved" in result.lower()
        assert len(session.plot_paths) == 1

    def test_box_plot(self, session):
        result = cmd_plot(session, "box y")
        assert "saved" in result.lower()

    def test_box_plot_grouped(self, session):
        result = cmd_plot(session, "box y by group")
        assert "saved" in result.lower()

    def test_unknown_plot_type(self, session):
        result = cmd_plot(session, "pie y")
        assert "Unknown" in result


class TestScriptExecution:
    """Run the demo.ost script via subprocess."""

    def test_demo_script_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "openstat", "run", str(ROOT / "examples" / "demo.ost")],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT),
        )
        # Script should complete without crashing
        assert result.returncode == 0, f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        # Verify key outputs appeared
        assert "Loaded" in result.stdout
        assert "age" in result.stdout

    def test_script_produces_outputs(self, tmp_path):
        """Run script and verify output files are created."""
        # Copy example data to tmp
        import shutil
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        shutil.copy(ROOT / "examples" / "data.csv", examples_dir / "data.csv")

        # Create a minimal script
        script = tmp_path / "test.ost"
        script.write_text(
            "load examples/data.csv\n"
            "describe\n"
            "summarize age income\n"
            "save outputs/test_out.csv\n"
        )

        result = subprocess.run(
            [sys.executable, "-m", "openstat", "run", str(script)],
            capture_output=True, text=True, timeout=60,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (tmp_path / "outputs" / "test_out.csv").exists()
