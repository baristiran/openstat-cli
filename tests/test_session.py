"""Tests for Session and I/O."""

import polars as pl
import pytest
from pathlib import Path

from openstat.session import Session
from openstat.config import get_config
from openstat.io.loader import load_file, save_file


class TestSession:
    def test_create_session(self):
        s = Session()
        assert s.df is None
        assert s.history == []

    def test_record(self):
        s = Session()
        s.record("load data.csv")
        assert len(s.history) == 1

    def test_require_data_raises(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No dataset"):
            s.require_data()

    def test_require_data_ok(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        assert s.require_data().height == 3

    def test_shape_str(self):
        s = Session()
        assert s.shape_str == "No data"
        s.df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        assert "2 rows" in s.shape_str


class TestUndo:
    def test_snapshot_and_undo(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        s.snapshot()
        s.df = s.df.filter(pl.col("x") > 1)
        assert s.df.height == 2
        assert s.undo()
        assert s.df.height == 3

    def test_undo_empty_returns_false(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert not s.undo()

    def test_undo_stack_bounded(self):
        max_undo = get_config().max_undo_stack
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        for i in range(max_undo + 10):
            s.snapshot()
        assert s.undo_depth == max_undo

    def test_snapshot_clones_data(self):
        """Snapshot should be independent copy, not reference."""
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        s.snapshot()
        s.df = pl.DataFrame({"x": [99]})  # replace entirely
        s.undo()
        assert s.df["x"].to_list() == [1, 2, 3]

    def test_multiple_undo(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        s.snapshot()
        s.df = s.df.head(4)  # 4 rows
        s.snapshot()
        s.df = s.df.head(3)  # 3 rows
        s.snapshot()
        s.df = s.df.head(2)  # 2 rows

        assert s.df.height == 2
        s.undo()
        assert s.df.height == 3
        s.undo()
        assert s.df.height == 4
        s.undo()
        assert s.df.height == 5


class TestIO:
    def test_load_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}).write_csv(path)
        df = load_file(path)
        assert df.shape == (2, 2)

    def test_load_parquet(self, tmp_path):
        path = tmp_path / "test.parquet"
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}).write_parquet(path)
        df = load_file(path)
        assert df.shape == (2, 2)

    def test_load_missing(self):
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent.csv")

    def test_load_unsupported(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(path)

    def test_save_csv(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        path = save_file(df, tmp_path / "out.csv")
        assert path.exists()
        loaded = pl.read_csv(path)
        assert loaded.shape == (3, 1)

    def test_save_parquet(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        path = save_file(df, tmp_path / "out.parquet")
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert loaded.shape == (3, 1)
