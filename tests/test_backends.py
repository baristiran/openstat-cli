"""Tests for DuckDB/LazyFrame backend (F6)."""

import pytest
import polars as pl

from openstat.session import Session
from openstat.commands.backend_cmds import cmd_set, cmd_sql

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


@pytest.fixture
def backend_session(tmp_path):
    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "name": ["a", "b", "c", "d", "e"],
    })
    return s


class TestSetBackend:
    def test_set_polars(self, backend_session):
        result = cmd_set(backend_session, "backend polars")
        assert "polars" in result
        assert backend_session._backend == "polars"

    @pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
    def test_set_duckdb(self, backend_session):
        result = cmd_set(backend_session, "backend duckdb")
        assert "duckdb" in result
        assert backend_session._backend == "duckdb"

    def test_set_unknown(self, backend_session):
        result = cmd_set(backend_session, "backend sqlite")
        assert "Unknown" in result

    def test_set_usage(self, backend_session):
        result = cmd_set(backend_session, "")
        assert "Usage" in result


class TestSQLCommand:
    def test_sql_polars_context(self, backend_session):
        result = cmd_sql(backend_session, '"SELECT * FROM data WHERE x > 2"')
        assert "3 rows" in result or "rows" in result

    def test_sql_no_data(self, tmp_path):
        s = Session(output_dir=tmp_path / "out")
        result = cmd_sql(s, '"SELECT 1"')
        assert "No data" in result

    def test_sql_usage(self, backend_session):
        result = cmd_sql(backend_session, "")
        assert "Usage" in result

    @pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
    def test_sql_duckdb(self, backend_session):
        cmd_set(backend_session, "backend duckdb")
        result = cmd_sql(backend_session, '"SELECT * FROM data WHERE x > 3"')
        assert "rows" in result
