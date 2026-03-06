"""DuckDB backend for large dataset processing."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _try_import_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError(
            "DuckDB backend requires duckdb. "
            "Install it with: pip install openstat[duckdb]"
        )


class DuckDBBackend:
    """DuckDB-based backend for large datasets."""

    def __init__(self) -> None:
        duckdb = _try_import_duckdb()
        self._conn = duckdb.connect()
        self._table_loaded = False

    def load(self, path: str) -> None:
        p = Path(path)
        suffix = p.suffix.lower()
        path_str = str(p).replace("'", "''")

        if suffix == ".csv":
            self._conn.execute(f"CREATE OR REPLACE TABLE data AS SELECT * FROM read_csv('{path_str}')")
        elif suffix == ".parquet":
            self._conn.execute(f"CREATE OR REPLACE TABLE data AS SELECT * FROM read_parquet('{path_str}')")
        elif suffix in (".xlsx", ".xls"):
            # DuckDB doesn't natively support Excel; load via Polars and register
            df = pl.read_excel(p)
            self._conn.register("data", df.to_pandas())
        else:
            from openstat.io.loader import load_file
            df = load_file(path)
            self._conn.register("data", df.to_pandas())

        self._table_loaded = True

    def to_polars(self) -> pl.DataFrame:
        if not self._table_loaded:
            raise RuntimeError("No data loaded")
        return self._conn.execute("SELECT * FROM data").pl()

    def sql(self, query: str) -> pl.DataFrame:
        if not self._table_loaded:
            raise RuntimeError("No data loaded. Use 'load' first.")
        return self._conn.execute(query).pl()

    def shape(self) -> tuple[int, int]:
        if not self._table_loaded:
            return (0, 0)
        result = self._conn.execute("SELECT COUNT(*) as n FROM data").fetchone()
        n_rows = result[0] if result else 0
        cols = self._conn.execute("SELECT * FROM data LIMIT 0").description
        n_cols = len(cols) if cols else 0
        return (n_rows, n_cols)

    def execute(self, query: str):
        """Execute raw SQL and return DuckDB result."""
        return self._conn.execute(query)
