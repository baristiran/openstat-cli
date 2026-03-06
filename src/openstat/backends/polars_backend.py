"""Polars backend — wraps the default Polars DataFrame operations."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from openstat.config import get_config


class PolarsBackend:
    """Default backend using Polars DataFrames."""

    def __init__(self) -> None:
        self._df: pl.DataFrame | None = None
        self._lazy: pl.LazyFrame | None = None

    def load(self, path: str) -> None:
        p = Path(path)
        suffix = p.suffix.lower()
        cfg = get_config()

        # Use LazyFrame for scan-capable formats
        if suffix == ".csv":
            self._lazy = pl.scan_csv(
                p,
                infer_schema_length=cfg.infer_schema_length,
            )
        elif suffix == ".parquet":
            self._lazy = pl.scan_parquet(p)
        else:
            # For other formats, load eagerly
            from openstat.io.loader import load_file
            self._df = load_file(path)
            self._lazy = self._df.lazy()

    def to_polars(self) -> pl.DataFrame:
        if self._df is not None:
            return self._df
        if self._lazy is not None:
            self._df = self._lazy.collect()
            return self._df
        raise RuntimeError("No data loaded")

    def sql(self, query: str) -> pl.DataFrame:
        df = self.to_polars()
        return pl.SQLContext({"data": df}).execute(query).collect()

    def shape(self) -> tuple[int, int]:
        df = self.to_polars()
        return df.shape
