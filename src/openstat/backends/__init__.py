"""Data backend abstraction: Polars (default) and DuckDB."""

from __future__ import annotations

from typing import Protocol

import polars as pl


class DataBackend(Protocol):
    """Protocol for data backends."""

    def load(self, path: str) -> None: ...
    def to_polars(self) -> pl.DataFrame: ...
    def sql(self, query: str) -> pl.DataFrame: ...
    def shape(self) -> tuple[int, int]: ...
