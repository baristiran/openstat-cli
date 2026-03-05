"""Shared type constants."""

from __future__ import annotations

import polars as pl

NUMERIC_DTYPES: set[type[pl.DataType]] = {
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
