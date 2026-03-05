"""Data loading and saving: CSV, Parquet, Stata (.dta), Excel (.xlsx)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from openstat.config import get_config
from openstat.logging_config import get_logger

log = get_logger("io")


def _load_dta(path: Path) -> pl.DataFrame:
    """Load a Stata .dta file using pyreadstat (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .dta files requires pyreadstat. "
            "Install it with: pip install openstat[stata]"
        )
    pandas_df, meta = pyreadstat.read_dta(str(path))
    return pl.from_pandas(pandas_df)


def _load_excel(path: Path) -> pl.DataFrame:
    """Load an Excel file (optional dependency)."""
    try:
        return pl.read_excel(path)
    except ImportError:
        raise ImportError(
            "Reading .xlsx files requires openpyxl. "
            "Install it with: pip install openstat[excel]"
        )


def _load_csv(path: Path) -> pl.DataFrame:
    cfg = get_config()
    return pl.read_csv(
        path,
        infer_schema_length=cfg.infer_schema_length,
        separator=cfg.csv_separator,
    )


_LOADERS = {
    ".csv": _load_csv,
    ".parquet": lambda p: pl.read_parquet(p),
    ".dta": _load_dta,
    ".xlsx": _load_excel,
    ".xls": _load_excel,
}

_SUPPORTED = ", ".join(_LOADERS.keys())


def load_file(path: str | Path) -> pl.DataFrame:
    """Load a data file into a Polars DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix}  (supported: {_SUPPORTED})")
    log.info("Loading %s (%s, %.1f KB)", p.name, suffix, p.stat().st_size / 1024)
    df = loader(p)
    log.info("Loaded %d rows x %d cols", df.height, df.width)
    return df


def _save_excel(df: pl.DataFrame, path: Path) -> None:
    """Save to Excel (optional dependency)."""
    try:
        df.write_excel(path)
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Writing .xlsx files requires xlsxwriter. "
            "Install it with: pip install openstat[excel]"
        )


def _save_dta(df: pl.DataFrame, path: Path) -> None:
    """Save to Stata .dta (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Writing .dta files requires pyreadstat. "
            "Install it with: pip install openstat[stata]"
        )
    pandas_df = df.to_pandas()
    pyreadstat.write_dta(pandas_df, str(path))


def save_file(df: pl.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to CSV, Parquet, Excel, or Stata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.write_csv(p, separator=get_config().csv_separator)
    elif suffix == ".parquet":
        df.write_parquet(p)
    elif suffix == ".xlsx":
        _save_excel(df, p)
    elif suffix == ".dta":
        _save_dta(df, p)
    else:
        raise ValueError(f"Unsupported save format: {suffix}  (use .csv, .parquet, .xlsx, or .dta)")
    return p
