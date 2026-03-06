"""Data loading and saving: CSV, Parquet, Stata (.dta), Excel (.xlsx)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from openstat.config import get_config
from openstat.logging_config import get_logger

log = get_logger("io")


def _load_dta(path: Path, session=None) -> pl.DataFrame:
    """Load a Stata .dta file using pyreadstat (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .dta files requires pyreadstat. "
            "Install it with: pip install openstat[stata]"
        )
    pandas_df, meta = pyreadstat.read_dta(str(path))
    if session is not None and meta.variable_value_labels:
        session._variable_labels = meta.variable_value_labels
    return pl.from_pandas(pandas_df)


def _load_sas7bdat(path: Path, session=None) -> pl.DataFrame:
    """Load a SAS .sas7bdat file using pyreadstat (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .sas7bdat files requires pyreadstat. "
            "Install it with: pip install openstat[sas]"
        )
    pandas_df, meta = pyreadstat.read_sas7bdat(str(path))
    if session is not None and meta.variable_value_labels:
        session._variable_labels = meta.variable_value_labels
    return pl.from_pandas(pandas_df)


def _load_sav(path: Path, session=None) -> pl.DataFrame:
    """Load an SPSS .sav file using pyreadstat (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .sav files requires pyreadstat. "
            "Install it with: pip install openstat[spss]"
        )
    pandas_df, meta = pyreadstat.read_sav(str(path))
    if session is not None and meta.variable_value_labels:
        session._variable_labels = meta.variable_value_labels
    return pl.from_pandas(pandas_df)


def _load_excel(path: Path, sheet: str | int | None = None) -> pl.DataFrame:
    """Load an Excel file (optional dependency).

    Args:
        sheet: Sheet name or 0-based index. If None, loads the first sheet.
               Pass 'list' to list available sheet names.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "Reading .xlsx files requires openpyxl. "
            "Install it with: pip install openstat[excel]"
        )

    if sheet == "list":
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        raise ValueError(f"Sheets in {path.name}: {', '.join(names)}")

    kwargs: dict = {}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    try:
        return pl.read_excel(path, **kwargs)
    except TypeError:
        # Older polars may not support sheet_name
        return pl.read_excel(path)


def _load_csv(path: Path, session=None) -> pl.DataFrame:
    cfg = get_config()
    return pl.read_csv(
        path,
        infer_schema_length=cfg.infer_schema_length,
        separator=cfg.csv_separator,
    )


_LOADERS = {
    ".csv": _load_csv,
    ".parquet": lambda p, session=None: pl.read_parquet(p),
    ".dta": _load_dta,
    ".xlsx": lambda p, session=None: _load_excel(p),
    ".xls": lambda p, session=None: _load_excel(p),
    ".sas7bdat": _load_sas7bdat,
    ".sav": _load_sav,
}

_SUPPORTED = ", ".join(_LOADERS.keys())


def load_file(path: str | Path, session=None) -> pl.DataFrame:
    """Load a data file into a Polars DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix}  (supported: {_SUPPORTED})")
    log.info("Loading %s (%s, %.1f KB)", p.name, suffix, p.stat().st_size / 1024)
    df = loader(p, session=session)
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


def _save_sav(df: pl.DataFrame, path: Path) -> None:
    """Save to SPSS .sav (optional dependency)."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Writing .sav files requires pyreadstat. "
            "Install it with: pip install openstat[spss]"
        )
    pandas_df = df.to_pandas()
    pyreadstat.write_sav(pandas_df, str(path))


def save_file(df: pl.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to CSV, Parquet, Excel, Stata, or SPSS."""
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
    elif suffix == ".sav":
        _save_sav(df, p)
    else:
        raise ValueError(
            f"Unsupported save format: {suffix}  "
            "(use .csv, .parquet, .xlsx, .dta, or .sav)"
        )
    return p
