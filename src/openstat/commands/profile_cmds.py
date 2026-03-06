"""Data profile and data dictionary commands."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("profile", usage="profile [col1 col2 ...] [--out=report.html]")
def cmd_profile(session: Session, args: str) -> str:
    """Generate a comprehensive data profile report.

    Shows for each column: type, missing count/%, unique values,
    min/max/mean/std/median/mode, top values, distribution shape.

    Options:
      --out=<path>    save as HTML report (default: outputs/profile.html)
      --cols=<list>   comma-separated column subset

    Examples:
      profile
      profile income age education
      profile --out=data_profile.html
    """
    import polars as pl

    ca = CommandArgs(args)
    try:
        df = session.require_data()
    except RuntimeError as e:
        return str(e)

    # Column subset
    cols_opt = ca.options.get("cols")
    if cols_opt:
        cols = [c.strip() for c in cols_opt.split(",")]
    elif ca.positional:
        cols = ca.positional
    else:
        cols = df.columns

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return f"Columns not found: {', '.join(missing_cols)}"

    NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    lines = [
        f"Data Profile: {session.dataset_name or 'dataset'}",
        f"Shape: {df.height:,} rows × {df.width} columns  |  Showing {len(cols)} columns",
        "=" * 72,
    ]

    for col in cols:
        series = df[col]
        n_miss = series.null_count()
        miss_pct = 100 * n_miss / df.height if df.height else 0
        n_uniq = series.drop_nulls().n_unique()
        dtype = str(series.dtype)

        lines.append(f"\n  {col}  [{dtype}]")
        lines.append(f"    Missing:  {n_miss:,} ({miss_pct:.1f}%)")
        lines.append(f"    Unique:   {n_uniq:,}")

        if series.dtype in NUMERIC:
            s = series.drop_nulls()
            if s.len() > 0:
                lines.append(f"    Mean:     {s.mean():.4f}")
                lines.append(f"    Std:      {s.std():.4f}" if s.len() > 1 else "    Std:      —")
                lines.append(f"    Min:      {s.min():.4f}")
                lines.append(f"    Median:   {s.median():.4f}")
                lines.append(f"    Max:      {s.max():.4f}")
                # Skewness / kurtosis
                try:
                    import numpy as np
                    arr = s.to_numpy()
                    from scipy.stats import skew, kurtosis
                    lines.append(f"    Skewness: {skew(arr):.3f}")
                    lines.append(f"    Kurtosis: {kurtosis(arr):.3f}")
                except Exception:
                    pass
                # Zeros / negatives
                n_zero = int((s == 0).sum())
                n_neg = int((s < 0).sum())
                if n_zero or n_neg:
                    lines.append(f"    Zeros:    {n_zero:,}   Negative: {n_neg:,}")
        else:
            # Categorical / string
            s = series.drop_nulls().cast(pl.Utf8)
            top = s.value_counts().sort("count", descending=True).head(5)
            if top.height > 0:
                top_vals = ", ".join(
                    f"{row[0]}({row[1]})" for row in top.iter_rows()
                )
                lines.append(f"    Top 5:    {top_vals}")

    lines.append("\n" + "=" * 72)

    # HTML output
    out_path = ca.options.get("out")
    if out_path:
        try:
            _save_profile_html(lines, out_path, session)
            lines.append(f"\nHTML report saved: {out_path}")
        except Exception as exc:
            lines.append(f"\nHTML save failed: {exc}")

    return "\n".join(lines)


def _save_profile_html(lines: list[str], path: str, session: Session) -> None:
    """Save a simple HTML version of the profile."""
    import polars as pl
    from pathlib import Path

    df = session.df
    NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    rows_html = ""
    if df is not None:
        for col in df.columns:
            series = df[col]
            n_miss = series.null_count()
            miss_pct = f"{100*n_miss/df.height:.1f}%" if df.height else "—"
            n_uniq = series.drop_nulls().n_unique()
            dtype = str(series.dtype)
            if series.dtype in NUMERIC:
                s = series.drop_nulls()
                stats = f"mean={s.mean():.3f}, std={s.std():.3f}" if s.len() > 0 else "—"
            else:
                stats = f"{n_uniq} unique values"
            rows_html += (
                f"<tr><td>{col}</td><td>{dtype}</td>"
                f"<td>{n_miss} ({miss_pct})</td>"
                f"<td>{n_uniq}</td><td>{stats}</td></tr>\n"
            )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>OpenStat Data Profile</title>
<style>
body {{font-family: sans-serif; margin: 2em; background: #f9f9f9;}}
h1 {{color: #333;}} table {{border-collapse: collapse; width: 100%;}}
th {{background: #4C72B0; color: white; padding: 8px;}}
td {{border: 1px solid #ddd; padding: 6px;}}
tr:nth-child(even) {{background: #f0f4ff;}}
</style></head><body>
<h1>Data Profile: {session.dataset_name or "dataset"}</h1>
<p>Shape: {session.shape_str}</p>
<table>
<tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Stats</th></tr>
{rows_html}
</table>
<pre>{"chr(10)".join(lines)}</pre>
</body></html>"""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")


@command("datadict", usage="datadict [--out=dict.xlsx|dict.md]")
def cmd_datadict(session: Session, args: str) -> str:
    """Generate a data dictionary for the current dataset.

    Creates a table with: variable name, type, missing%, unique count,
    min/max/mean for numeric, top values for categorical.

    Options:
      --out=<path>   save to Excel (.xlsx) or Markdown (.md)
                     Default: outputs/data_dictionary.md

    Examples:
      datadict
      datadict --out=dictionary.xlsx
      datadict --out=docs/variables.md
    """
    import polars as pl
    from pathlib import Path

    ca = CommandArgs(args)
    out_path = ca.options.get("out", "outputs/data_dictionary.md")

    try:
        df = session.require_data()
    except RuntimeError as e:
        return str(e)

    NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    records = []
    for col in df.columns:
        series = df[col]
        n_miss = series.null_count()
        miss_pct = f"{100*n_miss/df.height:.1f}%"
        n_uniq = series.drop_nulls().n_unique()
        dtype = str(series.dtype)

        if series.dtype in NUMERIC:
            s = series.drop_nulls()
            if s.len() > 0:
                extra = f"mean={s.mean():.3f}; range=[{s.min():.3f},{s.max():.3f}]"
            else:
                extra = "all missing"
        else:
            top = series.drop_nulls().cast(pl.Utf8).value_counts().sort("count", descending=True).head(3)
            top_vals = "; ".join(str(r[0]) for r in top.iter_rows())
            extra = f"top: {top_vals}"

        records.append({
            "Variable": col,
            "Type": dtype,
            "Missing": f"{n_miss} ({miss_pct})",
            "Unique": str(n_uniq),
            "Notes": extra,
            "Description": "",  # user fills in
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if out_path.endswith(".xlsx"):
        try:
            dict_df = pl.DataFrame(records)
            dict_df.write_excel(out_path)
            return f"Data dictionary saved: {out_path} ({len(records)} variables)"
        except ImportError:
            return "xlsxwriter required for Excel output. Try --out=dict.md"

    else:  # Markdown
        lines = [
            f"# Data Dictionary: {session.dataset_name or 'dataset'}",
            f"",
            f"Shape: {df.height:,} rows × {df.width} columns",
            f"",
            "| Variable | Type | Missing | Unique | Notes | Description |",
            "|---|---|---|---|---|---|",
        ]
        for r in records:
            lines.append(
                f"| {r['Variable']} | {r['Type']} | {r['Missing']} | "
                f"{r['Unique']} | {r['Notes']} | {r['Description']} |"
            )
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        return f"Data dictionary saved: {out_path} ({len(records)} variables)"
