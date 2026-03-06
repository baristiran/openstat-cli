"""Jupyter display helpers for OpenStat."""

from __future__ import annotations


def fit_result_to_html(result) -> str:
    """Convert a FitResult to HTML table for Jupyter rendering."""
    from rich.console import Console
    console = Console(record=True, width=120)
    text = result.summary_table()
    console.print(text)
    return console.export_html(inline_styles=True)


def dataframe_to_html(df, max_rows: int = 50) -> str:
    """Convert a Polars DataFrame to styled HTML for Jupyter."""
    pdf = df.head(max_rows).to_pandas()
    return pdf.to_html(classes="openstat-table", border=0, index=False)
