"""Automated EDA report generation (self-contained HTML)."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path


def _b64_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _color_corr(val: float) -> str:
    """Return background hex color for a correlation value."""
    if val >= 0.7:
        return "#2ecc71"
    if val >= 0.4:
        return "#a8e6b1"
    if val <= -0.7:
        return "#e74c3c"
    if val <= -0.4:
        return "#f5b7b1"
    return "#ffffff"


def generate_eda_report(session, path: str) -> str:
    """Generate a self-contained HTML EDA report.

    Parameters
    ----------
    session : Session
    path    : output file path (HTML)

    Returns
    -------
    Absolute path of the generated file.
    """
    from openstat.session import Session  # local import to avoid circulars

    df = session.require_data()
    import polars as pl

    # ── Section helpers ────────────────────────────────────────────────
    sections: list[str] = []

    # 1. Dataset Overview
    n_rows, n_cols = df.shape
    n_missing = sum(df[c].null_count() for c in df.columns)
    mem_kb = df.estimated_size() / 1024
    dtypes_html = "".join(
        f"<tr><td>{c}</td><td>{df[c].dtype}</td></tr>" for c in df.columns
    )
    sections.append(f"""
<h2>1. Dataset Overview</h2>
<table>
  <tr><th>Rows</th><td>{n_rows:,}</td></tr>
  <tr><th>Columns</th><td>{n_cols}</td></tr>
  <tr><th>Missing cells</th><td>{n_missing:,}</td></tr>
  <tr><th>Memory (approx)</th><td>{mem_kb:.1f} KB</td></tr>
</table>
<h3>Column Types</h3>
<table><tr><th>Column</th><th>Type</th></tr>{dtypes_html}</table>
""")

    # 2. Missing Values
    miss_rows = ""
    for c in df.columns:
        cnt = df[c].null_count()
        pct = cnt / n_rows * 100 if n_rows else 0
        miss_rows += f"<tr><td>{c}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>"
    sections.append(f"""
<h2>2. Missing Values</h2>
<table>
  <tr><th>Column</th><th>Missing</th><th>%</th></tr>
  {miss_rows}
</table>
""")

    # 3. Numeric Summary
    numeric_cols = [c for c in df.columns if df[c].dtype in (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )]
    if numeric_cols:
        num_rows = ""
        for c in numeric_cols:
            s = df[c].drop_nulls()
            if s.len() == 0:
                continue
            import numpy as np
            arr = s.to_numpy().astype(float)
            num_rows += (
                f"<tr><td>{c}</td>"
                f"<td>{arr.min():.4g}</td><td>{arr.max():.4g}</td>"
                f"<td>{arr.mean():.4g}</td><td>{np.median(arr):.4g}</td>"
                f"<td>{arr.std():.4g}</td>"
                f"<td>{float(pl.Series(arr).skew() or 0):.3f}</td>"
                f"<td>{float(pl.Series(arr).kurtosis() or 0):.3f}</td>"
                "</tr>"
            )
        sections.append(f"""
<h2>3. Numeric Summary</h2>
<table>
  <tr><th>Column</th><th>Min</th><th>Max</th><th>Mean</th>
      <th>Median</th><th>Std</th><th>Skew</th><th>Kurt</th></tr>
  {num_rows}
</table>
""")

    # 4. Categorical Summary
    cat_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean)]
    if cat_cols:
        cat_html = ""
        for c in cat_cols:
            vc = df[c].value_counts().sort("count", descending=True).head(10)
            rows = "".join(
                f"<tr><td>{row[c]}</td><td>{row['count']}</td></tr>"
                for row in vc.iter_rows(named=True)
            )
            cat_html += f"<h3>{c}</h3><table><tr><th>Value</th><th>Count</th></tr>{rows}</table>"
        sections.append(f"<h2>4. Categorical Summary</h2>{cat_html}")

    # 5. Correlation Matrix
    if len(numeric_cols) >= 2:
        import numpy as np
        mat = df.select(numeric_cols).to_numpy().astype(float)
        mask = ~np.isnan(mat).any(axis=1)
        mat = mat[mask]
        corr = np.corrcoef(mat.T) if mat.shape[0] > 1 else np.eye(len(numeric_cols))
        header = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in numeric_cols) + "</tr>"
        corr_rows = ""
        for i, ci in enumerate(numeric_cols):
            corr_rows += f"<tr><th>{ci}</th>"
            for j in range(len(numeric_cols)):
                v = corr[i, j]
                bg = _color_corr(v)
                corr_rows += f'<td style="background:{bg}">{v:.3f}</td>'
            corr_rows += "</tr>"
        sections.append(f"""
<h2>5. Correlation Matrix</h2>
<table>{header}{corr_rows}</table>
""")

    # 6. Distribution Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_imgs = ""
        for c in numeric_cols[:20]:  # cap at 20 cols
            arr = df[c].drop_nulls().to_numpy().astype(float)
            if len(arr) < 2:
                continue
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(arr, bins=min(30, max(5, len(arr) // 10)), edgecolor="white")
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
            b64 = _b64_fig(fig)
            plt.close(fig)
            plot_imgs += f'<img src="data:image/png;base64,{b64}" style="margin:4px" />'
        if plot_imgs:
            sections.append(f"<h2>6. Distribution Plots</h2><div>{plot_imgs}</div>")
    except Exception:
        pass

    # 7. Model Results
    if session.results:
        model_html = ""
        for mr in session.results:
            model_html += f"<h3>{mr.name} — {mr.formula}</h3><pre>{mr.table}</pre>"
        sections.append(f"<h2>7. Model Results</h2>{model_html}")

    # ── Assemble HTML ──────────────────────────────────────────────────
    body = "\n".join(sections)
    dataset_name = session.dataset_name or "Dataset"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDA Report — {dataset_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
  h1   {{ color: #2c3e50; }}
  h2   {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #ecf0f1; text-align: left; }}
  td:first-child {{ text-align: left; }}
  pre {{ background: #f8f9fa; padding: 12px; border-radius: 4px; font-size: 12px; }}
  img {{ border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>EDA Report — {dataset_name}</h1>
<p>{n_rows:,} rows × {n_cols} columns &nbsp;|&nbsp; Generated by OpenStat</p>
{body}
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    return os.path.abspath(path)
