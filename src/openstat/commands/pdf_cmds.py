"""PDF and Markdown export commands."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from openstat.commands.base import command, CommandArgs
from openstat.session import Session


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# ── Markdown export ──────────────────────────────────────────────────────────

def _export_md(session: Session, path: str) -> str:
    import polars as pl

    lines = [
        f"# OpenStat Results",
        f"",
        f"**Dataset:** {session.dataset_name or 'Unknown'}  |  "
        f"**Date:** {date.today().isoformat()}  |  "
        f"**Shape:** {session.shape_str}",
        f"",
    ]

    if session.df is not None:
        df = session.df
        lines += ["## Dataset Overview", ""]
        lines += [
            f"| Property | Value |",
            f"|---|---|",
            f"| Rows | {df.height:,} |",
            f"| Columns | {df.width} |",
            f"| Missing cells | {sum(df[c].null_count() for c in df.columns)} |",
        ]
        lines.append("")

        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        num_cols = [c for c in df.columns if df[c].dtype in NUMERIC]
        if num_cols:
            lines += ["## Summary Statistics", ""]
            lines.append("| Variable | N | Mean | SD | Min | Max |")
            lines.append("|---|---|---|---|---|---|")
            for c in num_cols[:30]:
                col = df[c].drop_nulls()
                if col.len() == 0:
                    continue
                mean = f"{col.mean():.4f}"
                sd = f"{col.std():.4f}" if col.len() > 1 else "—"
                lines.append(
                    f"| {c} | {col.len()} | {mean} | {sd} "
                    f"| {col.min():.4f} | {col.max():.4f} |"
                )
            lines.append("")

    for mr in session.results:
        lines += [
            f"## {mr.name}: {mr.formula}",
            "",
            "```",
            mr.table,
            "```",
            "",
        ]

    if session.plot_paths:
        lines += ["## Figures", ""]
        for p in session.plot_paths:
            if os.path.exists(p):
                lines.append(f"![Figure]({p})")
        lines.append("")

    content = "\n".join(lines)
    _ensure_dir(path)
    Path(path).write_text(content, encoding="utf-8")
    return os.path.abspath(path)


# ── PDF export ───────────────────────────────────────────────────────────────

def _export_pdf(session: Session, path: str) -> str:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
            HRFlowable,
        )
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        return (
            "reportlab is required for PDF export.\n"
            "Install: pip install reportlab"
        )

    import polars as pl

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    mono = ParagraphStyle("Mono", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=11)
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["Normal"]

    story = []

    # Title
    story.append(Paragraph("OpenStat Results", h1))
    story.append(Paragraph(
        f"Dataset: {session.dataset_name or 'Unknown'} &nbsp;|&nbsp; "
        f"Date: {date.today().isoformat()} &nbsp;|&nbsp; "
        f"Shape: {session.shape_str}",
        normal,
    ))
    story.append(HRFlowable(width="100%"))
    story.append(Spacer(1, 0.3 * cm))

    # Dataset overview table
    if session.df is not None:
        df = session.df
        story.append(Paragraph("Dataset Overview", h2))
        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        overview_data = [
            ["Property", "Value"],
            ["Rows", f"{df.height:,}"],
            ["Columns", str(df.width)],
            ["Missing cells", str(sum(df[c].null_count() for c in df.columns))],
            ["Numeric columns", str(sum(1 for c in df.columns if df[c].dtype in NUMERIC))],
        ]
        tbl = Table(overview_data, colWidths=[6 * cm, 10 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

        # Summary statistics
        num_cols = [c for c in df.columns if df[c].dtype in NUMERIC]
        if num_cols:
            story.append(Paragraph("Summary Statistics", h2))
            stats_data = [["Variable", "N", "Mean", "SD", "Min", "Max"]]
            for c in num_cols[:25]:
                col = df[c].drop_nulls()
                if col.len() == 0:
                    continue
                sd_str = f"{col.std():.4f}" if col.len() > 1 else "—"
                stats_data.append([
                    c, str(col.len()),
                    f"{col.mean():.4f}", sd_str,
                    f"{col.min():.4f}", f"{col.max():.4f}",
                ])
            st = Table(stats_data, colWidths=[4*cm, 1.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(st)
            story.append(Spacer(1, 0.5 * cm))

    # Model results
    for mr in session.results:
        story.append(Paragraph(f"{mr.name}: {mr.formula}", h2))
        # Wrap long table text in monospace paragraphs
        for line in mr.table.split("\n"):
            story.append(Paragraph(line.replace(" ", "&nbsp;") or "&nbsp;", mono))
        story.append(Spacer(1, 0.4 * cm))

    # Plots
    for plot_path in session.plot_paths:
        if os.path.exists(plot_path):
            story.append(Paragraph("Figure", h2))
            try:
                img = Image(plot_path, width=15 * cm, height=10 * cm, kind="proportional")
                story.append(img)
            except Exception:
                story.append(Paragraph(f"[Plot: {plot_path}]", normal))
            story.append(Spacer(1, 0.3 * cm))

    _ensure_dir(path)
    doc.build(story)
    return os.path.abspath(path)


# ── Commands ─────────────────────────────────────────────────────────────────

@command("export pdf", usage="export pdf [path]")
def cmd_export_pdf(session: Session, args: str) -> str:
    """Export results to a PDF report (requires reportlab)."""
    ca = CommandArgs(args)
    path = ca.positional[0] if ca.positional else "outputs/results.pdf"
    out = _export_pdf(session, path)
    if out.endswith(".pdf"):
        return f"PDF saved: {out}"
    return out


@command("export md", usage="export md [path]")
def cmd_export_md(session: Session, args: str) -> str:
    """Export results to a Markdown file."""
    ca = CommandArgs(args)
    path = ca.positional[0] if ca.positional else "outputs/results.md"
    out = _export_md(session, path)
    return f"Markdown saved: {out}"
