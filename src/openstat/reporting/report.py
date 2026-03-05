"""Markdown report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from openstat.session import Session


def generate_report(session: Session, output_path: str | Path) -> Path:
    """Generate a Markdown report from the session state."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# OpenStat Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Dataset info
    lines.append("## Dataset")
    lines.append("")
    if session.dataset_path:
        lines.append(f"- **Source**: `{session.dataset_path}`")
    if session.df is not None:
        r, c = session.df.shape
        lines.append(f"- **Shape**: {r:,} rows x {c} columns")
        lines.append(f"- **Columns**: {', '.join(session.df.columns)}")
    lines.append("")

    # Command history
    lines.append("## Commands Executed")
    lines.append("")
    lines.append("```")
    for cmd in session.history:
        lines.append(cmd)
    lines.append("```")
    lines.append("")

    # Model results
    if session.results:
        lines.append("## Model Results")
        lines.append("")
        for result in session.results:
            lines.append(result.table)
            lines.append("")

    # Plots
    if session.plot_paths:
        lines.append("## Plots")
        lines.append("")
        for plot_path in session.plot_paths:
            plot_p = Path(plot_path)
            name = plot_p.name
            # Use relative path from report location for portability
            try:
                rel = plot_p.resolve().relative_to(p.parent.resolve())
            except ValueError:
                rel = plot_p
            lines.append(f"![{name}]({rel})")
            lines.append("")

    content = "\n".join(lines)
    p.write_text(content, encoding="utf-8")
    return p
