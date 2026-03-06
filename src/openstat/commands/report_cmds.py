"""Report and help commands."""

from __future__ import annotations

from openstat.session import Session
from openstat.reporting.report import generate_report
from openstat.commands.base import command, get_registry


@command("report", usage="report [eda [path.html] | path.md]")
def cmd_report(session: Session, args: str) -> str:
    """Generate a Markdown report or automated EDA HTML report."""
    stripped = args.strip()
    if stripped.startswith("eda"):
        path = stripped[3:].strip() or "outputs/eda_report.html"
        try:
            from openstat.reporting.eda import generate_eda_report
            out = generate_eda_report(session, path)
            return f"EDA report saved: {out}"
        except Exception as e:
            return f"EDA report error: {e}"

    path = stripped or "outputs/report.md"
    try:
        out = generate_report(session, path)
        return f"Report saved: {out}"
    except Exception as e:
        return f"Report error: {e}"

