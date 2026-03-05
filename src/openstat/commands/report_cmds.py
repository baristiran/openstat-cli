"""Report and help commands."""

from __future__ import annotations

from openstat.session import Session
from openstat.reporting.report import generate_report
from openstat.commands.base import command, get_registry


@command("report", usage="report [path.md]")
def cmd_report(session: Session, args: str) -> str:
    """Generate a Markdown report of the session."""
    path = args.strip() or "outputs/report.md"
    try:
        out = generate_report(session, path)
        return f"Report saved: {out}"
    except Exception as e:
        return f"Report error: {e}"


@command("help", usage="help [command]")
def cmd_help(session: Session, args: str) -> str:
    """Show available commands or help for a specific command."""
    registry = get_registry()
    if args.strip() and args.strip() in registry:
        handler = registry[args.strip()]
        from openstat.commands.base import get_usage
        usage = get_usage(args.strip())
        doc = handler.__doc__ or "No description."
        return f"{args.strip()}: {doc}\nUsage: {usage}"

    lines = ["Available commands:", ""]
    for name, handler in sorted(registry.items()):
        doc = (handler.__doc__ or "").split("\n")[0]
        lines.append(f"  {name:<25} {doc}")
    lines.append("")
    lines.append("Type 'help <command>' for details. Type 'quit' to exit.")
    return "\n".join(lines)
