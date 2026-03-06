"""UX helpers: bookmark, history search, timer, multiline."""

from __future__ import annotations
import time
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ── Bookmark ─────────────────────────────────────────────────────────────────

@command("bookmark", usage="bookmark save|load|list|rm [name] [command]")
def cmd_bookmark(session: Session, args: str) -> str:
    """Save and recall frequently used commands as bookmarks.

    Sub-commands:
      bookmark save <name> <command>   — save a command as bookmark
      bookmark load <name>             — execute a bookmarked command
      bookmark list                    — list all bookmarks
      bookmark rm <name>               — remove a bookmark

    Examples:
      bookmark save myols "ols income educ age"
      bookmark load myols
      bookmark list
      bookmark rm myols
    """
    if not hasattr(session, "_bookmarks"):
        session._bookmarks = {}  # type: ignore[attr-defined]
    bm: dict = session._bookmarks  # type: ignore[attr-defined]

    tokens = args.strip().split(None, 2)
    if not tokens:
        return "Usage: bookmark save|load|list|rm [name] [command]"

    subcmd = tokens[0].lower()

    if subcmd == "list":
        if not bm:
            return "No bookmarks saved. Use: bookmark save <name> <command>"
        lines = ["Bookmarks:"]
        for name, cmd in bm.items():
            lines.append(f"  {name:<20} {cmd}")
        return "\n".join(lines)

    if subcmd == "rm":
        name = tokens[1] if len(tokens) > 1 else ""
        if name in bm:
            del bm[name]
            return f"Bookmark '{name}' removed."
        return f"Bookmark '{name}' not found."

    if subcmd == "save":
        if len(tokens) < 3:
            return "Usage: bookmark save <name> <command>"
        name = tokens[1]
        cmd_str = tokens[2].strip("\"'")
        bm[name] = cmd_str
        return f"Bookmark '{name}' saved: {cmd_str}"

    if subcmd == "load":
        name = tokens[1] if len(tokens) > 1 else ""
        if name not in bm:
            return f"Bookmark '{name}' not found. Use 'bookmark list' to see saved."
        cmd_str = bm[name]
        from openstat.commands.base import run_command
        return run_command(session, cmd_str)

    return f"Unknown sub-command: {subcmd}"


# ── History search ────────────────────────────────────────────────────────────

@command("history search", usage="history search <pattern> [--n=20]")
def cmd_history_search(session: Session, args: str) -> str:
    """Search command history for matching entries.

    Options:
      --n=<k>    maximum results to show (default: 20)

    Examples:
      history search ols
      history search "export" --n=10
      history search plot
    """
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: history search <pattern>"

    pattern = " ".join(ca.positional).lower()
    top_n = int(ca.options.get("n", 20))

    history = getattr(session, "history", [])
    if not history:
        return "History is empty."

    matches = [
        (i + 1, cmd)
        for i, cmd in enumerate(history)
        if pattern in cmd.lower()
    ]

    if not matches:
        return f"No history entries matching: {pattern}"

    lines = [f"History search: '{pattern}' ({len(matches)} matches)"]
    for idx, cmd in matches[-top_n:]:
        lines.append(f"  {idx:>5}  {cmd}")
    return "\n".join(lines)


@command("history show", usage="history show [--n=20]")
def cmd_history_show(session: Session, args: str) -> str:
    """Show recent command history.

    Options:
      --n=<k>   number of recent commands (default: 20)

    Examples:
      history show
      history show --n=50
    """
    ca = CommandArgs(args)
    top_n = int(ca.options.get("n", 20))
    history = getattr(session, "history", [])
    if not history:
        return "History is empty."
    recent = history[-top_n:]
    offset = len(history) - len(recent)
    lines = [f"Recent history ({len(recent)} of {len(history)} entries):"]
    for i, cmd in enumerate(recent, offset + 1):
        lines.append(f"  {i:>5}  {cmd}")
    return "\n".join(lines)


# ── Timer ────────────────────────────────────────────────────────────────────

@command("timer", usage="timer <command with args>")
def cmd_timer(session: Session, args: str) -> str:
    """Time the execution of a command.

    Runs the specified command and reports wall-clock time.

    Examples:
      timer ols income educ age
      timer bootstrap ols income educ age --reps=500
      timer describe
    """
    if not args.strip():
        return "Usage: timer <command> [args]"

    from openstat.commands.base import run_command

    start = time.perf_counter()
    result = run_command(session, args.strip())
    elapsed = time.perf_counter() - start

    sep = "-" * 50
    if elapsed < 1:
        time_str = f"{elapsed * 1000:.1f} ms"
    elif elapsed < 60:
        time_str = f"{elapsed:.3f} s"
    else:
        m, s = divmod(elapsed, 60)
        time_str = f"{int(m)}m {s:.1f}s"

    parts = [result, "", sep, f"Elapsed: {time_str}  (command: {args.strip()[:60]})"]
    return "\n".join(parts)


# ── Multiline input helper ────────────────────────────────────────────────────

@command("multiline", usage="multiline")
def cmd_multiline(session: Session, args: str) -> str:
    """Show instructions for multiline input in the REPL.

    In the OpenStat REPL, you can use backslash continuation:
      ols income educ age \\
          --robust \\
          --cluster=state

    Or use a semicolon to chain commands on one line:
      describe; summarize income educ

    Or write commands to a .ost script file and run:
      run my_analysis.ost

    For interactive multiline, use the pipeline command:
      pipeline define myflow "ols y x" | "plot coef" | "export pdf"
      pipeline run myflow
    """
    return cmd_multiline.__doc__ or "See: pipeline define / run"
