"""Reproducibility commands: set seed, session save/replay/info."""

from __future__ import annotations

import datetime
import re
from pathlib import Path

from openstat.commands.base import command, get_registry
from openstat.session import Session
from openstat import __version__


# Module-level seed tracking
_current_seed: int | None = None


def get_current_seed() -> int | None:
    return _current_seed



@command("session", usage="session info | session save <path> | session replay <path>")
def cmd_session(session: Session, args: str) -> str:
    """Session management: view info, save commands to script, replay a script.

    Examples:
      session info              — show session details
      session save analysis.ost — save all commands to a script file
      session replay script.ost — run an .ost script in current session
    """
    tokens = args.strip().split(None, 1)
    subcmd = tokens[0].lower() if tokens else "info"

    if subcmd == "info":
        seed = getattr(session, "_repro_seed", _current_seed)
        lines = [
            "Session Information",
            "=" * 50,
            f"  OpenStat version : {__version__}",
            f"  Dataset          : {session.dataset_name or '(none)'}",
            f"  Shape            : {session.shape_str}",
            f"  Random seed      : {seed if seed is not None else '(not set)'}",
            f"  Commands run     : {len(session.history)}",
            f"  Models fitted    : {len(session.results)}",
            f"  Plots generated  : {len(session.plot_paths)}",
            f"  Output dir       : {session.output_dir}",
            f"  Log file         : {session._log_path or '(none)'}",
            f"  Timestamp        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return "\n".join(lines)

    elif subcmd == "save":
        path = tokens[1].strip() if len(tokens) > 1 else "session_script.ost"
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        seed = getattr(session, "_repro_seed", _current_seed)

        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(f"# OpenStat script — saved {datetime.datetime.now().isoformat()}\n")
            f.write(f"# OpenStat version: {__version__}\n")
            if seed is not None:
                f.write(f"# Random seed: {seed}\n")
                f.write(f"set seed {seed}\n")
            f.write(f"# Dataset: {session.dataset_name or '(none)'}\n")
            f.write("\n")
            for cmd_line in session.history:
                # Skip the current 'session save' command
                if cmd_line.strip().startswith("session save"):
                    continue
                f.write(f"{cmd_line}\n")

        return f"Session saved to: {path_obj.absolute()} ({len(session.history)} commands)"

    elif subcmd == "replay":
        path = tokens[1].strip() if len(tokens) > 1 else None
        if not path:
            return "Usage: session replay <path.ost>"
        if not Path(path).exists():
            return f"File not found: {path}"
        # Use run_script
        from openstat.repl import run_script
        try:
            run_script(path, session)
            return f"Replayed: {path}"
        except SystemExit:
            return f"Replay stopped due to error in: {path}"
        except Exception as exc:
            return f"Replay error: {exc}"

    else:
        return (
            "Usage:\n"
            "  session info              — view session details\n"
            "  session save <path.ost>   — save commands to script\n"
            "  session replay <path.ost> — run a script file"
        )


@command("version", usage="version")
def cmd_version(session: Session, args: str) -> str:
    """Show OpenStat version and environment information."""
    import sys
    import platform

    lines = [
        f"OpenStat {__version__}",
        f"Python  {sys.version.split()[0]}",
        f"Platform {platform.system()} {platform.machine()}",
    ]
    deps = [
        ("polars", "polars"),
        ("numpy", "numpy"),
        ("statsmodels", "statsmodels"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    ]
    for name, mod in deps:
        try:
            m = __import__(mod)
            lines.append(f"  {name:<15} {m.__version__}")
        except ImportError:
            lines.append(f"  {name:<15} (not installed)")

    seed = getattr(session, "_repro_seed", _current_seed)
    if seed is not None:
        lines.append(f"Random seed: {seed}")

    return "\n".join(lines)
