"""Watch command: re-run a script automatically when the file changes."""

from __future__ import annotations

import time
import os

from openstat.commands.base import command, CommandArgs
from openstat.session import Session


@command("watch", usage="watch <script.ost> [--interval=2]")
def cmd_watch(session: Session, args: str) -> str:
    """Watch a script file and re-run it whenever it changes.

    Monitors the file's modification time every N seconds (default 2).
    Press Ctrl+C to stop watching.

    Examples:
      watch analysis.ost
      watch pipeline.ost --interval=5
    """
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: watch <script.ost> [--interval=2]"

    script_path = ca.positional[0]
    interval = float(ca.options.get("interval", 2))

    if not os.path.exists(script_path):
        return f"File not found: {script_path}"

    from openstat.script_runner import run_script_advanced

    last_mtime = os.path.getmtime(script_path)

    print(f"Watching {script_path} (interval={interval}s). Press Ctrl+C to stop.")

    try:
        # Run once immediately
        run_script_advanced(script_path, session)
        print(f"[Initial run complete]")

        while True:
            time.sleep(interval)
            try:
                current_mtime = os.path.getmtime(script_path)
            except FileNotFoundError:
                print(f"File removed: {script_path}. Stopping watch.")
                break

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                print(f"\n[{_timestamp()}] File changed — re-running {script_path}...")
                try:
                    run_script_advanced(script_path, session)
                    print(f"[{_timestamp()}] Done.")
                except Exception as exc:
                    print(f"[{_timestamp()}] Error: {exc}")

    except KeyboardInterrupt:
        pass

    return f"Watch stopped: {script_path}"


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
