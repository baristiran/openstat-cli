"""Interactive REPL for OpenStat."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.console import Console

from openstat import __version__
from openstat.session import Session
from openstat.commands import COMMANDS
from openstat.logging_config import get_logger

console = Console()
log = get_logger("repl")

_HISTORY_DIR = Path.home() / ".openstat"
_HISTORY_FILE = _HISTORY_DIR / "history"

_BANNER = f"""\
[bold cyan]OpenStat v{__version__}[/bold cyan] — Open-source statistical analysis tool
Type [green]help[/green] for commands, [green]quit[/green] to exit.
"""

_EXIT_COMMANDS = {"quit", "exit", "q"}


_FILE_COMMANDS = {"load", "save", "merge", "run"}
_FILLNA_STRATEGIES = ("mean", "median", "mode", "forward", "backward")
_CAST_TYPES = ("int", "float", "str", "bool")


class _DynamicCompleter(Completer):
    """Tab-complete command names, column names, file paths, and sub-options."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._commands = sorted(set(COMMANDS.keys()) | _EXIT_COMMANDS)
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document: Document, complete_event):  # type: ignore[override]
        text = document.text_before_cursor
        words = text.split()
        word = document.get_word_before_cursor()

        if len(words) <= 1 and not text.endswith(" "):
            # First word → command names
            for cmd in self._commands:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))
        else:
            cmd = words[0].lower() if words else ""

            # File path completion for load/save/merge
            if cmd in _FILE_COMMANDS and len(words) <= 2:
                yield from self._path_completer.get_completions(document, complete_event)
                return

            # Sub-commands for 'plot'
            if cmd == "plot" and len(words) <= 2:
                for sub in ("hist", "scatter", "line", "box", "bar", "heatmap", "diagnostics"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # Strategy completion for 'fillna'
            if cmd == "fillna" and len(words) == 3:
                for s in _FILLNA_STRATEGIES:
                    if s.startswith(word):
                        yield Completion(s, start_position=-len(word))

            # Type completion for 'cast'
            if cmd == "cast" and len(words) == 3:
                for t in _CAST_TYPES:
                    if t.startswith(word):
                        yield Completion(t, start_position=-len(word))

            # Option completions for new v0.2.0 commands
            if cmd == "margins" and word.startswith("--at="):
                prefix = "--at="
                for opt in ("means", "average"):
                    full = prefix + opt
                    if full.startswith(word):
                        yield Completion(full, start_position=-len(word))

            if cmd == "margins" and word.startswith("--a"):
                yield Completion("--at=", start_position=-len(word))

            if cmd == "quantreg" and word.startswith("tau"):
                for tau in ("tau=0.25", "tau=0.5", "tau=0.75", "tau=0.9"):
                    if tau.startswith(word):
                        yield Completion(tau, start_position=-len(word))

            if cmd == "bootstrap":
                if word.startswith("n"):
                    for opt in ("n=100", "n=500", "n=1000"):
                        if opt.startswith(word):
                            yield Completion(opt, start_position=-len(word))
                if word.startswith("ci"):
                    for opt in ("ci=90", "ci=95", "ci=99"):
                        if opt.startswith(word):
                            yield Completion(opt, start_position=-len(word))

            if cmd in ("ols", "logit", "probit", "poisson", "negbin"):
                if word.startswith("--c"):
                    yield Completion("--cluster=", start_position=-len(word))
                if word.startswith("--r"):
                    yield Completion("--robust", start_position=-len(word))

            if cmd == "poisson" and word.startswith("--e"):
                yield Completion("--exposure=", start_position=-len(word))

            if cmd == "estat" and len(words) <= 2:
                for sub in ("hettest", "ovtest", "linktest", "ic", "all"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            if cmd == "estimates" and len(words) <= 2:
                if "table".startswith(word):
                    yield Completion("table", start_position=-len(word))

            # Column names (if data loaded) — default fallback
            if self.session.df is not None:
                for col in self.session.df.columns:
                    if col.startswith(word):
                        yield Completion(col, start_position=-len(word))


def _dispatch(session: Session, line: str) -> str | None:
    """Parse and execute a single command line."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split(None, 1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd_name in _EXIT_COMMANDS:
        return "__QUIT__"

    handler = COMMANDS.get(cmd_name)
    if handler is None:
        return f"Unknown command: {cmd_name}. Type 'help' for available commands."

    session.record(line)
    log.debug("dispatch: %s (args=%r)", cmd_name, args)
    try:
        result = handler(session, args)
        log.debug("result length: %d", len(result) if result else 0)
        return result
    except Exception as e:
        log.exception("Unhandled error in command '%s'", cmd_name)
        import logging
        import traceback
        msg = f"Internal error: {e}"
        if logging.getLogger("openstat").isEnabledFor(logging.DEBUG):
            msg += "\n" + traceback.format_exc()
        return msg


def run_repl(session: Session | None = None) -> None:
    """Start the interactive REPL."""
    if session is None:
        session = Session()

    console.print(_BANNER)

    completer = _DynamicCompleter(session)

    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        completer=completer,
    )

    while True:
        try:
            line = prompt_session.prompt("openstat> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        result = _dispatch(session, line)
        if result == "__QUIT__":
            console.print("Bye!")
            break
        if result:
            console.print(result)


def run_script(
    path: str, session: Session | None = None, *, strict: bool = False
) -> None:
    """Execute an .ost script file.

    If strict=True, stop on first error and raise SystemExit(1).
    """
    if session is None:
        session = Session()

    log.info("Running script: %s (strict=%s)", path, strict)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        console.print(f"[dim]>>> {line}[/dim]")
        result = _dispatch(session, line)
        if result == "__QUIT__":
            break
        if result:
            console.print(result)
            if strict:
                # Strip Rich markup before checking for error prefixes
                import re as _re
                plain = _re.sub(r"\[/?[^\]]*\]", "", result)
                if plain.startswith(("Error", "Internal error")):
                    log.error("Script failed at line %d: %s", i, line)
                    raise SystemExit(1)
        console.print()  # blank line between commands
