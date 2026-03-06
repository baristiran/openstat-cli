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


_FILE_COMMANDS = {"load", "save", "merge", "run", "append"}
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

            if cmd in ("ols", "logit", "probit", "poisson", "negbin",
                       "tobit", "mlogit", "ologit", "oprobit", "did"):
                if word.startswith("--c"):
                    yield Completion("--cluster=", start_position=-len(word))
                if word.startswith("--r"):
                    yield Completion("--robust", start_position=-len(word))

            # Tobit limit completions
            if cmd == "tobit":
                if word.startswith("ll"):
                    yield Completion("ll(0)", start_position=-len(word))
                if word.startswith("ul"):
                    yield Completion("ul()", start_position=-len(word))

            # psmatch option completions
            if cmd == "psmatch":
                if word.startswith("treat"):
                    yield Completion("treatment()", start_position=-len(word))
                if word.startswith("cal"):
                    yield Completion("caliper(0.2)", start_position=-len(word))
                if word.startswith("nn"):
                    for sub in ("nn(1)", "nn(3)", "nn(5)"):
                        if sub.startswith(word):
                            yield Completion(sub, start_position=-len(word))

            # egen function completions
            if cmd == "egen" and "=" in text:
                for fn in ("mean", "sum", "min", "max", "median", "count",
                            "rank", "group", "rowtotal", "rowmean"):
                    if fn.startswith(word):
                        yield Completion(fn + "(", start_position=-len(word))

            if cmd == "poisson" and word.startswith("--e"):
                yield Completion("--exposure=", start_position=-len(word))

            if cmd == "estat" and len(words) <= 2:
                for sub in ("hettest", "ovtest", "linktest", "ic", "all",
                            "icc", "lrtest", "firststage", "overid", "endogtest",
                            "phtest", "deff", "screeplot", "loadings"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # power sub-commands
            if cmd == "power" and len(words) <= 2:
                for sub in ("onemean", "twomeans", "oneprop", "twoprop", "ols"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # power option completions
            if cmd == "power" and len(words) > 2:
                for opt in ("n(", "alpha(0.05)", "power(0.80)", "delta(", "sd(1)",
                             "p0(", "pa(", "p1(", "p2(", "f2(", "k(", "ratio(1)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # sampsi option completions
            if cmd == "sampsi" and len(words) > 2:
                for opt in ("sd(1)", "alpha(0.05)", "power(0.80)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # report sub-commands
            if cmd == "report" and len(words) <= 2:
                for sub in ("eda",):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # export format completions
            if cmd == "export" and len(words) <= 2:
                for fmt in ("docx", "pptx"):
                    if fmt.startswith(word):
                        yield Completion(fmt, start_position=-len(word))

            # factor / pca option completions
            if cmd in ("factor", "pca") and len(words) > 1:
                for opt in ("n(", "method(pc)", "method(ml)", "--norotate"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # estat loadings blanks option
            if cmd == "estat" and len(words) > 2 and words[1] == "loadings":
                for opt in ("blanks(0.3)", "blanks(0.5)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # nonparametric tests
            if cmd in ("ranksum", "kwallis") and "by" not in text:
                if "by".startswith(word):
                    yield Completion("by(", start_position=-len(word))

            if cmd == "spearman" and self.session.df is not None:
                pass  # handled by column name fallback below

            # ML commands
            if cmd in ("lasso", "ridge", "elasticnet"):
                for opt in ("alpha(", "cv(5)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            if cmd == "elasticnet":
                if "l1ratio".startswith(word):
                    yield Completion("l1ratio(0.5)", start_position=-len(word))

            if cmd == "cart":
                for opt in ("depth(5)", "task(regression)", "task(classification)", "minleaf(5)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            if cmd == "crossval":
                for opt in ("method(ols)", "method(lasso)", "method(ridge)", "method(cart)",
                            "k(5)", "k(10)", "scoring(r2)", "scoring(neg_mean_squared_error)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # clustering
            if cmd == "cluster" and len(words) <= 2:
                for sub in ("kmeans", "hierarchical"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            if cmd == "cluster":
                for opt in ("k(3)", "k(5)", "linkage(ward)", "linkage(complete)", "linkage(average)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            if cmd == "discriminant":
                for opt in ("method(lda)", "method(qda)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # MANOVA / ANOVA2
            if cmd == "manova" and "=" not in text:
                if "=".startswith(word):
                    yield Completion("=", start_position=-len(word))

            # ARCH/GARCH
            if cmd in ("arch", "garch"):
                for opt in ("p(1)", "q(1)", "dist(normal)", "dist(t)",
                            "model(GARCH)", "model(EGARCH)", "model(GJR-GARCH)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # Bayesian
            if cmd == "bayes" and len(words) <= 2:
                if "ols".startswith(word) or ":".startswith(word):
                    yield Completion(": ols", start_position=-len(word))

            if cmd == "bayes":
                for opt in ("samples(4000)", "priorscale(10)", "ci(0.95)"):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            # reshape / collapse / encode
            if cmd == "reshape" and len(words) <= 2:
                for sub in ("wide", "long"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            if cmd in ("reshape", "collapse"):
                for opt in ("i(", "j(", "by("):
                    if opt.startswith(word):
                        yield Completion(opt, start_position=-len(word))

            if cmd == "collapse" and len(words) <= 2:
                for stat in ("(mean)", "(sum)", "(count)", "(median)", "(std)",
                             "(min)", "(max)"):
                    if stat.startswith(word):
                        yield Completion(stat, start_position=-len(word))

            if cmd == "estimates" and len(words) <= 2:
                if "table".startswith(word):
                    yield Completion("table", start_position=-len(word))

            # xtreg estimator completions
            if cmd == "xtreg":
                for sub in ("fe", "re", "be", "--robust", "--cluster="):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # arima order completion
            if cmd == "arima" and word.startswith("order"):
                for sub in ("order(1,0,0)", "order(1,1,0)", "order(1,1,1)", "order(2,1,1)"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # var lags completion
            if cmd == "var" and word.startswith("lags"):
                for sub in ("lags(1)", "lags(2)", "lags(3)", "lags(4)"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # mi sub-commands
            if cmd == "mi" and len(words) <= 2:
                for sub in ("impute", "estimate:", "describe"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # svy: sub-commands
            if cmd == "svy:" and len(words) <= 2:
                for sub in ("summarize", "ols", "logit"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # sts sub-commands
            if cmd == "sts" and len(words) <= 2:
                for sub in ("graph", "test"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # plugin sub-commands
            if cmd == "plugin" and len(words) <= 2:
                for sub in ("list", "info"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # set sub-commands
            if cmd == "set" and len(words) <= 2:
                if "backend".startswith(word):
                    yield Completion("backend", start_position=-len(word))
            if cmd == "set" and len(words) == 3:
                for sub in ("polars", "duckdb"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

            # plot sub-commands expanded
            if cmd == "plot" and len(words) <= 2:
                for sub in ("acf", "pacf"):
                    if sub.startswith(word):
                        yield Completion(sub, start_position=-len(word))

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

    # Expand aliases before parsing
    try:
        from openstat.commands.alias_cmds import resolve_alias
        line = resolve_alias(line)
    except ImportError:
        pass

    parts = line.split(None, 2)
    # Try two-word command first (e.g. "export pdf", "import do")
    if len(parts) >= 2:
        two_word = f"{parts[0].lower()} {parts[1].lower()}"
        if two_word in COMMANDS:
            cmd_name = two_word
            args = parts[2] if len(parts) > 2 else ""
        else:
            cmd_name = parts[0].lower()
            args = " ".join(parts[1:]) if len(parts) > 1 else ""
    else:
        cmd_name = parts[0].lower()
        args = ""

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
        # Write to session log file if active
        if session._log_file is not None:
            try:
                import re as _re
                plain = _re.sub(r"\[/?[^\]]*\]", "", result or "")
                session._log_file.write(f". {line}\n{plain}\n\n")
                session._log_file.flush()
            except Exception:
                pass
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

    # Discover and load plugins
    try:
        from openstat.commands.plugin_cmds import init_plugins
        loaded = init_plugins()
        if loaded:
            console.print(f"[dim]Plugins loaded: {', '.join(loaded)}[/dim]")
    except Exception:
        pass

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

    Supports foreach, forvalues, and if/else control flow.
    If strict=True, stop on first error and raise SystemExit(1).
    """
    if session is None:
        session = Session()

    log.info("Running script: %s (strict=%s)", path, strict)

    from openstat.script_runner import run_script_advanced
    run_script_advanced(path, session, console, _dispatch, strict=strict)
