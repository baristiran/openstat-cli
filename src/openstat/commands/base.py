"""Command registration infrastructure.

Provides a @command decorator that auto-registers handler functions,
and a CommandArgs helper for standardized argument parsing.

Usage:
    from openstat.commands.base import command

    @command("mycommand", usage="mycommand <arg>")
    def cmd_mycommand(session, args):
        '''One-line description shown in help.'''
        ...
        return "result text"
"""

from __future__ import annotations

import io
import re
from typing import Callable

from rich.console import Console

from openstat.session import Session
from openstat.logging_config import get_logger

log = get_logger("commands")

# Type alias for command handlers
Handler = Callable[[Session, str], str]

# Global registry — populated by @command decorator
_REGISTRY: dict[str, Handler] = {}
_USAGE: dict[str, str] = {}


class CommandArgs:
    """Standardized argument parser for commands.

    Handles: positional args, --flags, key=value options.

    Usage:
        ca = CommandArgs(args)
        ca.positional    # list of positional tokens
        ca.has_flag("--robust")  # True/False
        ca.get_option("how", "inner")  # key=value with default
        ca.rest_after("on")  # everything after keyword "on"
    """

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self._tokens = raw.split()
        self.flags: set[str] = set()
        self.options: dict[str, str] = {}
        self.positional: list[str] = []

        for tok in self._tokens:
            if tok.startswith("--"):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    self.options[k.lstrip("-")] = v
                else:
                    self.flags.add(tok)
            elif "=" in tok and not tok.startswith('"') and not tok.startswith("'"):
                k, v = tok.split("=", 1)
                self.options[k] = v
            else:
                self.positional.append(tok)

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def get_option(self, key: str, default: str | None = None) -> str | None:
        return self.options.get(key, default)

    def get_option_float(self, key: str, default: float) -> float:
        val = self.options.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def rest_after(self, keyword: str) -> str | None:
        """Return everything after a keyword (case-insensitive)."""
        parts = re.split(rf"\b{keyword}\b", self.raw, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) < 2:
            return None
        return parts[1].strip()

    def strip_flags_and_options(self) -> str:
        """Return raw string with all --flags and key=value removed."""
        result = self.raw
        for flag in self.flags:
            result = result.replace(flag, "")
        for k, v in self.options.items():
            result = result.replace(f"--{k}={v}", "")
            result = result.replace(f"{k}={v}", "")
        return result.strip()

    def __bool__(self) -> bool:
        return bool(self.raw.strip())


def command(name: str, *, usage: str = "") -> Callable[[Handler], Handler]:
    """Decorator to register a command handler.

    Args:
        name:  Command name as typed by the user.
        usage: One-line usage example shown in help.
    """
    def decorator(fn: Handler) -> Handler:
        if name in _REGISTRY:
            log.warning("Command '%s' re-registered (overriding previous)", name)
        _REGISTRY[name] = fn
        _USAGE[name] = usage or f"{name} ..."
        return fn
    return decorator


def get_registry() -> dict[str, Handler]:
    """Return a live read-only view of the command registry.

    The returned mapping always reflects the current state of the
    registry, so commands registered after import time (e.g. plugins)
    are visible automatically.
    """
    from types import MappingProxyType
    return MappingProxyType(_REGISTRY)  # type: ignore[return-value]


def get_usage(name: str) -> str:
    return _USAGE.get(name, "")


def rich_to_str(fn) -> str:
    """Capture Rich output as plain text (no stdout side-effect)."""
    buf = io.StringIO()
    console = Console(file=buf, width=120, record=True)
    fn(console)
    return console.export_text().rstrip()


def friendly_error(e: Exception, context: str) -> str:
    """Convert common Polars/statsmodels errors to user-friendly messages."""
    msg = str(e)
    etype = type(e).__name__
    if "not found" in msg.lower() or "ColumnNotFoundError" in etype:
        return f"[red]Error:[/red] {context}: Column not found. Check column names with 'describe'."
    if "type" in msg.lower() and ("str" in msg.lower() or "string" in msg.lower()):
        return f"[red]Error:[/red] {context}: Type mismatch — cannot use arithmetic on text columns."
    if "singular" in msg.lower() or "linalg" in msg.lower():
        return f"[red]Error:[/red] {context}: Matrix is singular — check for perfect multicollinearity or constant columns."
    log.debug("Unhandled error in %s: %s: %s", context, etype, msg)
    return f"[red]Error:[/red] {context}: {e}"
