"""Alias and theme commands."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, get_registry, friendly_error
from openstat.session import Session

# Module-level alias store
_ALIASES: dict[str, str] = {}


def get_aliases() -> dict[str, str]:
    return _ALIASES


def resolve_alias(line: str) -> str:
    """If line starts with a known alias, expand it."""
    token = line.split()[0] if line.split() else ""
    if token in _ALIASES:
        rest = line[len(token):].strip()
        expanded = _ALIASES[token]
        return f"{expanded} {rest}".strip() if rest else expanded
    return line


@command("alias", usage="alias [<name> <expansion>] | alias list | alias rm <name>")
def cmd_alias(session: Session, args: str) -> str:
    """Define or manage command aliases.

    Examples:
      alias reg ols          — 'reg y x' → 'ols y x'
      alias desc describe
      alias list             — show all aliases
      alias rm reg           — remove alias
    """
    tokens = args.strip().split(None, 2)

    if not tokens or tokens[0] == "list":
        if not _ALIASES:
            return "No aliases defined. Use: alias <name> <expansion>"
        lines = ["Aliases:", "  {:<15} {}".format("Name", "Expansion"), "-" * 40]
        for k, v in sorted(_ALIASES.items()):
            lines.append(f"  {k:<15} {v}")
        return "\n".join(lines)

    if tokens[0] == "rm":
        if len(tokens) < 2:
            return "Usage: alias rm <name>"
        name = tokens[1]
        if name in _ALIASES:
            del _ALIASES[name]
            return f"Alias '{name}' removed."
        return f"Alias '{name}' not found."

    if len(tokens) < 2:
        return "Usage: alias <name> <expansion>"

    name = tokens[0]
    expansion = " ".join(tokens[1:])

    # Prevent aliasing built-in if it would shadow itself
    if name == expansion.split()[0]:
        return f"Cannot alias '{name}' to itself."

    _ALIASES[name] = expansion
    return f"Alias set: {name} → {expansion}"


# ── theme ─────────────────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "prompt": "bold cyan",
        "output": "white",
        "error": "bold red",
        "info": "bright_blue",
    },
    "light": {
        "prompt": "bold blue",
        "output": "black",
        "error": "bold red",
        "info": "blue",
    },
    "solarized": {
        "prompt": "bold yellow",
        "output": "bright_white",
        "error": "bold magenta",
        "info": "cyan",
    },
    "matrix": {
        "prompt": "bold green",
        "output": "green",
        "error": "bold red",
        "info": "bright_green",
    },
}

_ACTIVE_THEME: str = "dark"


def get_active_theme() -> dict[str, str]:
    return _THEMES.get(_ACTIVE_THEME, _THEMES["dark"])


@command("theme", usage="theme [<name>] | theme list")
def cmd_theme(session: Session, args: str) -> str:
    """Set or list color themes.

    Available themes: dark (default), light, solarized, matrix.

    Examples:
      theme            — show current theme
      theme list       — list all themes
      theme solarized  — switch to solarized
    """
    tokens = args.strip().split()

    if not tokens:
        return f"Current theme: {_ACTIVE_THEME}"

    if tokens[0] == "list":
        lines = ["Available themes:"]
        for name in _THEMES:
            marker = " (active)" if name == _ACTIVE_THEME else ""
            lines.append(f"  {name}{marker}")
        return "\n".join(lines)

    name = tokens[0].lower()
    if name not in _THEMES:
        available = ", ".join(_THEMES)
        return f"Unknown theme: {name}. Available: {available}"

    import openstat.commands.alias_cmds as _self
    _self._ACTIVE_THEME = name
    return f"Theme set to: {name}"
