"""i18n command: locale get|set <code>."""

from __future__ import annotations

from openstat.commands.base import command
from openstat.session import Session


@command("locale", usage="locale [get | set <code>]")
def cmd_locale(session: Session, args: str) -> str:
    """Get or set the display language.

    Examples:
      locale            — show current locale
      locale get        — show current locale
      locale set tr     — switch to Turkish
      locale set en     — switch to English
    """
    from openstat.i18n import get_locale, set_locale, _STRINGS

    tokens = args.strip().split()
    subcmd = tokens[0].lower() if tokens else "get"

    if subcmd in ("get", ""):
        return f"Current locale: {get_locale()}"

    elif subcmd == "set":
        if len(tokens) < 2:
            available = ", ".join(sorted(_STRINGS))
            return f"Usage: locale set <code>   Available: {available}"
        code = tokens[1].lower()
        try:
            set_locale(code)
            return f"Locale set to: {code}"
        except ValueError as exc:
            return str(exc)

    elif subcmd == "list":
        available = ", ".join(sorted(_STRINGS))
        return f"Available locales: {available}"

    else:
        return "Usage: locale [get | set <code> | list]"
