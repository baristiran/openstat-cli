"""String and date manipulation commands."""

from __future__ import annotations

import re

import polars as pl

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("split", usage="split varname [, sep(,) gen(newvar)]")
def cmd_split(session: Session, args: str) -> str:
    """Split a string column into multiple columns."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: split varname [sep(,) gen(prefix)]"
    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."
    sep = opts.get("sep", " ")
    prefix = opts.get("gen", var)
    session.snapshot()
    try:
        parts = df[var].str.split(sep)
        max_parts = max(len(p) for p in parts.to_list())
        new_df = df
        for i in range(max_parts):
            col_name = f"{prefix}{i+1}"
            new_df = new_df.with_columns(
                parts.list.get(i, null_on_oob=True).alias(col_name)
            )
        session.df = new_df
        return f"Split '{var}' into {max_parts} columns: {[f'{prefix}{i+1}' for i in range(max_parts)]}"
    except Exception as exc:
        return f"split error: {exc}"


@command("strtrim", usage="strtrim varname [gen(newvar)]")
def cmd_strtrim(session: Session, args: str) -> str:
    """Trim whitespace from a string column."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: strtrim varname [gen(newvar)]"
    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."
    new_var = opts.get("gen", var)
    session.snapshot()
    try:
        session.df = df.with_columns(pl.col(var).str.strip_chars().alias(new_var))
        return f"Trimmed '{var}' → '{new_var}'"
    except Exception as exc:
        return f"strtrim error: {exc}"


@command("strupper", usage="strupper varname [gen(newvar)]")
def cmd_strupper(session: Session, args: str) -> str:
    """Convert string column to uppercase."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: strupper varname [gen(newvar)]"
    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."
    new_var = opts.get("gen", var)
    session.snapshot()
    try:
        session.df = df.with_columns(pl.col(var).str.to_uppercase().alias(new_var))
        return f"Uppercased '{var}' → '{new_var}'"
    except Exception as exc:
        return f"strupper error: {exc}"


@command("strlower", usage="strlower varname [gen(newvar)]")
def cmd_strlower(session: Session, args: str) -> str:
    """Convert string column to lowercase."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: strlower varname [gen(newvar)]"
    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."
    new_var = opts.get("gen", var)
    session.snapshot()
    try:
        session.df = df.with_columns(pl.col(var).str.to_lowercase().alias(new_var))
        return f"Lowercased '{var}' → '{new_var}'"
    except Exception as exc:
        return f"strlower error: {exc}"


@command("strreplace", usage="strreplace varname old new [gen(newvar)]")
def cmd_strreplace(session: Session, args: str) -> str:
    """Replace substring in a string column."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 3:
        return "Usage: strreplace varname old new [gen(newvar)]"
    var, old_str, new_str = positional[0], positional[1], positional[2]
    if var not in df.columns:
        return f"Column '{var}' not found."
    new_var = opts.get("gen", var)
    session.snapshot()
    try:
        session.df = df.with_columns(pl.col(var).str.replace_all(old_str, new_str).alias(new_var))
        return f"Replaced '{old_str}' with '{new_str}' in '{var}' → '{new_var}'"
    except Exception as exc:
        return f"strreplace error: {exc}"

