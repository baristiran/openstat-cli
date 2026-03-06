"""Regex column commands: regex extract/replace/match/split."""

from __future__ import annotations

import re as _re

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("regex", usage="regex extract|replace|match|split <col> <pattern> [options]")
def cmd_regex(session: Session, args: str) -> str:
    """Apply regex operations to a string column.

    Sub-commands:
      regex extract <col> <pattern> [into(<newcol>)]
          — extract first capture group; stores in new column
      regex replace <col> <pattern> <replacement> [into(<newcol>)]
          — replace all matches with replacement string
      regex match <col> <pattern> [into(<newcol>)]
          — add boolean column: 1 if row matches, 0 otherwise
      regex split <col> <pattern> [into(<newcol>)]
          — split on pattern, store list as string repr

    Examples:
      regex extract email "([^@]+)@" into(username)
      regex replace phone "[^0-9]" "" into(phone_clean)
      regex match address "\\bStreet\\b" into(is_street)
      regex split tags "," into(tag_list)
    """
    import polars as pl

    ca = CommandArgs(args)
    if len(ca.positional) < 3:
        return "Usage: regex extract|replace|match|split <col> <pattern> ..."

    subcmd = ca.positional[0].lower()
    col_name = ca.positional[1]
    pattern = ca.positional[2]

    try:
        df = session.require_data()
        if col_name not in df.columns:
            return f"Column not found: {col_name}"

        # Validate regex
        try:
            compiled = _re.compile(pattern)
        except _re.error as exc:
            return f"Invalid regex: {exc}"

        into_raw = ca.rest_after("into")
        new_col = into_raw.strip().strip("()") if into_raw else None

        if subcmd == "extract":
            new_col = new_col or f"{col_name}_extracted"
            vals = df[col_name].cast(pl.Utf8).to_list()
            results = []
            for v in vals:
                if v is None:
                    results.append(None)
                    continue
                m = compiled.search(v)
                if m:
                    results.append(m.group(1) if m.lastindex else m.group(0))
                else:
                    results.append(None)
            session.snapshot()
            session.df = df.with_columns(pl.Series(new_col, results))
            n_matched = sum(1 for r in results if r is not None)
            return f"Extracted to '{new_col}': {n_matched}/{df.height} rows matched."

        elif subcmd == "replace":
            repl = ca.positional[3] if len(ca.positional) > 3 else ""
            new_col = new_col or col_name
            vals = df[col_name].cast(pl.Utf8).to_list()
            results = [compiled.sub(repl, v) if v is not None else None for v in vals]
            session.snapshot()
            session.df = df.with_columns(pl.Series(new_col, results))
            n_changed = sum(1 for orig, new in zip(vals, results) if orig != new)
            return f"Replaced in '{new_col}': {n_changed}/{df.height} rows changed."

        elif subcmd == "match":
            new_col = new_col or f"{col_name}_match"
            vals = df[col_name].cast(pl.Utf8).to_list()
            results = [1 if (v is not None and compiled.search(v)) else 0 for v in vals]
            session.snapshot()
            session.df = df.with_columns(pl.Series(new_col, results, dtype=pl.Int8))
            n_match = sum(results)
            return f"Match column '{new_col}': {n_match}/{df.height} rows matched."

        elif subcmd == "split":
            new_col = new_col or f"{col_name}_split"
            vals = df[col_name].cast(pl.Utf8).to_list()
            results = [str(compiled.split(v)) if v is not None else None for v in vals]
            session.snapshot()
            session.df = df.with_columns(pl.Series(new_col, results))
            return f"Split column '{new_col}' created."

        else:
            return f"Unknown sub-command: {subcmd}. Use extract, replace, match, or split."

    except Exception as e:
        return friendly_error(e, "regex")
