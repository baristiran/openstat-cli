"""DSL commands: local, global, forval, foreach, assert, display."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("local", usage="local name value")
def cmd_local(session: Session, args: str) -> str:
    """Define a local macro variable."""
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        return "Usage: local name value"
    name, value = parts[0], parts[1]
    if not hasattr(session, "_locals"):
        session._locals = {}
    session._locals[name] = value
    return f"local `{name}' = {value}"


@command("global", usage="global name value")
def cmd_global(session: Session, args: str) -> str:
    """Define a global macro variable."""
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        return "Usage: global name value"
    name, value = parts[0], parts[1]
    if not hasattr(session, "_globals"):
        session._globals = {}
    session._globals[name] = value
    return f"global ${name} = {value}"


@command("display", usage="display expression_or_text")
def cmd_display(session: Session, args: str) -> str:
    """Display text or evaluate a simple numeric expression."""
    text = args.strip().strip('"').strip("'")
    # Substitute local macros `name'
    if hasattr(session, "_locals"):
        for k, v in session._locals.items():
            text = text.replace(f"`{k}'", v)
    # Substitute global macros $name
    if hasattr(session, "_globals"):
        for k, v in session._globals.items():
            text = text.replace(f"${k}", v)
    # Try simple arithmetic evaluation
    try:
        result = eval(text, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception:
        return text


@command("assert", usage="assert condition_description [var op value]")
def cmd_assert(session: Session, args: str) -> str:
    """Assert that a condition holds in the data. Returns pass/fail."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    # Simple form: assert var op value (e.g., assert age > 0)
    # Use polars expression
    expr_str = args.strip()
    # Try to parse var op value
    m = re.match(r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)', expr_str)
    if not m:
        return f"assert syntax: varname op value (e.g., assert age > 0)"
    var, op, val_str = m.group(1), m.group(2), m.group(3).strip()
    if var not in df.columns:
        return f"Column '{var}' not found."
    try:
        val = float(val_str)
        import polars as pl
        ops = {"==": pl.col(var) == val, "!=": pl.col(var) != val,
               ">": pl.col(var) > val, "<": pl.col(var) < val,
               ">=": pl.col(var) >= val, "<=": pl.col(var) <= val}
        mask = ops[op]
        n_fail = int(df.filter(~mask).height)
        if n_fail == 0:
            return f"Assertion passed: {var} {op} {val} holds for all {df.height} observations."
        else:
            return f"Assertion FAILED: {n_fail} of {df.height} observations violate {var} {op} {val}."
    except Exception as exc:
        return f"assert error: {exc}"


@command("forval", usage="forval i=start/end : command args")
def cmd_forval(session: Session, args: str) -> str:
    """Execute a command for each value in a range. forval i=1/5 : display `i'"""
    m = re.match(r'(\w+)\s*=\s*(\d+)\s*/\s*(\d+)\s*:\s*(.+)', args.strip())
    if not m:
        return "Usage: forval i=start/end : command args"
    var, start, end, cmd_str = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).strip()
    from openstat.commands.base import run_command
    outputs = []
    for i in range(start, end + 1):
        if not hasattr(session, "_locals"):
            session._locals = {}
        session._locals[var] = str(i)
        expanded = cmd_str.replace(f"`{var}'", str(i))
        try:
            out = run_command(session, expanded)
            if out:
                outputs.append(out)
        except Exception as exc:
            outputs.append(f"forval error at i={i}: {exc}")
            break
    return "\n".join(outputs) if outputs else f"forval completed {end - start + 1} iterations."


@command("foreach", usage="foreach var in list : command")
def cmd_foreach(session: Session, args: str) -> str:
    """Execute a command for each item in a list."""
    m = re.match(r'(\w+)\s+in\s+(.+?)\s*:\s*(.+)', args.strip())
    if not m:
        return "Usage: foreach var in item1 item2 ... : command"
    var, items_str, cmd_str = m.group(1), m.group(2).strip(), m.group(3).strip()
    items = items_str.split()
    from openstat.commands.base import run_command
    outputs = []
    for item in items:
        if not hasattr(session, "_locals"):
            session._locals = {}
        session._locals[var] = item
        expanded = cmd_str.replace(f"`{var}'", item)
        try:
            out = run_command(session, expanded)
            if out:
                outputs.append(out)
        except Exception as exc:
            outputs.append(f"foreach error at {var}={item}: {exc}")
            break
    return "\n".join(outputs) if outputs else f"foreach completed {len(items)} iterations."
