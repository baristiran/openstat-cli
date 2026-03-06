"""Advanced .ost script runner with foreach, forvalues, and if/else support."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openstat.session import Session


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------

def _collect_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Starting *after* the opening '{', collect lines until the matching '}'.

    Returns (body_lines, next_index_after_closing_brace).

    When a line starts with '}' at depth==1, we consider the block closed even
    if more text follows on that line (e.g., '} else {').  The caller is
    responsible for examining the remainder of that closing line.
    """
    body: list[str] = []
    depth = 1
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        # A line starting with '}' at the current top level closes this block
        if stripped.startswith("}"):
            # Account for nested close + new open on same line: '} else {'
            # We stop here; return index pointing at this same line so the
            # caller can inspect it for 'else'.
            return body, i
        for ch in stripped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        body.append(lines[i])
        i += 1
    return body, i  # unterminated block — caller handles


def _parse_statements(lines: list[str]) -> list:
    """Parse lines into a flat list of statement objects.

    Each statement is one of:
      ("line",   text)
      ("foreach", varname, values_list, body_lines)
      ("forvalues", varname, num_sequence, body_lines)
      ("if",     condition, if_body, else_body_or_None)
    """
    statements: list = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # ---- foreach var in val1 val2 ... {
        m = re.match(r'^foreach\s+(\w+)\s+in\s+(.+?)(?:\s*\{)?\s*$', stripped)
        if m:
            varname = m.group(1)
            values_raw = m.group(2).rstrip("{").strip()
            values = values_raw.split()
            body, i = _collect_block(lines, i + 1)
            # advance past the closing '}'
            if i < len(lines) and lines[i].strip().startswith("}"):
                i += 1
            statements.append(("foreach", varname, values, body))
            continue

        # ---- forvalues var = start/end  or  start(step)end
        m = re.match(r'^forvalues\s+(\w+)\s*=\s*(.+?)(?:\s*\{)?\s*$', stripped)
        if m:
            varname = m.group(1)
            seq_raw = m.group(2).rstrip("{").strip()
            seq = _parse_numseq(seq_raw)
            body, i = _collect_block(lines, i + 1)
            if i < len(lines) and lines[i].strip().startswith("}"):
                i += 1
            statements.append(("forvalues", varname, seq, body))
            continue

        # ---- if condition {
        m = re.match(r'^if\s+(.+?)(?:\s*\{)?\s*$', stripped)
        if m and not stripped.startswith("if_"):
            condition = m.group(1).rstrip("{").strip()
            if_body, i = _collect_block(lines, i + 1)

            # Check for else on the closing '}' line: '} else {'
            else_body: list[str] | None = None
            if i < len(lines):
                close_line = lines[i].strip()
                # '} else {' or '} else' or standalone '}'
                else_m = re.match(r'^\}\s*else\s*\{?\s*$', close_line)
                if else_m:
                    if close_line.rstrip().endswith("{"):
                        # Body starts on next line
                        else_body, i = _collect_block(lines, i + 1)
                        if i < len(lines) and lines[i].strip().startswith("}"):
                            i += 1
                    else:
                        # '} else' — opening brace on next line
                        i += 1
                        while i < len(lines) and lines[i].strip() in ("", "{"):
                            if lines[i].strip() == "{":
                                i += 1
                                break
                            i += 1
                        else_body, i = _collect_block(lines, i)
                        if i < len(lines) and lines[i].strip().startswith("}"):
                            i += 1
                else:
                    # plain closing '}' — advance past it
                    i += 1

            statements.append(("if", condition, if_body, else_body))
            continue

        # ---- Plain line
        statements.append(("line", stripped))
        i += 1

    return statements


def _parse_numseq(raw: str) -> list[int | float]:
    """Parse forvalues sequence: '1/10', '1(2)10', '1.0(0.5)5.0'."""
    # start(step)end
    m = re.match(r'^([\d.]+)\((-?[\d.]+)\)([\d.]+)$', raw.strip())
    if m:
        start, step, end = float(m.group(1)), float(m.group(2)), float(m.group(3))
        result = []
        v = start
        while (step > 0 and v <= end + 1e-9) or (step < 0 and v >= end - 1e-9):
            result.append(int(v) if v == int(v) else v)
            v += step
        return result

    # start/end
    m = re.match(r'^([\d.]+)/([\d.]+)$', raw.strip())
    if m:
        start, end = float(m.group(1)), float(m.group(2))
        return [int(v) if v == int(v) else v for v in range(int(start), int(end) + 1)]

    # Single value
    try:
        v = float(raw.strip())
        return [int(v) if v == int(v) else v]
    except ValueError:
        return []


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "1", "yes"}


def _eval_condition(condition: str, local_vars: dict[str, str], session: "Session") -> bool:
    """Evaluate a simple if-condition.

    Supports:
      - data_loaded         → True if dataset is loaded
      - col_exists <colname>→ True if column exists in data
      - N > 100             → True if row count > 100
      - {var} == value      → comparison with local variable
    """
    # Substitute local variables
    cond = _substitute(condition, local_vars)

    cond = cond.strip()

    if cond.lower() == "data_loaded":
        return session.df is not None

    m = re.match(r'^col_exists\s+(\S+)$', cond.lower())
    if m:
        col = m.group(1)
        return session.df is not None and col in session.df.columns

    if cond.lower() == "n":
        return session.df is not None and len(session.df) > 0

    # N <op> <number>
    m = re.match(r'^N\s*(>|<|>=|<=|==|!=)\s*([\d.]+)$', cond)
    if m and session.df is not None:
        op, val = m.group(1), float(m.group(2))
        n = len(session.df)
        return {">"  : n > val, "<"  : n < val, ">=" : n >= val,
                "<=" : n <= val, "==" : n == val, "!=" : n != val}[op]

    # Generic comparison: lhs op rhs
    m = re.match(r'^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$', cond)
    if m:
        lhs, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        # Strip quotes from rhs if string
        rhs_s = rhs.strip('"\'')
        lhs_s = lhs.strip('"\'')
        try:
            lhs_v: float | str = float(lhs_s)
            rhs_v: float | str = float(rhs_s)
        except ValueError:
            lhs_v = lhs_s
            rhs_v = rhs_s
        return {"==" : lhs_v == rhs_v, "!=" : lhs_v != rhs_v,
                ">"  : lhs_v > rhs_v,  "<"  : lhs_v < rhs_v,   # type: ignore[operator]
                ">=" : lhs_v >= rhs_v, "<=" : lhs_v <= rhs_v}[op]  # type: ignore[operator]

    return cond.lower() in _BOOL_TRUE


def _substitute(text: str, local_vars: dict[str, str]) -> str:
    """Replace {varname} with its value from local_vars."""
    for k, v in local_vars.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_statements(
    statements: list,
    session: "Session",
    console,
    dispatcher,
    *,
    strict: bool = False,
    local_vars: dict[str, str] | None = None,
) -> bool:
    """Execute a list of parsed statements. Returns True if script should continue."""
    if local_vars is None:
        local_vars = {}

    for stmt in statements:
        kind = stmt[0]

        if kind == "line":
            line = _substitute(stmt[1], local_vars)
            if not line or line.startswith("#"):
                continue
            console.print(f"[dim]>>> {line}[/dim]")
            result = dispatcher(session, line)
            if result == "__QUIT__":
                return False
            if result:
                console.print(result)
                if strict:
                    import re as _re
                    plain = _re.sub(r"\[/?[^\]]*\]", "", result)
                    if plain.startswith(("Error", "Internal error")):
                        raise SystemExit(1)
            console.print()

        elif kind == "foreach":
            _, varname, values, body = stmt
            body_stmts = _parse_statements(body)
            for val in values:
                new_locals = {**local_vars, varname: str(val)}
                cont = execute_statements(
                    body_stmts, session, console, dispatcher,
                    strict=strict, local_vars=new_locals,
                )
                if not cont:
                    return False

        elif kind == "forvalues":
            _, varname, seq, body = stmt
            body_stmts = _parse_statements(body)
            for val in seq:
                new_locals = {**local_vars, varname: str(val)}
                cont = execute_statements(
                    body_stmts, session, console, dispatcher,
                    strict=strict, local_vars=new_locals,
                )
                if not cont:
                    return False

        elif kind == "if":
            _, condition, if_body, else_body = stmt
            if _eval_condition(condition, local_vars, session):
                branch = if_body
            else:
                branch = else_body or []
            branch_stmts = _parse_statements(branch)
            cont = execute_statements(
                branch_stmts, session, console, dispatcher,
                strict=strict, local_vars=local_vars,
            )
            if not cont:
                return False

    return True


def run_script_advanced(
    path: str,
    session: "Session",
    console,
    dispatcher,
    *,
    strict: bool = False,
) -> None:
    """Run an .ost script with foreach/forvalues/if-else support."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    statements = _parse_statements(lines)
    execute_statements(statements, session, console, dispatcher, strict=strict)
