"""Epidemiology commands: cs (cohort study), cc (case-control), ir (incidence rate)."""

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


def _fmt_epi(r: dict) -> str:
    lines = [f"\n{r.get('test', 'Result')}", "=" * 50]
    skip = {"test", "table_2x2", "_model"}
    for k, v in r.items():
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:.4f}")
        else:
            lines.append(f"  {k:<30} {v}")
    t = r.get("table_2x2")
    if t:
        lines.append("\n  2x2 Table:")
        lines.append(f"  {'':15} Exposed  Unexposed")
        lines.append(f"  {'Cases':15} {t['a']:>8}  {t['b']:>8}")
        lines.append(f"  {'Non-cases':15} {t['c']:>8}  {t['d']:>8}")
    return "\n".join(lines)


@command("cs", usage="cs outcome exposure")
def cmd_cs(session: Session, args: str) -> str:
    """Cohort study analysis: risk ratio, ARR, NNT."""
    from openstat.stats.epidemiology import cohort_study
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: cs outcome exposure"
    outcome, exposure = positional[0], positional[1]
    for v in (outcome, exposure):
        if v not in df.columns:
            return f"Column '{v}' not found."
    try:
        r = cohort_study(df, outcome, exposure)
        return _fmt_epi(r)
    except Exception as exc:
        return f"cs error: {exc}"


@command("cc", usage="cc outcome exposure")
def cmd_cc(session: Session, args: str) -> str:
    """Case-control analysis: odds ratio with 95% CI."""
    from openstat.stats.epidemiology import case_control
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: cc outcome exposure"
    outcome, exposure = positional[0], positional[1]
    for v in (outcome, exposure):
        if v not in df.columns:
            return f"Column '{v}' not found."
    try:
        r = case_control(df, outcome, exposure)
        return _fmt_epi(r)
    except Exception as exc:
        return f"cc error: {exc}"


@command("ir", usage="ir outcome person_time_var")
def cmd_ir(session: Session, args: str) -> str:
    """Incidence rate analysis."""
    from openstat.stats.epidemiology import incidence_rate
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: ir outcome person_time_var"
    outcome, pt_var = positional[0], positional[1]
    for v in (outcome, pt_var):
        if v not in df.columns:
            return f"Column '{v}' not found."
    try:
        r = incidence_rate(df, outcome, pt_var)
        return _fmt_epi(r)
    except Exception as exc:
        return f"ir error: {exc}"
