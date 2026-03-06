"""Bootstrap and permutation test commands."""

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


def _fmt(r: dict) -> str:
    lines = [f"\n{r.get('test', 'Result')}", "=" * 55]
    skip = {"test", "groups", "_model"}
    for k, v in r.items():
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(f"  {k:<35} {v:.6f}")
        elif isinstance(v, list):
            lines.append(f"  {k:<35} {v}")
        else:
            lines.append(f"  {k:<35} {v}")
    lines.append("=" * 55)
    return "\n".join(lines)


@command("bootstrap", usage="bootstrap var [by(groupvar)] [stat(mean)] [n(2000)] [ci(0.95)]")
def cmd_bootstrap(session: Session, args: str) -> str:
    """Bootstrap confidence interval. With by(): tests difference between groups."""
    from openstat.stats.resampling import bootstrap_ci, bootstrap_diff
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: bootstrap var [by(group)] [stat(mean)] [n(2000)] [ci(0.95)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    by = opts.get("by")
    stat = opts.get("stat", "mean")
    n_boot = int(opts.get("n", 2000))
    ci = float(opts.get("ci", 0.95))
    try:
        if by:
            if by not in df.columns:
                return f"Group column '{by}' not found."
            r = bootstrap_diff(df, col, by, stat=stat, n_boot=n_boot, ci=ci)
        else:
            r = bootstrap_ci(df, col, stat=stat, n_boot=n_boot, ci=ci)
        return _fmt(r)
    except Exception as exc:
        return f"bootstrap error: {exc}"


@command("permtest", usage="permtest var by(groupvar) [stat(mean)] [n(2000)] [--greater|--less]")
def cmd_permtest(session: Session, args: str) -> str:
    """Permutation test for difference between two groups."""
    from openstat.stats.resampling import permutation_test
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: permtest var by(groupvar) [stat(mean)] [n(2000)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    by = opts.get("by")
    if not by:
        return "Specify group variable: permtest var by(groupvar)"
    if by not in df.columns:
        return f"Group column '{by}' not found."
    stat = opts.get("stat", "mean")
    n_perm = int(opts.get("n", 2000))
    alt = "two-sided"
    if "--greater" in args:
        alt = "greater"
    elif "--less" in args:
        alt = "less"
    try:
        r = permutation_test(df, col, by, stat=stat, n_perm=n_perm, alternative=alt)
        return _fmt(r)
    except Exception as exc:
        return f"permtest error: {exc}"


@command("jackknife", usage="jackknife var [stat(mean)]")
def cmd_jackknife(session: Session, args: str) -> str:
    """Jackknife bias and standard error estimation."""
    from openstat.stats.resampling import jackknife_ci
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: jackknife var [stat(mean)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    stat = opts.get("stat", "mean")
    try:
        r = jackknife_ci(df, col, stat=stat)
        return _fmt(r)
    except Exception as exc:
        return f"jackknife error: {exc}"
