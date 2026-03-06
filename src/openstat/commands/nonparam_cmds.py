"""Nonparametric test commands: ranksum, signrank, kwallis, spearman."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session
from openstat.stats.nonparametric import (
    spearman_corr,
    ranksum_test,
    signrank_test,
    kruskal_wallis_test,
)


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


def _fmt(d: dict) -> str:
    lines = [f"\n{d.get('test', 'Result')}", "-" * 55]
    skip = {"test", "groups", "n_per_group", "_model"}
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:.6f}")
        elif isinstance(v, list):
            lines.append(f"  {k:<30} {v}")
        else:
            lines.append(f"  {k:<30} {v}")
    lines.append("-" * 55)
    return "\n".join(lines)


@command("ranksum", usage="ranksum var by(groupvar) [--less|--greater]")
def cmd_ranksum(session: Session, args: str) -> str:
    """Wilcoxon rank-sum (Mann-Whitney U) test for two independent groups."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: ranksum var by(groupvar)"

    var = positional[0]
    by = opts.get("by")
    if by is None:
        return "Specify group variable: ranksum var by(groupvar)"

    alt = "two-sided"
    if "--less" in args:
        alt = "less"
    elif "--greater" in args:
        alt = "greater"

    try:
        r = ranksum_test(df, var, by, alternative=alt)
        return _fmt(r)
    except Exception as exc:
        return f"ranksum error: {exc}"


@command("signrank", usage="signrank var1 [var2] [mu(0)]")
def cmd_signrank(session: Session, args: str) -> str:
    """Wilcoxon signed-rank test (one-sample or paired)."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: signrank var1 [var2] [mu(0)]"

    mu = float(opts.get("mu", 0.0))
    var1 = positional[0]
    var2 = positional[1] if len(positional) > 1 and positional[1] in df.columns else None

    try:
        r = signrank_test(df, var1, var2, mu=mu)
        return _fmt(r)
    except Exception as exc:
        return f"signrank error: {exc}"


@command("kwallis", usage="kwallis var by(groupvar)")
def cmd_kwallis(session: Session, args: str) -> str:
    """Kruskal-Wallis H test for k independent groups."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: kwallis var by(groupvar)"

    var = positional[0]
    by = opts.get("by")
    if by is None:
        return "Specify group variable: kwallis var by(groupvar)"

    try:
        r = kruskal_wallis_test(df, var, by)
        lines = [_fmt(r)]
        lines.append("\nGroup counts:")
        for g, n in zip(r["groups"], r["n_per_group"]):
            lines.append(f"  {g!s:<20} n = {n}")
        return "\n".join(lines)
    except Exception as exc:
        return f"kwallis error: {exc}"


@command("spearman", usage="spearman var1 var2 [var3 ...]")
def cmd_spearman(session: Session, args: str) -> str:
    """Spearman rank correlation matrix."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "spearman requires at least 2 numeric variables."

    try:
        r = spearman_corr(df, cols)
    except Exception as exc:
        return f"spearman error: {exc}"

    rho = r["rho"]
    pvals = r["pvalues"]
    k = len(cols)
    w = max(len(c) for c in cols) + 2

    lines = ["\nSpearman Rank Correlation", "=" * (w + k * 9 + 2)]
    header = " " * w + "".join(f"  {c[:7]:>7}" for c in cols)
    lines.append(header)
    lines.append("-" * (w + k * 9 + 2))
    for i, ci in enumerate(cols):
        row = f"{ci:<{w}}"
        for j in range(k):
            row += f"  {rho[i][j]:>7.4f}"
        lines.append(row)
    lines.append("")
    lines.append("P-values:")
    for i, ci in enumerate(cols):
        row = f"{ci:<{w}}"
        for j in range(k):
            if i == j:
                row += f"  {'  .':>7}"
            else:
                row += f"  {pvals[i][j]:>7.4f}"
        lines.append(row)
    return "\n".join(lines)
