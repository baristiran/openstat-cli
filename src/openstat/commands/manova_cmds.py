"""MANOVA and two-way ANOVA commands."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str], set[str]]:
    opts: dict[str, str] = {}
    flags: set[str] = set()
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = []
    for tok in rest.split():
        tok = tok.strip(',')
        if not tok:
            continue
        if tok.startswith('--'):
            flags.add(tok.lstrip('-').lower())
        elif tok:
            positional.append(tok)
    return positional, opts, flags


@command("anova2", usage="anova2 depvar factor1 factor2 [, --nointeraction]")
def cmd_anova2(session: Session, args: str) -> str:
    """Two-way ANOVA with optional interaction term."""
    df = session.require_data()
    positional, opts, flags = _stata_opts(args)
    if len(positional) < 3:
        return "Usage: anova2 depvar factor1 factor2 [, --nointeraction]"

    dep = positional[0]
    f1 = positional[1]
    f2 = positional[2]
    interaction = "nointeraction" not in flags

    try:
        from openstat.stats.manova import twoway_anova
        result = twoway_anova(df, dep, f1, f2, interaction=interaction)
    except Exception as exc:
        return f"anova2 error: {exc}"

    lines = [f"\nTwo-way ANOVA: {dep} ~ {f1} + {f2}", "=" * 70]
    lines.append(
        f"  {'Source':<35}  {'df':>4}  {'SS':>12}  {'MS':>12}  {'F':>8}  {'p-value':>8}"
    )
    lines.append("  " + "-" * 66)
    for row in result["table"]:
        src = row["source"][:35]
        f_str = f"{row['F']:>8.3f}" if not (row["F"] != row["F"]) else "      ."
        p_str = f"{row['p_value']:>8.4f}" if not (row["p_value"] != row["p_value"]) else "      ."
        ss = row["SS"] if row["SS"] == row["SS"] else float("nan")
        ms = row["MS"] if row["MS"] == row["MS"] else float("nan")
        ss_s = f"{ss:>12.4f}" if ss == ss else f"{'':>12}"
        ms_s = f"{ms:>12.4f}" if ms == ms else f"{'':>12}"
        lines.append(f"  {src:<35}  {row['df']:>4}  {ss_s}  {ms_s}  {f_str}  {p_str}")
    lines.append("=" * 70)
    lines.append(f"  R² = {result['r_squared']:.4f}   N = {result['n_obs']}")
    return "\n".join(lines)


@command("manova", usage="manova depvar1 depvar2 ... = groupvar")
def cmd_manova(session: Session, args: str) -> str:
    """One-way MANOVA: test group differences on multiple outcomes."""
    df = session.require_data()
    # parse: "y1 y2 y3 = groupvar"
    if "=" not in args:
        return "Usage: manova depvar1 depvar2 ... = groupvar"

    parts = args.split("=", 1)
    dep_vars = [c.strip() for c in parts[0].split() if c.strip() in df.columns]
    group = parts[1].strip()

    if not dep_vars:
        return "No valid dependent variables found."
    if group not in df.columns:
        return f"Group variable '{group}' not found."

    try:
        from openstat.stats.manova import fit_manova
        result = fit_manova(df, dep_vars, group)
    except Exception as exc:
        return f"manova error: {exc}"

    lines = [
        f"\nMANOVA: {', '.join(dep_vars)} ~ {group}",
        f"  N = {result['n_obs']}, Groups = {result['n_groups']}",
        "=" * 75,
        f"  {'Effect':<20}  {'Test':<20}  {'Stat':>8}  {'F':>8}  {'Num df':>6}  {'Den df':>6}  {'p':>8}",
        "  " + "-" * 71,
    ]
    for eff in result["effects"]:
        p_str = f"{eff['p_value']:>8.4f}" if eff['p_value'] == eff['p_value'] else "      ."
        lines.append(
            f"  {eff['effect'][:20]:<20}  {eff['test'][:20]:<20}"
            f"  {eff['statistic']:>8.4f}  {eff['F']:>8.3f}"
            f"  {eff['num_df']:>6.1f}  {eff['den_df']:>6.1f}  {p_str}"
        )
    lines.append("=" * 75)
    return "\n".join(lines)
