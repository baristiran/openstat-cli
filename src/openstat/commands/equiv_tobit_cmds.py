"""Equivalence test and Tobit commands."""

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


@command("tost", usage="tost var [by(group)] [mu(0) delta(0.5) alpha(0.05)]")
def cmd_tost(session: Session, args: str) -> str:
    """Two One-Sided Tests (TOST) for equivalence."""
    from openstat.stats.equiv_tobit import tost_onemean, tost_twomeans
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: tost var [by(group)] [mu(0) delta(0.5) alpha(0.05)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    by = opts.get("by")
    delta = float(opts.get("delta", 0.5))
    alpha = float(opts.get("alpha", 0.05))
    try:
        if by:
            if by not in df.columns:
                return f"Group column '{by}' not found."
            r = tost_twomeans(df, col, by, delta=delta, alpha=alpha)
        else:
            mu = float(opts.get("mu", 0.0))
            r = tost_onemean(df, col, mu=mu, delta=delta, alpha=alpha)
        return _fmt(r)
    except Exception as exc:
        return f"tost error: {exc}"


@command("tobit", usage="tobit dep var1 var2 ... [ll(0) ul(none)]")
def cmd_tobit(session: Session, args: str) -> str:
    """Tobit regression for censored outcomes."""
    from openstat.stats.equiv_tobit import fit_tobit
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: tobit dep var1 [var2 ...] [ll(0) ul(none)]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    if dep not in df.columns:
        return f"Column '{dep}' not found."
    left = float(opts["ll"]) if "ll" in opts else 0.0
    right = float(opts["ul"]) if "ul" in opts else None
    try:
        r = fit_tobit(df, dep, indeps, left=left, right=right)
        session._last_model = r
        lines = ["\nTobit Regression", "=" * 55]
        lines.append(f"  {'Dep. Variable':<30} {dep}")
        lines.append(f"  {'N obs':<30} {r['n_obs']}")
        lines.append(f"  {'Left censoring':<30} {r['left_censoring']} (n={r['n_censored_left']})")
        lines.append(f"  {'Right censoring':<30} {r['right_censoring']} (n={r['n_censored_right']})")
        lines.append(f"  {'Log-likelihood':<30} {r['log_likelihood']:.4f}")
        lines.append(f"  {'AIC':<30} {r['aic']:.4f}")
        lines.append(f"  {'Sigma':<30} {r['sigma']:.4f}")
        lines.append(f"\n  {'Variable':<25} {'Coef':>10}")
        lines.append("  " + "-" * 37)
        for nm, coef in r["params"].items():
            lines.append(f"  {nm:<25} {coef:>10.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"tobit error: {exc}"
