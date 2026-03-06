"""Influence and diagnostics commands: dfbeta, leverage, cooksd, outlier, avplot, coefplot."""

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


def _get_last_ols(session: Session):
    m = session._last_model
    if m is None:
        return None, None, None
    if hasattr(m, "model") and hasattr(m, "dep"):
        return m, m.dep, m.indeps
    if isinstance(m, dict) and "dep" in m and "indeps" in m:
        return m, m["dep"], m["indeps"]
    return None, None, None


@command("dfbeta", usage="dfbeta [dep indeps]")
def cmd_dfbeta(session: Session, args: str) -> str:
    """Compute DFBETAs for last OLS or specified variables."""
    from openstat.stats.influence import compute_influence
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) >= 2:
        dep, indeps = positional[0], positional[1:]
    else:
        _, dep, indeps = _get_last_ols(session)
        if dep is None:
            return "Specify dep and indeps, or fit a model first."
    indeps = [c for c in indeps if c in df.columns]
    if not indeps or dep not in df.columns:
        return "Invalid variables."
    try:
        r = compute_influence(df, dep, indeps)
        lines = ["\nDFBETA Statistics", "-" * 50]
        for name, vals in r["dfbetas"].items():
            import numpy as np
            arr = np.array(vals)
            lines.append(f"  {name:<20} max|DFBETA| = {np.abs(arr).max():.4f}")
        lines.append(f"\n  Threshold (2/sqrt(n)): {2/r['n_obs']**0.5:.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"dfbeta error: {exc}"


@command("leverage", usage="leverage [dep indeps]")
def cmd_leverage(session: Session, args: str) -> str:
    """Show leverage statistics for OLS regression."""
    from openstat.stats.influence import compute_influence
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) >= 2:
        dep, indeps = positional[0], positional[1:]
    else:
        _, dep, indeps = _get_last_ols(session)
        if dep is None:
            return "Specify dep and indeps, or fit a model first."
    indeps = [c for c in indeps if c in df.columns]
    try:
        r = compute_influence(df, dep, indeps)
        lines = ["\nLeverage Statistics", "-" * 50]
        lines.append(f"  {'High leverage threshold':<35} {r['high_leverage_threshold']:.4f}")
        lines.append(f"  {'Observations with high leverage':<35} {r['n_high_leverage']}")
        import numpy as np
        lev = np.array(r["leverage"])
        lines.append(f"  {'Mean leverage':<35} {lev.mean():.4f}")
        lines.append(f"  {'Max leverage':<35} {lev.max():.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"leverage error: {exc}"


@command("cooksd", usage="cooksd [dep indeps]")
def cmd_cooksd(session: Session, args: str) -> str:
    """Compute Cook's distance for influence detection."""
    from openstat.stats.influence import compute_influence
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) >= 2:
        dep, indeps = positional[0], positional[1:]
    else:
        _, dep, indeps = _get_last_ols(session)
        if dep is None:
            return "Specify dep and indeps, or fit a model first."
    indeps = [c for c in indeps if c in df.columns]
    try:
        r = compute_influence(df, dep, indeps)
        lines = ["\nCook's Distance", "-" * 50]
        lines.append(f"  {'Threshold (4/n)':<35} {r['high_cooks_threshold']:.4f}")
        lines.append(f"  {'Influential observations':<35} {r['n_high_cooks']}")
        import numpy as np
        cd = np.array(r["cooks_d"])
        lines.append(f"  {'Max Cook''s D':<35} {cd.max():.4f}")
        if r["n_high_cooks"] > 0:
            top = np.argsort(cd)[::-1][:5]
            lines.append(f"  Top influential obs (index): {top.tolist()}")
        return "\n".join(lines)
    except Exception as exc:
        return f"cooksd error: {exc}"


@command("outlier", usage="outlier dep indeps [threshold(3.0)]")
def cmd_outlier(session: Session, args: str) -> str:
    """Detect outliers by studentized residuals."""
    from openstat.stats.influence import detect_outliers
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: outlier dep indeps [threshold(3.0)]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    threshold = float(opts.get("threshold", 3.0))
    try:
        r = detect_outliers(df, dep, indeps, threshold=threshold)
        lines = [f"\nOutlier Detection (|studentized resid| > {threshold})", "-" * 50]
        lines.append(f"  Outliers found: {r['n_outliers']}")
        if r["outlier_indices"]:
            lines.append(f"  Outlier indices: {r['outlier_indices'][:20]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"outlier error: {exc}"
