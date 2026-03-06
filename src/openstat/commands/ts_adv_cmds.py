"""Advanced time-series commands: granger, johansen, vecm, stl, tssmooth."""

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


@command("granger", usage="granger dep cause [maxlag(4)]")
def cmd_granger(session: Session, args: str) -> str:
    """Granger causality test."""
    from openstat.stats.ts_advanced import granger_causality
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: granger dep cause [maxlag(4)]"
    dep, cause = positional[0], positional[1]
    maxlag = int(opts.get("maxlag", 4))
    try:
        r = granger_causality(df, dep, cause, maxlag=maxlag)
        lines = [f"\nGranger Causality: {cause} → {dep}", "-" * 50]
        for lag, pval in r["lag_pvalues"].items():
            lines.append(f"  Lag {lag:2d}: F-test p-value = {pval:.4f}")
        lines.append(f"\n  Min p-value: {r['min_pvalue']:.4f} at lag {r['best_lag']}")
        lines.append(f"  Granger-causes at 5%: {'YES' if r['reject_null_5pct'] else 'NO'}")
        return "\n".join(lines)
    except Exception as exc:
        return f"granger error: {exc}"


@command("johansen", usage="johansen var1 var2 [var3 ...] [lags(1)]")
def cmd_johansen(session: Session, args: str) -> str:
    """Johansen cointegration test."""
    from openstat.stats.ts_advanced import johansen_test
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "johansen requires at least 2 variables."
    k_ar_diff = int(opts.get("lags", 1))
    try:
        r = johansen_test(df, cols, k_ar_diff=k_ar_diff)
        lines = ["\nJohansen Cointegration Test", "=" * 55]
        lines.append(f"  Variables: {', '.join(cols)}")
        lines.append(f"  Cointegrating vectors: {r['n_cointegrating_vectors']}")
        lines.append("\n  Trace Statistics:")
        lines.append(f"  {'r=0':>10} {'Statistic':>12} {'CV 95%':>10} {'CV 90%':>10}")
        for i, (ts, cv95, cv90) in enumerate(zip(r["trace_statistics"], r["trace_cv_95"], r["trace_cv_90"])):
            lines.append(f"  r<={i:<8} {ts:>12.4f} {cv95:>10.4f} {cv90:>10.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"johansen error: {exc}"


@command("vecm", usage="vecm var1 var2 [var3 ...] [lags(1) rank(1)]")
def cmd_vecm(session: Session, args: str) -> str:
    """Vector Error Correction Model."""
    from openstat.stats.ts_advanced import fit_vecm
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "vecm requires at least 2 variables."
    k_ar_diff = int(opts.get("lags", 1))
    coint_rank = int(opts.get("rank", 1))
    try:
        r = fit_vecm(df, cols, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
        session._last_model = r
        lines = ["\nVECM Results", "=" * 50]
        lines.append(f"  Variables: {', '.join(cols)}")
        lines.append(f"  Cointegration rank: {coint_rank}, AR lags: {k_ar_diff}")
        lines.append(f"\n  Alpha (adjustment coefficients):")
        for row in r["alpha"]:
            lines.append(f"    {row}")
        return "\n".join(lines)
    except Exception as exc:
        return f"vecm error: {exc}"


@command("stl", usage="stl varname [period(12)]")
def cmd_stl(session: Session, args: str) -> str:
    """STL decomposition (trend + seasonal + residual)."""
    from openstat.stats.ts_advanced import stl_decompose
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: stl varname [period(12)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    period = int(opts.get("period", 12))
    try:
        r = stl_decompose(df, col, period=period)
        lines = [f"\nSTL Decomposition: {col} (period={period})", "-" * 45]
        lines.append(f"  {'Strength of trend':<30} {r['strength_trend']:.4f}")
        lines.append(f"  {'Strength of seasonal':<30} {r['strength_seasonal']:.4f}")
        lines.append(f"  Trend range: [{min(r['trend']):.4f}, {max(r['trend']):.4f}]")
        lines.append(f"  Seasonal range: [{min(r['seasonal']):.4f}, {max(r['seasonal']):.4f}]")
        session._last_model = r
        return "\n".join(lines)
    except Exception as exc:
        return f"stl error: {exc}"


@command("tssmooth", usage="tssmooth varname [method(ma|exp) window(3) alpha(0.3)]")
def cmd_tssmooth(session: Session, args: str) -> str:
    """Smooth a time series column."""
    from openstat.stats.ts_advanced import tssmooth
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: tssmooth varname [method(ma|exp) window(3) alpha(0.3)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    method = opts.get("method", "ma")
    window = int(opts.get("window", 3))
    alpha = float(opts.get("alpha", 0.3))
    session.snapshot()
    try:
        session.df = tssmooth(df, col, method=method, window=window, alpha=alpha)
        new_col = f"{col}_smooth"
        return f"Smoothed '{col}' → '{new_col}' using {method} (window={window}, alpha={alpha})"
    except Exception as exc:
        return f"tssmooth error: {exc}"
