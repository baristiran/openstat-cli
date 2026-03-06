"""Data quality commands: duplicates, winsor, standardize, normalize, mdpattern."""

from __future__ import annotations

import re

import polars as pl

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("winsor", usage="winsor varname [p(0.05) gen(newvar)]")
def cmd_winsor(session: Session, args: str) -> str:
    """Winsorize a variable at specified percentile (both tails)."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: winsor varname [p(0.05) gen(newvar)]"
    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."
    p = float(opts.get("p", 0.05))
    new_var = opts.get("gen", f"{var}_w")
    session.snapshot()
    try:
        series = df[var].cast(pl.Float64)
        lo = float(series.quantile(p))
        hi = float(series.quantile(1 - p))
        winsorized = series.clip(lo, hi)
        session.df = df.with_columns(winsorized.alias(new_var))
        n_lo = int((series < lo).sum())
        n_hi = int((series > hi).sum())
        return (
            f"Winsorized '{var}' → '{new_var}'\n"
            f"  Lower cutoff ({p*100:.1f}%): {lo:.4f}  ({n_lo} obs clipped)\n"
            f"  Upper cutoff ({(1-p)*100:.1f}%): {hi:.4f}  ({n_hi} obs clipped)"
        )
    except Exception as exc:
        return f"winsor error: {exc}"


@command("standardize", usage="standardize var1 [var2 ...] [gen(prefix_)]")
def cmd_standardize(session: Session, args: str) -> str:
    """Z-score standardize variables: (x - mean) / std."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if not cols:
        return "No valid numeric variables found."
    prefix = opts.get("gen", "")
    session.snapshot()
    try:
        new_df = df
        new_cols = []
        for col in cols:
            s = df[col].cast(pl.Float64)
            m = float(s.mean())
            sd = float(s.std())
            new_name = f"{prefix}{col}_z" if not prefix else f"{prefix}{col}"
            new_df = new_df.with_columns(((s - m) / max(sd, 1e-10)).alias(new_name))
            new_cols.append(new_name)
        session.df = new_df
        return f"Standardized {len(cols)} variable(s): {new_cols}"
    except Exception as exc:
        return f"standardize error: {exc}"


@command("normalize", usage="normalize var1 [var2 ...] [gen(prefix_)]")
def cmd_normalize(session: Session, args: str) -> str:
    """Min-max normalize variables to [0, 1]."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if not cols:
        return "No valid numeric variables found."
    prefix = opts.get("gen", "")
    session.snapshot()
    try:
        new_df = df
        new_cols = []
        for col in cols:
            s = df[col].cast(pl.Float64)
            lo = float(s.min())
            hi = float(s.max())
            new_name = f"{prefix}{col}_norm" if not prefix else f"{prefix}{col}"
            new_df = new_df.with_columns(((s - lo) / max(hi - lo, 1e-10)).alias(new_name))
            new_cols.append(new_name)
        session.df = new_df
        return f"Normalized {len(cols)} variable(s) to [0,1]: {new_cols}"
    except Exception as exc:
        return f"normalize error: {exc}"


@command("mdpattern", usage="mdpattern [var1 var2 ...]")
def cmd_mdpattern(session: Session, args: str) -> str:
    """Display missing data pattern for all (or specified) variables."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns] or df.columns

    lines = ["\nMissing Data Pattern", "=" * 60]
    lines.append(f"  N = {df.height} observations, {len(cols)} variables\n")

    col_w = max(len(c) for c in cols) + 2
    header = f"  {'Variable':<{col_w}} {'Missing':>8} {'%Missing':>10} {'Complete':>10}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + 32))

    total_missing = 0
    for col in cols:
        n_miss = int(df[col].is_null().sum())
        pct = 100.0 * n_miss / df.height if df.height > 0 else 0.0
        n_complete = df.height - n_miss
        total_missing += n_miss
        bar = "░" * int(pct / 5)  # bar in 5% increments
        lines.append(f"  {col:<{col_w}} {n_miss:>8} {pct:>9.1f}% {n_complete:>10}  {bar}")

    lines.append("  " + "-" * (col_w + 32))
    overall_pct = 100.0 * total_missing / (df.height * len(cols)) if df.height > 0 else 0.0
    lines.append(f"  {'Total missing cells':<{col_w}} {total_missing:>8} {overall_pct:>9.1f}%")

    # Complete cases
    n_complete_rows = int(df.select(cols).drop_nulls().height)
    lines.append(f"\n  Complete rows (no missing in any selected var): {n_complete_rows} ({100*n_complete_rows/df.height:.1f}%)")

    return "\n".join(lines)

# Backward-compat alias — cmd_duplicates moved to data_cmds
from openstat.commands.data_cmds import cmd_duplicates  # noqa: F401
