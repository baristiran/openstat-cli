"""Factor analysis and PCA commands."""

from __future__ import annotations

import os
import re

import numpy as np

from openstat.commands.base import command
from openstat.session import Session
from openstat.stats.factor import fit_pca, fit_factor


# ── Stata-style arg parser (same as power_cmds) ────────────────────────────

def _stata_parse(raw: str) -> tuple[list[str], dict[str, str], set[str]]:
    opts: dict[str, str] = {}
    positional: list[str] = []
    flags: set[str] = set()

    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)

    rest = re.sub(r'\w+\([^)]*\)', '', raw)

    for tok in rest.split():
        tok = tok.strip(',')
        if not tok:
            continue
        if '=' in tok:
            k, v = tok.split('=', 1)
            opts[k.lower().lstrip('-')] = v
        elif tok.startswith('--'):
            flags.add(tok.lstrip('-').lower())
        elif re.match(r'^\w+$', tok):
            positional.append(tok)

    return positional, opts, flags


def _loadings_table(cols: list[str], loadings: list, blanks: float = 0.0) -> str:
    arr = np.array(loadings)  # shape (p, k)
    k = arr.shape[1]
    header = f"  {'Variable':<15}" + "".join(f"  {'F' + str(i+1):>8}" for i in range(k))
    lines = [header, "-" * (17 + k * 10)]
    for i, col in enumerate(cols):
        row = f"  {col:<15}"
        for j in range(k):
            val = arr[i, j]
            if abs(val) < blanks:
                row += f"  {'':>8}"
            else:
                row += f"  {val:>8.4f}"
        lines.append(row)
    return "\n".join(lines)


@command("pca", usage="pca varlist [, n(k)]")
def cmd_pca(session: Session, args: str) -> str:
    """Principal component analysis."""
    df = session.require_data()
    positional, opts, flags = _stata_parse(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "pca requires at least 2 numeric variables."

    n_components = int(opts["n"]) if "n" in opts else None

    result = fit_pca(df, cols, n_components=n_components)
    session._last_model = result
    session._last_model_vars = (None, cols)

    eigvals = result["eigenvalues"]
    evr = result["explained_variance_ratio"]
    cum = result["cumulative_variance"]
    loadings = result["loadings"]
    k = result["n_components"]

    lines = [f"\nPCA — {len(cols)} variables, {k} components", "=" * 55]
    lines.append(f"  {'Component':<12}  {'Eigenvalue':>12}  {'Var%':>8}  {'Cum%':>8}")
    lines.append("-" * 55)
    for i in range(k):
        lines.append(
            f"  {'Comp' + str(i+1):<12}  {eigvals[i]:>12.4f}  {evr[i]*100:>7.2f}%  {cum[i]*100:>7.2f}%"
        )
    lines.append("=" * 55)
    lines.append("\nLoadings:")
    lines.append(_loadings_table(cols, loadings))
    lines.append("\nRun 'estat screeplot' for a scree plot.")
    return "\n".join(lines)


@command("factor", usage="factor varlist [, n(k) method(pc|ml) --norotate]")
def cmd_factor(session: Session, args: str) -> str:
    """Factor analysis with optional varimax rotation."""
    df = session.require_data()
    positional, opts, flags = _stata_parse(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "factor requires at least 2 numeric variables."

    n_factors = int(opts.get("n", 2))
    method = opts.get("method", "pc").lower()
    rotate = "norotate" not in flags

    result = fit_factor(df, cols, n_factors=n_factors, method=method, rotate=rotate)
    session._last_model = result
    session._last_model_vars = (None, cols)

    loadings = result["loadings"]
    comm = result["communalities"]
    uniq = result["uniqueness"]
    k = result["n_factors"]
    rot_str = " (varimax)" if rotate and k > 1 else ""

    lines = [f"\nFactor Analysis — {method.upper()}{rot_str}", "=" * 60]
    lines.append("\nLoadings:")
    lines.append(_loadings_table(cols, loadings))
    lines.append("\n  " + f"{'Variable':<15}  {'Communality':>12}  {'Uniqueness':>12}")
    lines.append("  " + "-" * 42)
    for i, col in enumerate(cols):
        lines.append(f"  {col:<15}  {comm[i]:>12.4f}  {uniq[i]:>12.4f}")
    lines.append("\nRun 'estat loadings' or 'estat screeplot' for more detail.")
    return "\n".join(lines)


@command("estat", usage="estat screeplot|loadings [, blanks(0.3)]")
def cmd_estat(session: Session, args: str) -> str:
    """Post-estimation: screeplot or loadings table."""
    positional, opts, flags = _stata_parse(args)
    sub = positional[0].lower() if positional else ""

    if sub == "screeplot":
        result = session._last_model
        if result is None or "eigenvalues" not in result:
            return "No PCA/factor result in memory. Run 'pca' or 'factor' first."

        eigvals = result["eigenvalues"]
        lines = ["\nScree Plot", "=" * 50]
        max_e = max(eigvals) if eigvals else 1
        for i, e in enumerate(eigvals):
            bar = int(30 * e / max_e)
            lines.append(f"  Comp{i+1:>2}  {'█' * bar}  {e:.3f}")
        lines.append("=" * 50)

        try:
            import matplotlib.pyplot as plt

            os.makedirs(str(session.output_dir), exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(eigvals) + 1), eigvals, "o-")
            ax.axhline(1, linestyle="--", color="red", linewidth=0.8)
            ax.set_xlabel("Component")
            ax.set_ylabel("Eigenvalue")
            ax.set_title("Scree Plot")
            path = str(session.output_dir / "screeplot.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            session.plot_paths.append(path)
            lines.append(f"\nScree plot saved: {path}")
        except Exception:
            pass

        return "\n".join(lines)

    elif sub == "loadings":
        result = session._last_model
        if result is None or "loadings" not in result:
            return "No PCA/factor result in memory."

        blanks = float(opts.get("blanks", 0.0))
        cols = result["cols"]
        loadings = result["loadings"]
        lines = ["\nFactor/Component Loadings"]
        lines.append(_loadings_table(cols, loadings, blanks=blanks))
        return "\n".join(lines)

    else:
        return f"Unknown estat subcommand: {sub}\nAvailable: screeplot, loadings"
