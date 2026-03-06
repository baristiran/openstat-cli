"""Extra visualization commands: corrplot, pairplot, violin, qq, residplot."""

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


def _save_or_show(fig, path: str | None, default_name: str) -> str:
    import os
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/{default_name}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


@command("corrplot", usage="corrplot [var1 var2 ...] [saving(path.png)]")
def cmd_corrplot(session: Session, args: str) -> str:
    """Correlation matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib not installed."
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if df[c].dtype in (
            __import__("polars").Float64, __import__("polars").Float32,
            __import__("polars").Int64, __import__("polars").Int32,
        )][:12]
    if len(cols) < 2:
        return "corrplot requires at least 2 numeric variables."
    try:
        import polars as pl
        data = df.select(cols).drop_nulls().to_numpy().astype(float)
        corr = np.corrcoef(data.T)
        fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols) - 1)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr[i, j]) > 0.6 else "black")
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        path = opts.get("saving")
        out = _save_or_show(fig, path, "corrplot.png")
        plt.close(fig)
        return f"Correlation plot saved: {out}"
    except Exception as exc:
        return f"corrplot error: {exc}"


@command("pairplot", usage="pairplot [var1 var2 ...] [saving(path.png)]")
def cmd_pairplot(session: Session, args: str) -> str:
    """Scatter matrix (pairplot) of numeric variables."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib not installed."
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if not cols:
        import polars as pl
        cols = [c for c in df.columns if df[c].dtype in (
            pl.Float64, pl.Float32, pl.Int64, pl.Int32,
        )][:6]
    if len(cols) < 2:
        return "pairplot requires at least 2 numeric variables."
    try:
        data = df.select(cols).drop_nulls().to_numpy().astype(float)
        k = len(cols)
        fig, axes = plt.subplots(k, k, figsize=(2.5 * k, 2.5 * k))
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if i == j:
                    ax.hist(data[:, i], bins=20, color="steelblue", alpha=0.7)
                    ax.set_ylabel("")
                else:
                    ax.scatter(data[:, j], data[:, i], alpha=0.3, s=10, color="steelblue")
                if i == k - 1:
                    ax.set_xlabel(cols[j], fontsize=8)
                if j == 0:
                    ax.set_ylabel(cols[i], fontsize=8)
                ax.tick_params(labelsize=6)
        fig.suptitle("Scatter Matrix", fontsize=12)
        fig.tight_layout()
        path = opts.get("saving")
        out = _save_or_show(fig, path, "pairplot.png")
        plt.close(fig)
        return f"Pair plot saved: {out}"
    except Exception as exc:
        return f"pairplot error: {exc}"


@command("violin", usage="violin var [by(groupvar)] [saving(path.png)]")
def cmd_violin(session: Session, args: str) -> str:
    """Violin plot for distribution visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib not installed."
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: violin var [by(groupvar)] [saving(path.png)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    by = opts.get("by")
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        if by and by in df.columns:
            groups = df[by].drop_nulls().unique().sort().to_list()
            data_groups = [df.filter(__import__("polars").col(by) == g)[col].drop_nulls().to_numpy().astype(float)
                           for g in groups]
            parts = ax.violinplot(data_groups, showmedians=True)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(g) for g in groups])
            ax.set_xlabel(str(by))
        else:
            data = df[col].drop_nulls().to_numpy().astype(float)
            ax.violinplot([data], showmedians=True)
            ax.set_xticks([1])
            ax.set_xticklabels([col])
        ax.set_ylabel(col)
        ax.set_title(f"Violin Plot: {col}")
        fig.tight_layout()
        path = opts.get("saving")
        out = _save_or_show(fig, path, f"violin_{col}.png")
        plt.close(fig)
        return f"Violin plot saved: {out}"
    except Exception as exc:
        return f"violin error: {exc}"


@command("qqplot", usage="qqplot var [saving(path.png)]")
def cmd_qqplot(session: Session, args: str) -> str:
    """Quantile-Quantile (Q-Q) normality plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats as sp_stats
    except ImportError:
        return "matplotlib or scipy not installed."
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: qqplot var [saving(path.png)]"
    col = positional[0]
    if col not in df.columns:
        return f"Column '{col}' not found."
    try:
        data = df[col].drop_nulls().to_numpy().astype(float)
        fig, ax = plt.subplots(figsize=(6, 6))
        (osm, osr), (slope, intercept, r) = sp_stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=15, color="steelblue", label="Data")
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, label="Normal line")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(f"Q-Q Plot: {col}")
        ax.legend()
        # Shapiro-Wilk stat
        if len(data) <= 5000:
            sw_stat, sw_p = sp_stats.shapiro(data[:5000])
            ax.text(0.05, 0.95, f"Shapiro-Wilk p = {sw_p:.4f}", transform=ax.transAxes,
                    fontsize=9, verticalalignment="top")
        fig.tight_layout()
        path = opts.get("saving")
        out = _save_or_show(fig, path, f"qqplot_{col}.png")
        plt.close(fig)
        return f"Q-Q plot saved: {out} (R²={r**2:.4f})"
    except Exception as exc:
        return f"qqplot error: {exc}"


@command("residplot", usage="residplot dep var1 [var2 ...] [saving(path.png)]")
def cmd_residplot(session: Session, args: str) -> str:
    """Residual vs fitted and scale-location plots for OLS."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "matplotlib not installed."
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: residplot dep var1 [var2 ...]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    if dep not in df.columns:
        return f"Column '{dep}' not found."
    if not indeps:
        return "No valid predictor variables."
    try:
        sub = df.select([dep] + indeps).drop_nulls()
        y = sub[dep].to_numpy().astype(float)
        X = np.column_stack([np.ones(len(y)), sub.select(indeps).to_numpy().astype(float)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        resid = y - fitted
        std_resid = resid / max(resid.std(), 1e-10)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Fitted
        axes[0].scatter(fitted, resid, alpha=0.4, s=15, color="steelblue")
        axes[0].axhline(0, color="red", lw=1)
        axes[0].set_xlabel("Fitted values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted")

        # Scale-Location
        axes[1].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.4, s=15, color="steelblue")
        axes[1].set_xlabel("Fitted values")
        axes[1].set_ylabel("√|Standardized residuals|")
        axes[1].set_title("Scale-Location")

        fig.tight_layout()
        path = opts.get("saving")
        out = _save_or_show(fig, path, f"residplot_{dep}.png")
        plt.close(fig)
        return f"Residual plots saved: {out}"
    except Exception as exc:
        return f"residplot error: {exc}"
