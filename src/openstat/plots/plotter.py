"""Plotting: histogram, scatter, line, box, bar, heatmap via matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from openstat.config import get_config
from openstat.types import NUMERIC_DTYPES


def _validate_col(df: pl.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")


def _unique_path(directory: Path, stem: str, suffix: str = ".png") -> Path:
    """Return a path that does not collide with existing files.

    First try ``stem.suffix``; if it exists, try ``stem_2.suffix``, etc.
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def plot_histogram(
    df: pl.DataFrame, col: str, output_dir: Path, *, bins: int = 30
) -> Path:
    """Create a histogram and save to PNG."""
    cfg = get_config()
    _validate_col(df, col)
    data = df[col].drop_nulls().to_numpy()

    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    ax.hist(data, bins=bins, edgecolor="white", alpha=0.85, color="#4C72B0")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {col}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, f"hist_{col}")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_scatter(
    df: pl.DataFrame, y_col: str, x_col: str, output_dir: Path
) -> Path:
    """Create a scatter plot and save to PNG."""
    cfg = get_config()
    _validate_col(df, y_col)
    _validate_col(df, x_col)

    sub = df.select([x_col, y_col]).drop_nulls()
    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()

    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    ax.scatter(x, y, alpha=0.6, s=20, color="#4C72B0")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, f"scatter_{y_col}_vs_{x_col}")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_line(
    df: pl.DataFrame, y_col: str, x_col: str, output_dir: Path
) -> Path:
    """Create a line plot and save to PNG."""
    cfg = get_config()
    _validate_col(df, y_col)
    _validate_col(df, x_col)

    sub = df.select([x_col, y_col]).drop_nulls().sort(x_col)
    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()

    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, color="#4C72B0")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} over {x_col}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, f"line_{y_col}_vs_{x_col}")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_box(
    df: pl.DataFrame, col: str, output_dir: Path, *, group_col: str | None = None
) -> Path:
    """Create a box plot, optionally grouped. Save to PNG."""
    _validate_col(df, col)
    if group_col:
        _validate_col(df, group_col)

    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))

    if group_col:
        groups = df[group_col].unique().sort().to_list()
        data = []
        labels = []
        for g in groups:
            vals = df.filter(pl.col(group_col) == g)[col].drop_nulls().to_numpy()
            if len(vals) > 0:
                data.append(vals)
                labels.append(str(g))
        ax.boxplot(data, tick_labels=labels)
        ax.set_xlabel(group_col)
        ax.set_title(f"{col} by {group_col}")
    else:
        data = df[col].drop_nulls().to_numpy()
        ax.boxplot(data)
        ax.set_title(f"Box plot of {col}")

    ax.set_ylabel(col)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = f"_{col}_by_{group_col}" if group_col else f"_{col}"
    path = _unique_path(output_dir, f"box{name_suffix}")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_bar(
    df: pl.DataFrame, col: str, output_dir: Path, *, group_col: str | None = None
) -> Path:
    """Create a bar chart. Shows mean of a numeric col by group, or counts of a categorical col."""
    _validate_col(df, col)

    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))

    if group_col:
        _validate_col(df, group_col)
        agg = df.group_by(group_col).agg(pl.col(col).mean().alias("mean")).sort(group_col)
        labels = [str(v) for v in agg[group_col].to_list()]
        values = agg["mean"].to_numpy()
        ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.set_xlabel(group_col)
        ax.set_ylabel(f"Mean of {col}")
        ax.set_title(f"Mean {col} by {group_col}")
    else:
        counts = (
            df.group_by(col).len()
            .sort("len", descending=True)
            .rename({"len": "count"})
            .head(20)
        )
        labels = [str(v) for v in counts[col].to_list()]
        values = counts["count"].to_numpy()
        ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Bar chart of {col}")

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = f"_{col}_by_{group_col}" if group_col else f"_{col}"
    path = _unique_path(output_dir, f"bar{name_suffix}")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_heatmap(
    df: pl.DataFrame, cols: list[str] | None, output_dir: Path
) -> Path:
    """Create a correlation heatmap for numeric columns."""
    cfg = get_config()
    if cols:
        for c in cols:
            _validate_col(df, c)
        num_cols = [c for c in cols if df[c].dtype in NUMERIC_DTYPES]
    else:
        num_cols = [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]

    if len(num_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for a heatmap")

    sub = df.select(num_cols).drop_nulls()
    n = len(num_cols)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = sub.select(pl.corr(num_cols[i], num_cols[j])).item()
            corr_matrix[i, j] = r if r is not None else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n + 2), max(5, n + 1)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "heatmap_corr")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Diagnostic plots (post-estimation)
# ---------------------------------------------------------------------------


def plot_residuals_vs_fitted(
    fitted: np.ndarray, residuals: np.ndarray, output_dir: Path
) -> Path:
    """Residuals vs. fitted values plot."""
    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    ax.scatter(fitted, residuals, alpha=0.6, s=20, color="#4C72B0")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "resid_vs_fitted")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_qq(std_residuals: np.ndarray, output_dir: Path) -> Path:
    """Normal Q-Q plot of standardized residuals."""
    cfg = get_config()
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    sorted_resid = np.sort(std_residuals)
    n = len(sorted_resid)
    theoretical = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    ax.scatter(theoretical, sorted_resid, alpha=0.6, s=20, color="#4C72B0")
    # Add 45-degree reference line
    lims = [min(theoretical.min(), sorted_resid.min()),
            max(theoretical.max(), sorted_resid.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Normal Q-Q Plot")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "qq_plot")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_scale_location(
    fitted: np.ndarray, std_residuals: np.ndarray, output_dir: Path
) -> Path:
    """Scale-Location plot (sqrt of abs standardized residuals vs fitted)."""
    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    sqrt_abs_resid = np.sqrt(np.abs(std_residuals))
    ax.scatter(fitted, sqrt_abs_resid, alpha=0.6, s=20, color="#4C72B0")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel(r"$\sqrt{|Standardized\ residuals|}$")
    ax.set_title("Scale-Location")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "scale_location")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Coefficient plot (post-estimation)
# ---------------------------------------------------------------------------


def plot_coef(
    params: dict,
    ci_lower: dict,
    ci_upper: dict,
    output_dir: Path,
    *,
    title: str = "Coefficient Plot",
    drop_const: bool = True,
) -> Path:
    """Coefficient plot with 95% CI error bars. Saves to PNG."""
    cfg = get_config()

    _CONST_NAMES = {"const", "Intercept", "_cons"}
    names = [k for k in params if not (drop_const and k in _CONST_NAMES)]
    if not names:
        names = list(params.keys())

    coefs = np.array([params[n] for n in names])
    err_lower = np.array([params[n] - ci_lower[n] for n in names])
    err_upper = np.array([ci_upper[n] - params[n] for n in names])

    fig_h = max(cfg.plot_figsize_h, len(names) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, fig_h))

    y_pos = np.arange(len(names))
    ax.errorbar(
        coefs, y_pos,
        xerr=[err_lower, err_upper],
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "coef_plot")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path


def plot_interaction(
    df: pl.DataFrame,
    y_col: str,
    x_col: str,
    mod_col: str,
    output_dir: Path,
    *,
    n_levels: int = 3,
) -> Path:
    """Interaction plot: y vs x for low/medium/high levels of moderator.

    Uses ±1 SD split for continuous moderators, unique values for categorical.
    """
    cfg = get_config()
    _validate_col(df, y_col)
    _validate_col(df, x_col)
    _validate_col(df, mod_col)



    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))

    mod_series = df[mod_col].drop_nulls()
    is_numeric_mod = mod_series.dtype in (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )

    if is_numeric_mod:
        mu = mod_series.mean()
        sd = mod_series.std()
        cuts = {
            f"{mod_col} Low (−1SD)":  df.filter(pl.col(mod_col) < mu - sd),
            f"{mod_col} Mean":        df.filter((pl.col(mod_col) >= mu - sd) & (pl.col(mod_col) <= mu + sd)),
            f"{mod_col} High (+1SD)": df.filter(pl.col(mod_col) > mu + sd),
        }
    else:
        unique_vals = mod_series.unique().sort().to_list()[:n_levels]
        cuts = {
            f"{mod_col}={v}": df.filter(pl.col(mod_col) == v)
            for v in unique_vals
        }

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for i, (label, sub) in enumerate(cuts.items()):
        if sub.is_empty():
            continue
        x_vals = sub[x_col].drop_nulls().to_numpy()
        y_vals = sub[y_col].drop_nulls().to_numpy()
        if len(x_vals) == 0:
            continue
        # regression line for this group
        if len(x_vals) > 1:
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_range, m * x_range + b, color=colors[i % len(colors)], label=label, linewidth=2)
        ax.scatter(x_vals, y_vals, alpha=0.25, color=colors[i % len(colors)], s=20)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Interaction: {y_col} ~ {x_col} × {mod_col}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, "interaction_plot")
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return path
