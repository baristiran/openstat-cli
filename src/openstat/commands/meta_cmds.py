"""Meta-analysis commands: meta, forest plot, funnel plot."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

from openstat.commands.base import command
from openstat.session import Session


def _meta_fe(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Fixed-effects meta-analysis (inverse-variance weighting)."""
    weights = 1.0 / variances
    pooled = np.sum(weights * effects) / np.sum(weights)
    se = math.sqrt(1.0 / np.sum(weights))
    z = pooled / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low = pooled - 1.96 * se
    ci_high = pooled + 1.96 * se

    # Cochran's Q
    q = np.sum(weights * (effects - pooled) ** 2)
    df = len(effects) - 1
    q_pval = 1 - stats.chi2.cdf(q, df) if df > 0 else float("nan")
    i2 = max(0.0, (q - df) / q * 100) if q > df else 0.0

    return {
        "pooled": pooled, "se": se, "z": z, "p": p,
        "ci_low": ci_low, "ci_high": ci_high,
        "Q": q, "Q_df": df, "Q_p": q_pval, "I2": i2,
    }


def _meta_re(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Random-effects meta-analysis (DerSimonian-Laird)."""
    weights_fe = 1.0 / variances
    pooled_fe = np.sum(weights_fe * effects) / np.sum(weights_fe)
    q = np.sum(weights_fe * (effects - pooled_fe) ** 2)
    df = len(effects) - 1

    # Between-study variance (tau^2) via DerSimonian-Laird
    c = np.sum(weights_fe) - np.sum(weights_fe ** 2) / np.sum(weights_fe)
    tau2 = max(0.0, (q - df) / c)

    # RE weights
    weights_re = 1.0 / (variances + tau2)
    pooled_re = np.sum(weights_re * effects) / np.sum(weights_re)
    se_re = math.sqrt(1.0 / np.sum(weights_re))
    z = pooled_re / se_re
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low = pooled_re - 1.96 * se_re
    ci_high = pooled_re + 1.96 * se_re

    q_pval = 1 - stats.chi2.cdf(q, df) if df > 0 else float("nan")
    i2 = max(0.0, (q - df) / q * 100) if q > df else 0.0

    return {
        "pooled": pooled_re, "se": se_re, "z": z, "p": p,
        "ci_low": ci_low, "ci_high": ci_high,
        "tau2": tau2, "tau": math.sqrt(tau2),
        "Q": q, "Q_df": df, "Q_p": q_pval, "I2": i2,
        "model": "Random-effects (DerSimonian-Laird)",
    }


def _sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def _plot_forest(
    study_labels: list[str],
    effects: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    pooled: float,
    pooled_ci_low: float,
    pooled_ci_high: float,
    output_dir: Path,
    title: str = "Forest Plot",
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(study_labels)
    fig_h = max(4, n * 0.45 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(n, 0, -1, dtype=float)

    # Individual studies
    ax.errorbar(
        effects, y_pos,
        xerr=[effects - ci_low, ci_high - effects],
        fmt="s", color="#4C72B0", ecolor="#4C72B0",
        capsize=3, markersize=5, linewidth=1.2, label="Studies",
    )

    # Pooled diamond
    diamond_x = [pooled_ci_low, pooled, pooled_ci_high, pooled, pooled_ci_low]
    diamond_y = [0.0, 0.25, 0.0, -0.25, 0.0]
    ax.fill(diamond_x, diamond_y, color="#E66100", zorder=5, label="Pooled")
    ax.plot([pooled, pooled], [-0.3, n + 0.3], color="#E66100",
            linestyle="--", linewidth=1, alpha=0.6)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_labels)
    ax.set_xlabel("Effect Size")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    from openstat.plots.plotter import _unique_path
    path = _unique_path(output_dir, "forest_plot")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_funnel(
    effects: np.ndarray,
    se: np.ndarray,
    pooled: float,
    output_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_se = se.max()
    se_range = np.linspace(0, max_se * 1.1, 100)
    ci_l = pooled - 1.96 * se_range
    ci_u = pooled + 1.96 * se_range

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(effects, se, alpha=0.7, s=40, color="#4C72B0")
    ax.plot(ci_l, se_range, "r--", linewidth=1, label="95% CI")
    ax.plot(ci_u, se_range, "r--", linewidth=1)
    ax.axvline(pooled, color="gray", linestyle="--", linewidth=1, label=f"Pooled={pooled:.3f}")
    ax.invert_yaxis()
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot")
    ax.legend(fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    from openstat.plots.plotter import _unique_path
    path = _unique_path(output_dir, "funnel_plot")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@command("meta", usage="meta <effect_col> <se_col> [study=<label_col>] [--re|--fe] [--forest] [--funnel]")
def cmd_meta(session: Session, args: str) -> str:
    """Meta-analysis: fixed-effects or random-effects pooling with forest/funnel plots.

    Requires columns for effect sizes and their standard errors.
    Uses DerSimonian-Laird for random-effects (default).

    Examples:
      meta es se
      meta es se study=author --re
      meta es se study=author --forest --funnel
      meta logOR se_logOR study=trial --fe
    """
    import re
    import polars as pl

    df = session.require_data()
    args = args.strip()

    # Parse options
    fe = "--fe" in args
    re_flag = "--re" in args or not fe  # default: random-effects
    forest = "--forest" in args
    funnel = "--funnel" in args
    args_clean = re.sub(r"--\w+", "", args).strip()

    # study=col
    study_col = None
    m = re.search(r"study[= ](\w+)", args_clean)
    if m:
        study_col = m.group(1)
        args_clean = args_clean[:m.start()] + args_clean[m.end():]

    tokens = args_clean.split()
    if len(tokens) < 2:
        return (
            "Usage: meta <effect_col> <se_col> [study=<label_col>] [--re|--fe] [--forest] [--funnel]\n"
            "Effect sizes must be pre-computed (e.g., Cohen's d, log OR, standardized mean diff)."
        )

    eff_col, se_col = tokens[0], tokens[1]
    for c in [eff_col, se_col]:
        if c not in df.columns:
            return f"Column not found: {c}"

    sub = df.select([c for c in [eff_col, se_col, study_col] if c]).drop_nulls()
    effects = sub[eff_col].to_numpy().astype(float)
    ses = sub[se_col].to_numpy().astype(float)
    variances = ses ** 2
    k = len(effects)

    if k < 2:
        return "Need at least 2 studies for meta-analysis."

    labels = (
        [str(v) for v in sub[study_col].to_list()]
        if study_col
        else [f"Study {i+1}" for i in range(k)]
    )

    # Run analysis
    method = "FE" if fe else "RE"
    res = _meta_fe(effects, variances) if fe else _meta_re(effects, variances)

    lines = [
        f"Studies (k): {k}",
        f"Method: {'Fixed-effects (IV)' if fe else 'Random-effects (DerSimonian-Laird)'}",
        "",
        "Individual Studies:",
        f"  {'Study':<20} {'Effect':>8} {'SE':>7} {'95% CI':>18}",
        "  " + "-" * 55,
    ]
    ci_low_arr = effects - 1.96 * ses
    ci_high_arr = effects + 1.96 * ses
    for lbl, eff, se_i, lo, hi in zip(labels, effects, ses, ci_low_arr, ci_high_arr):
        lines.append(f"  {lbl:<20} {eff:>8.4f} {se_i:>7.4f} [{lo:>7.4f}, {hi:>7.4f}]")

    lines += [
        "",
        "Pooled Estimate:",
        f"  Effect = {res['pooled']:.4f}  (95% CI: [{res['ci_low']:.4f}, {res['ci_high']:.4f}])",
        f"  SE = {res['se']:.4f}   z = {res['z']:.3f}   p = {res['p']:.4f} {_sig(res['p'])}",
        "",
        "Heterogeneity:",
        f"  Q({res['Q_df']}) = {res['Q']:.3f}  p = {res['Q_p']:.4f}",
        f"  I² = {res['I2']:.1f}%",
    ]
    if not fe:
        lines.append(f"  τ² = {res['tau2']:.4f}   τ = {res['tau']:.4f}")

    out = "\n" + "=" * 60 + "\nMeta-Analysis Results\n" + "=" * 60 + "\n"
    out += "\n".join(lines) + "\n" + "=" * 60

    # Plots
    plot_msgs = []
    if forest:
        try:
            p = _plot_forest(
                labels, effects, ci_low_arr, ci_high_arr,
                res["pooled"], res["ci_low"], res["ci_high"],
                session.output_dir,
                title=f"Forest Plot (pooled={res['pooled']:.3f})",
            )
            session.plot_paths.append(str(p))
            plot_msgs.append(f"Forest plot saved: {p}")
        except Exception as exc:
            plot_msgs.append(f"Forest plot error: {exc}")

    if funnel:
        try:
            p = _plot_funnel(effects, ses, res["pooled"], session.output_dir)
            session.plot_paths.append(str(p))
            plot_msgs.append(f"Funnel plot saved: {p}")
        except Exception as exc:
            plot_msgs.append(f"Funnel plot error: {exc}")

    if plot_msgs:
        out += "\n" + "\n".join(plot_msgs)

    return out
