"""Nonparametric hypothesis tests and rank-based statistics."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats


# ── Spearman rank correlation ──────────────────────────────────────────────

def spearman_corr(df: pl.DataFrame, cols: list[str]) -> dict:
    """Spearman rank correlation matrix."""
    X = df.select(cols).to_numpy().astype(float)
    n = X.shape[1]
    rho = np.eye(n)
    pvals = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                pvals[i, j] = 0.0
                continue
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            r, p = sp_stats.spearmanr(X[mask, i], X[mask, j])
            rho[i, j] = r
            pvals[i, j] = p
    return {"rho": rho.tolist(), "pvalues": pvals.tolist(), "cols": cols}


# ── Mann-Whitney / Wilcoxon rank-sum ──────────────────────────────────────

def ranksum_test(
    df: pl.DataFrame,
    var: str,
    group: str,
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon rank-sum test (Mann-Whitney U) for two independent groups.

    alternative: 'two-sided', 'less', 'greater'
    """
    grp_vals = df[group].drop_nulls().unique().to_list()
    if len(grp_vals) != 2:
        raise ValueError(f"'{group}' must have exactly 2 groups, found {len(grp_vals)}")

    g1 = df.filter(pl.col(group) == grp_vals[0])[var].drop_nulls().to_numpy().astype(float)
    g2 = df.filter(pl.col(group) == grp_vals[1])[var].drop_nulls().to_numpy().astype(float)

    stat, p = sp_stats.mannwhitneyu(g1, g2, alternative=alternative)
    z = (stat - len(g1) * len(g2) / 2) / np.sqrt(
        len(g1) * len(g2) * (len(g1) + len(g2) + 1) / 12
    )
    return {
        "test": "Wilcoxon rank-sum (Mann-Whitney U)",
        "var": var,
        "group": group,
        "groups": grp_vals,
        "n1": len(g1),
        "n2": len(g2),
        "U_statistic": float(stat),
        "z_statistic": float(z),
        "p_value": float(p),
        "alternative": alternative,
    }


# ── Wilcoxon signed-rank ───────────────────────────────────────────────────

def signrank_test(
    df: pl.DataFrame,
    var1: str,
    var2: str | None = None,
    *,
    mu: float = 0.0,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon signed-rank test.

    One-sample: var2=None, tests median of var1 == mu.
    Paired: tests median of (var1 - var2) == 0.
    """
    x = df[var1].drop_nulls().to_numpy().astype(float)
    if var2 is not None:
        y = df[var2].drop_nulls().to_numpy().astype(float)
        diff = x[: len(y)] - y[: len(x)]
    else:
        diff = x - mu

    stat, p = sp_stats.wilcoxon(diff, alternative=alternative)
    return {
        "test": "Wilcoxon signed-rank",
        "var1": var1,
        "var2": var2,
        "mu": mu,
        "n": len(diff),
        "W_statistic": float(stat),
        "p_value": float(p),
        "alternative": alternative,
    }


# ── Kruskal-Wallis ─────────────────────────────────────────────────────────

def kruskal_wallis_test(
    df: pl.DataFrame,
    var: str,
    group: str,
) -> dict:
    """Kruskal-Wallis H test for k independent groups."""
    groups = df[group].drop_nulls().unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[var].drop_nulls().to_numpy().astype(float)
        for g in groups
    ]
    stat, p = sp_stats.kruskal(*samples)
    df_stat = len(groups) - 1
    return {
        "test": "Kruskal-Wallis H",
        "var": var,
        "group": group,
        "k_groups": len(groups),
        "H_statistic": float(stat),
        "df": df_stat,
        "p_value": float(p),
        "groups": groups,
        "n_per_group": [len(s) for s in samples],
    }
