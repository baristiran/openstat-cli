"""Bootstrap and permutation test statistics."""

from __future__ import annotations

import numpy as np
import polars as pl


def bootstrap_ci(
    df: pl.DataFrame,
    col: str,
    stat: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    data = df[col].drop_nulls().to_numpy().astype(float)
    n = len(data)

    stat_fns = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "var": np.var,
        "min": np.min,
        "max": np.max,
    }
    if stat not in stat_fns:
        raise ValueError(f"Unknown statistic: {stat}. Use: {', '.join(stat_fns)}")

    fn = stat_fns[stat]
    observed = float(fn(data))

    boot_stats = np.array([
        fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    alpha = 1 - ci
    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    se = float(boot_stats.std())
    bias = float(boot_stats.mean() - observed)

    return {
        "test": f"Bootstrap CI ({stat})",
        "col": col,
        "stat": stat,
        "observed": observed,
        "n_obs": n,
        "n_boot": n_boot,
        "ci_level": ci,
        "ci_lo": lo,
        "ci_hi": hi,
        "se_boot": se,
        "bias": bias,
    }


def bootstrap_diff(
    df: pl.DataFrame,
    col: str,
    by: str,
    stat: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the difference in a statistic between two groups."""
    rng = np.random.default_rng(seed)
    groups = df[by].drop_nulls().unique().sort().to_list()
    if len(groups) != 2:
        raise ValueError(f"bootstrap_diff requires exactly 2 groups, got {len(groups)}")

    g1 = df.filter(pl.col(by) == groups[0])[col].drop_nulls().to_numpy().astype(float)
    g2 = df.filter(pl.col(by) == groups[1])[col].drop_nulls().to_numpy().astype(float)

    stat_fns = {
        "mean": np.mean, "median": np.median, "std": np.std,
        "var": np.var, "min": np.min, "max": np.max,
    }
    fn = stat_fns.get(stat, np.mean)
    observed_diff = float(fn(g1) - fn(g2))

    boot_diffs = np.array([
        fn(rng.choice(g1, size=len(g1), replace=True)) -
        fn(rng.choice(g2, size=len(g2), replace=True))
        for _ in range(n_boot)
    ])

    alpha = 1 - ci
    lo = float(np.quantile(boot_diffs, alpha / 2))
    hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
    # Shift bootstrap distribution to null (mean=0) for p-value
    boot_centered = boot_diffs - boot_diffs.mean()
    p_value = float((np.abs(boot_centered) >= np.abs(observed_diff)).mean())

    return {
        "test": f"Bootstrap Difference ({stat})",
        "col": col, "by": by,
        "groups": [str(g) for g in groups],
        "observed_diff": observed_diff,
        "n_boot": n_boot,
        "ci_level": ci,
        "ci_lo": lo,
        "ci_hi": hi,
        "se_boot": float(boot_diffs.std()),
        "p_value": p_value,
    }


def permutation_test(
    df: pl.DataFrame,
    col: str,
    by: str,
    stat: str = "mean",
    n_perm: int = 2000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> dict:
    """Permutation test for difference between two groups."""
    rng = np.random.default_rng(seed)
    groups = df[by].drop_nulls().unique().sort().to_list()
    if len(groups) != 2:
        raise ValueError(f"permutation_test requires exactly 2 groups, got {len(groups)}")

    g1 = df.filter(pl.col(by) == groups[0])[col].drop_nulls().to_numpy().astype(float)
    g2 = df.filter(pl.col(by) == groups[1])[col].drop_nulls().to_numpy().astype(float)

    stat_fns = {"mean": np.mean, "median": np.median, "std": np.std}
    fn = stat_fns.get(stat, np.mean)

    observed = float(fn(g1) - fn(g2))
    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    perm_stats = np.array([
        fn(perm := rng.permutation(combined), ) - fn(perm[n1:])  # noqa: confusing but valid
        for _ in range(n_perm)
    ])
    # Fix: proper permutation
    perm_stats = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        perm_stats[i] = fn(perm[:n1]) - fn(perm[n1:])

    if alternative == "two-sided":
        p_value = float((np.abs(perm_stats) >= np.abs(observed)).mean())
    elif alternative == "greater":
        p_value = float((perm_stats >= observed).mean())
    else:
        p_value = float((perm_stats <= observed).mean())

    return {
        "test": "Permutation Test",
        "col": col, "by": by,
        "stat": stat,
        "groups": [str(g) for g in groups],
        "observed_diff": observed,
        "n_perm": n_perm,
        "alternative": alternative,
        "p_value": p_value,
        "reject_5pct": p_value < 0.05,
    }


def jackknife_ci(
    df: pl.DataFrame,
    col: str,
    stat: str = "mean",
) -> dict:
    """Jackknife (leave-one-out) bias and standard error estimate."""
    data = df[col].drop_nulls().to_numpy().astype(float)
    n = len(data)
    stat_fns = {
        "mean": np.mean, "median": np.median, "std": np.std,
        "var": np.var, "min": np.min, "max": np.max,
    }
    if stat not in stat_fns:
        raise ValueError(f"Unknown statistic: {stat}")
    fn = stat_fns[stat]
    observed = float(fn(data))

    jack_stats = np.array([
        fn(np.delete(data, i))
        for i in range(n)
    ])
    jack_mean = jack_stats.mean()
    bias = float((n - 1) * (jack_mean - observed))
    se = float(np.sqrt((n - 1) / n * np.sum((jack_stats - jack_mean) ** 2)))

    return {
        "test": f"Jackknife ({stat})",
        "col": col,
        "stat": stat,
        "observed": observed,
        "n_obs": n,
        "bias": bias,
        "se_jackknife": se,
        "bias_corrected": observed - bias,
    }
