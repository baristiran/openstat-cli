"""Power analysis — one/two-sample means, proportions, OLS."""

from __future__ import annotations

import math

from scipy import stats as sp_stats


# ── Helpers ────────────────────────────────────────────────────────────────

def _solve(fn, lo=1, hi=1_000_000, tol=1e-6):
    """Bisection solver: find x in [lo, hi] where fn(x) ≈ 0."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if fn(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# ── One-sample mean ────────────────────────────────────────────────────────

def power_onemean(
    effect_size: float | None = None,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    sd: float = 1.0,
    delta: float | None = None,
    two_sided: bool = True,
) -> dict:
    """
    Power analysis for one-sample t-test.

    Provide exactly two of: effect_size (or delta/sd), n, power.
    """
    if delta is not None and effect_size is None:
        effect_size = delta / sd

    sides = 2 if two_sided else 1
    za2 = sp_stats.norm.ppf(1 - alpha / sides)

    def _power_from_n(n_val):
        zb = abs(effect_size) * math.sqrt(n_val) - za2
        return sp_stats.norm.cdf(zb)

    if n is None and power is not None and effect_size is not None:
        # Solve for n
        n_val = _solve(lambda x: _power_from_n(x) - power)
        n = math.ceil(n_val)
        achieved = _power_from_n(n)
    elif power is None and n is not None and effect_size is not None:
        achieved = _power_from_n(n)
        power = achieved
    elif effect_size is None and n is not None and power is not None:
        # Solve for detectable effect size
        zb = sp_stats.norm.ppf(power)
        effect_size = (za2 + zb) / math.sqrt(n)
        achieved = power
    else:
        raise ValueError("Provide exactly two of: effect_size, n, power")

    return {
        "test": "One-sample t-test",
        "effect_size": round(effect_size, 6),
        "alpha": alpha,
        "n": n,
        "power": round(power, 6),
        "two_sided": two_sided,
    }


# ── Two-sample means ───────────────────────────────────────────────────────

def power_twomeans(
    effect_size: float | None = None,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    ratio: float = 1.0,
    sd: float = 1.0,
    delta: float | None = None,
    two_sided: bool = True,
) -> dict:
    """Power analysis for two-sample independent t-test."""
    if delta is not None and effect_size is None:
        effect_size = delta / sd

    sides = 2 if two_sided else 1
    za2 = sp_stats.norm.ppf(1 - alpha / sides)

    def _power_from_n(n1):
        n2 = n1 * ratio
        se = math.sqrt(1 / n1 + 1 / n2)
        zb = abs(effect_size) / se - za2
        return sp_stats.norm.cdf(zb)

    if n is None and power is not None and effect_size is not None:
        n_val = _solve(lambda x: _power_from_n(x) - power)
        n = math.ceil(n_val)
        achieved = _power_from_n(n)
        power = achieved
    elif power is None and n is not None and effect_size is not None:
        power = _power_from_n(n)
    elif effect_size is None and n is not None and power is not None:
        zb = sp_stats.norm.ppf(power)
        n2 = n * ratio
        se = math.sqrt(1 / n + 1 / n2)
        effect_size = (za2 + zb) * se
        achieved = power
    else:
        raise ValueError("Provide exactly two of: effect_size, n, power")

    return {
        "test": "Two-sample t-test",
        "effect_size": round(effect_size, 6),
        "alpha": alpha,
        "n1": n,
        "n2": math.ceil(n * ratio),
        "power": round(power, 6),
        "ratio": ratio,
        "two_sided": two_sided,
    }


# ── One proportion ─────────────────────────────────────────────────────────

def power_oneproportion(
    p0: float,
    pa: float,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    two_sided: bool = True,
) -> dict:
    """Power analysis for one-sample proportion z-test."""
    sides = 2 if two_sided else 1
    za2 = sp_stats.norm.ppf(1 - alpha / sides)
    effect_size = abs(pa - p0) / math.sqrt(p0 * (1 - p0))

    def _power_from_n(n_val):
        se_null = math.sqrt(p0 * (1 - p0) / n_val)
        se_alt = math.sqrt(pa * (1 - pa) / n_val)
        z = (abs(pa - p0) - za2 * se_null) / se_alt
        return sp_stats.norm.cdf(z)

    if n is None and power is not None:
        n_val = _solve(lambda x: _power_from_n(x) - power)
        n = math.ceil(n_val)
        power = _power_from_n(n)
    elif power is None and n is not None:
        power = _power_from_n(n)
    else:
        raise ValueError("Provide exactly one of: n, power")

    return {
        "test": "One-sample proportion z-test",
        "p0": p0,
        "pa": pa,
        "effect_size": round(effect_size, 6),
        "alpha": alpha,
        "n": n,
        "power": round(power, 6),
        "two_sided": two_sided,
    }


# ── Two proportions ────────────────────────────────────────────────────────

def power_twoproportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    two_sided: bool = True,
) -> dict:
    """Power analysis for two-sample proportion z-test."""
    sides = 2 if two_sided else 1
    za2 = sp_stats.norm.ppf(1 - alpha / sides)
    p_avg = (p1 + p2) / 2
    effect_size = abs(p2 - p1) / math.sqrt(p_avg * (1 - p_avg))

    def _power_from_n(n_val):
        se_null = math.sqrt(2 * p_avg * (1 - p_avg) / n_val)
        se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_val)
        z = (abs(p2 - p1) - za2 * se_null) / se_alt
        return sp_stats.norm.cdf(z)

    if n is None and power is not None:
        n_val = _solve(lambda x: _power_from_n(x) - power)
        n = math.ceil(n_val)
        power = _power_from_n(n)
    elif power is None and n is not None:
        power = _power_from_n(n)
    else:
        raise ValueError("Provide exactly one of: n, power")

    return {
        "test": "Two-sample proportion z-test",
        "p1": p1,
        "p2": p2,
        "effect_size": round(effect_size, 6),
        "alpha": alpha,
        "n": n,
        "power": round(power, 6),
        "two_sided": two_sided,
    }


# ── OLS / multiple regression ──────────────────────────────────────────────

def power_ols(
    f2: float | None = None,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    k: int = 1,
) -> dict:
    """
    Power analysis for OLS / multiple regression (Cohen's f²).

    f2 = R² / (1 - R²)
    k  = number of predictors
    """

    def _power_from_n(n_val):
        df1 = k
        df2 = n_val - k - 1
        if df2 <= 0:
            return 0.0
        nc = f2 * n_val
        return 1 - sp_stats.f.cdf(
            sp_stats.f.ppf(1 - alpha, df1, df2), df1, df2, nc
        )

    if n is None and power is not None and f2 is not None:
        n_val = _solve(lambda x: _power_from_n(x) - power, lo=k + 2)
        n = math.ceil(n_val)
        power = _power_from_n(n)
    elif power is None and n is not None and f2 is not None:
        power = _power_from_n(n)
    elif f2 is None and n is not None and power is not None:
        f2 = _solve(
            lambda f: _power_from_n_f2(n, f, alpha, k) - power,  # type: ignore[arg-type]
            lo=0.0001,
            hi=10,
        )
        power = _power_from_n(n)
    else:
        raise ValueError("Provide exactly two of: f2, n, power")

    return {
        "test": "OLS / Multiple Regression",
        "f2": round(f2, 6),
        "alpha": alpha,
        "n": n,
        "k": k,
        "power": round(power, 6),
    }


def _power_from_n_f2(n, f2, alpha, k):
    df1 = k
    df2 = n - k - 1
    if df2 <= 0:
        return 0.0
    nc = f2 * n
    return 1 - sp_stats.f.cdf(
        sp_stats.f.ppf(1 - alpha, df1, df2), df1, df2, nc
    )


# ── sampsi (Stata-style) ───────────────────────────────────────────────────

def sampsi(
    mu1: float,
    mu2: float,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> dict:
    """Compute required sample size for two-sample t-test (Stata sampsi style)."""
    effect_size = abs(mu2 - mu1) / sd
    return power_twomeans(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        two_sided=two_sided,
    )
