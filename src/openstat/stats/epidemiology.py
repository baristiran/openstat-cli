"""Epidemiology functions: risk ratios, odds ratios, incidence rates, NNT."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def _ci_log(est: float, n: int, p: float, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate log-based CI for ratio estimates."""
    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    if p <= 0 or p >= 1 or n == 0:
        return float("nan"), float("nan")
    se_log = math.sqrt((1 - p) / (n * p))
    return math.exp(math.log(est) - z * se_log), math.exp(math.log(est) + z * se_log)


def cohort_study(df: pl.DataFrame, outcome: str, exposure: str) -> dict:
    """
    Cohort study: compute RR, ARR, NNT from a 2×2 table.
    outcome and exposure must be binary (0/1).
    """
    from scipy.stats import chi2_contingency, fisher_exact
    sub = df.select([outcome, exposure]).drop_nulls()
    exp = sub[exposure].to_numpy().astype(int)
    out = sub[outcome].to_numpy().astype(int)

    a = int(((exp == 1) & (out == 1)).sum())   # exposed, outcome
    b = int(((exp == 1) & (out == 0)).sum())   # exposed, no outcome
    c = int(((exp == 0) & (out == 1)).sum())   # unexposed, outcome
    d = int(((exp == 0) & (out == 0)).sum())   # unexposed, no outcome

    n_exp = a + b
    n_unexp = c + d
    r_exp = a / n_exp if n_exp > 0 else float("nan")
    r_unexp = c / n_unexp if n_unexp > 0 else float("nan")

    rr = r_exp / r_unexp if r_unexp > 0 else float("nan")
    arr = r_exp - r_unexp
    nnt = 1 / abs(arr) if arr != 0 else float("nan")

    table = [[a, b], [c, d]]
    chi2, p_chi2, _, _ = chi2_contingency(table)
    _, p_fisher = fisher_exact(table)

    rr_lo, rr_hi = _ci_log(rr, n_exp, r_exp) if rr == rr else (float("nan"), float("nan"))

    return {
        "test": "Cohort Study (RR)",
        "exposure": exposure, "outcome": outcome,
        "table_2x2": {"a": a, "b": b, "c": c, "d": d},
        "n_exposed": n_exp, "n_unexposed": n_unexp,
        "risk_exposed": r_exp, "risk_unexposed": r_unexp,
        "risk_ratio": rr, "rr_ci_95_lo": rr_lo, "rr_ci_95_hi": rr_hi,
        "arr": arr, "nnt": nnt,
        "chi2": float(chi2), "p_chi2": float(p_chi2),
        "p_fisher": float(p_fisher),
    }


def case_control(df: pl.DataFrame, outcome: str, exposure: str) -> dict:
    """
    Case-control study: compute OR with 95% CI (Woolf method).
    """
    from scipy.stats import chi2_contingency, fisher_exact
    sub = df.select([outcome, exposure]).drop_nulls()
    exp = sub[exposure].to_numpy().astype(int)
    out = sub[outcome].to_numpy().astype(int)

    a = int(((exp == 1) & (out == 1)).sum())
    b = int(((exp == 0) & (out == 1)).sum())
    c = int(((exp == 1) & (out == 0)).sum())
    d = int(((exp == 0) & (out == 0)).sum())

    or_ = (a * d) / (b * c) if b * c > 0 else float("nan")
    # Woolf 95% CI
    if or_ > 0 and or_ == or_:
        from scipy.stats import norm
        z = float(norm.ppf(0.975))
        se_log_or = math.sqrt(1/max(a, 1) + 1/max(b, 1) + 1/max(c, 1) + 1/max(d, 1))
        or_lo = math.exp(math.log(or_) - z * se_log_or)
        or_hi = math.exp(math.log(or_) + z * se_log_or)
    else:
        or_lo = or_hi = float("nan")

    table = [[a, b], [c, d]]
    chi2, p_chi2, _, _ = chi2_contingency(table)
    _, p_fisher = fisher_exact(table)

    return {
        "test": "Case-Control (OR)",
        "exposure": exposure, "outcome": outcome,
        "table_2x2": {"a": a, "b": b, "c": c, "d": d},
        "odds_ratio": or_, "or_ci_95_lo": or_lo, "or_ci_95_hi": or_hi,
        "chi2": float(chi2), "p_chi2": float(p_chi2),
        "p_fisher": float(p_fisher),
    }


def incidence_rate(df: pl.DataFrame, outcome: str, person_time: str) -> dict:
    """Compute incidence rate = cases / total person-time."""
    sub = df.select([outcome, person_time]).drop_nulls()
    cases = int(sub[outcome].sum())
    pt = float(sub[person_time].sum())
    ir = cases / pt if pt > 0 else float("nan")
    # Exact Poisson CI (Byar's approximation)
    from scipy.stats import chi2
    lo = 0.5 * float(chi2.ppf(0.025, 2 * cases)) / pt if cases > 0 else 0.0
    hi = 0.5 * float(chi2.ppf(0.975, 2 * (cases + 1))) / pt
    return {
        "test": "Incidence Rate",
        "outcome": outcome, "person_time_col": person_time,
        "cases": cases, "person_time": pt,
        "incidence_rate": ir,
        "ir_ci_95_lo": lo, "ir_ci_95_hi": hi,
    }
