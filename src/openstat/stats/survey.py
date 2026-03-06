"""Survey-weighted estimation: weighted means, WLS, Taylor linearization, DEFF."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats

from openstat.stats.models import FitResult


def weighted_summary(df: pl.DataFrame, cols: list[str], weight_var: str) -> str:
    """Compute weighted summary statistics."""
    pdf = df.to_pandas()
    weights = pdf[weight_var].values

    lines = ["Survey-Weighted Summary Statistics:"]
    lines.append(f"{'Variable':>15} {'Wt.Mean':>12} {'Wt.SE':>12} {'N':>8}")
    lines.append("-" * 50)

    for col in cols:
        vals = pdf[col].values
        mask = ~np.isnan(vals) & ~np.isnan(weights)
        v = vals[mask]
        w = weights[mask]
        n = len(v)
        if n == 0:
            lines.append(f"{col:>15} {'—':>12} {'—':>12} {0:>8}")
            continue

        # Weighted mean
        wt_mean = np.average(v, weights=w)
        # Weighted variance (for SE estimation)
        w_sum = w.sum()
        w_sum2 = (w ** 2).sum()
        wt_var = np.average((v - wt_mean) ** 2, weights=w) * w_sum ** 2 / (w_sum ** 2 - w_sum2)
        wt_se = np.sqrt(wt_var / n)

        lines.append(f"{col:>15} {wt_mean:>12.4f} {wt_se:>12.4f} {n:>8}")

    return "\n".join(lines)


def fit_weighted_ols(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    weight_var: str,
    strata_var: str | None = None,
    psu_var: str | None = None,
) -> tuple[FitResult, object]:
    """Fit weighted OLS with optional Taylor linearization for complex survey SE."""
    all_cols = [dep] + indeps + [weight_var]
    if strata_var:
        all_cols.append(strata_var)
    if psu_var:
        all_cols.append(psu_var)

    pdf = df.select(all_cols).to_pandas().dropna()
    weights = pdf[weight_var].values
    y = pdf[dep].values
    X = sm.add_constant(pdf[indeps].values)
    var_names = ["const"] + indeps

    # Fit WLS
    model = sm.WLS(y, X, weights=weights)
    result = model.fit()

    # Taylor linearization for SE if PSU/strata provided
    if psu_var and strata_var:
        vcov = _taylor_linearization(pdf, y, X, weights, result.params,
                                     psu_var, strata_var)
        se = np.sqrt(np.diag(vcov))
    else:
        se = result.bse

    params = {name: float(result.params[i]) for i, name in enumerate(var_names)}
    std_errors = {name: float(se[i]) for i, name in enumerate(var_names)}
    t_vals = {name: float(result.params[i] / se[i]) if se[i] > 0 else 0.0 for i, name in enumerate(var_names)}
    p_vals = {name: float(2 * (1 - sp_stats.t.cdf(abs(t_vals[name]), result.df_resid))) for name in var_names}
    ci_low = {name: params[name] - 1.96 * std_errors[name] for name in var_names}
    ci_high = {name: params[name] + 1.96 * std_errors[name] for name in var_names}

    warnings_list = [f"Weight variable: {weight_var}"]
    if strata_var:
        warnings_list.append(f"Strata: {strata_var}")
    if psu_var:
        warnings_list.append(f"PSU: {psu_var}")

    fit = FitResult(
        model_type="Svy: OLS",
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=var_names,
        n_obs=int(len(pdf)),
        params=params,
        std_errors=std_errors,
        t_values=t_vals,
        p_values=p_vals,
        conf_int_low=ci_low,
        conf_int_high=ci_high,
        r_squared=float(result.rsquared),
        warnings=warnings_list,
    )

    return fit, result


def fit_weighted_logit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    weight_var: str,
) -> tuple[FitResult, object]:
    """Fit weighted logistic regression."""
    all_cols = [dep] + indeps + [weight_var]
    pdf = df.select(all_cols).to_pandas().dropna()
    weights = pdf[weight_var].values
    y = pdf[dep].values
    X = sm.add_constant(pdf[indeps].values)
    var_names = ["const"] + indeps

    model = sm.Logit(y, X)
    result = model.fit(disp=0, freq_weights=weights)

    params = {name: float(result.params[i]) for i, name in enumerate(var_names)}
    std_errors = {name: float(result.bse[i]) for i, name in enumerate(var_names)}
    t_vals = {name: float(result.tvalues[i]) for i, name in enumerate(var_names)}
    p_vals = {name: float(result.pvalues[i]) for i, name in enumerate(var_names)}
    ci = result.conf_int()
    ci_low = {name: float(ci[i, 0]) for i, name in enumerate(var_names)}
    ci_high = {name: float(ci[i, 1]) for i, name in enumerate(var_names)}

    fit = FitResult(
        model_type="Svy: Logit",
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=var_names,
        n_obs=int(len(pdf)),
        params=params,
        std_errors=std_errors,
        t_values=t_vals,
        p_values=p_vals,
        conf_int_low=ci_low,
        conf_int_high=ci_high,
        pseudo_r2=float(result.prsquared),
        log_likelihood=float(result.llf),
        warnings=[f"Weight variable: {weight_var}"],
    )

    return fit, result


def _taylor_linearization(pdf, y, X, weights, beta, psu_var, strata_var):
    """Taylor series (sandwich) variance estimation for complex survey designs."""
    import pandas as pd

    resid = y - X @ beta
    score = X * (resid * weights)[:, np.newaxis]

    strata = pdf[strata_var].values
    psu = pdf[psu_var].values

    unique_strata = np.unique(strata)
    n_h = len(unique_strata)
    k = X.shape[1]
    meat = np.zeros((k, k))

    for h in unique_strata:
        mask_h = strata == h
        psus_in_h = np.unique(psu[mask_h])
        m_h = len(psus_in_h)
        if m_h < 2:
            continue

        # Sum of scores within each PSU
        score_psu = np.zeros((m_h, k))
        for j, p in enumerate(psus_in_h):
            mask_p = mask_h & (psu == p)
            score_psu[j] = score[mask_p].sum(axis=0)

        score_mean = score_psu.mean(axis=0)
        for j in range(m_h):
            diff = score_psu[j] - score_mean
            meat += np.outer(diff, diff) * m_h / (m_h - 1)

    bread = np.linalg.inv(X.T @ np.diag(weights) @ X)
    vcov = bread @ meat @ bread
    return vcov


def compute_deff(df: pl.DataFrame, col: str, weight_var: str,
                 psu_var: str | None, strata_var: str | None) -> float:
    """Compute design effect (DEFF) for a variable.

    DEFF = var(complex design) / var(SRS of same size)
    """
    pdf = df.select([col, weight_var] + ([psu_var] if psu_var else []) + ([strata_var] if strata_var else [])).to_pandas().dropna()
    vals = pdf[col].values
    weights = pdf[weight_var].values
    n = len(vals)

    # SRS variance
    var_srs = np.var(vals, ddof=1) / n

    # Weighted variance
    wt_mean = np.average(vals, weights=weights)
    w_sum = weights.sum()
    w_sum2 = (weights ** 2).sum()
    # Kish's approximation of design effect
    deff = 1 + (w_sum2 / w_sum ** 2 - 1 / n) * n
    return max(deff, 1.0)
