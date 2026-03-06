"""Equivalence tests (TOST) and Tobit/Heckman regression."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats


def tost_onemean(
    df: pl.DataFrame,
    col: str,
    mu: float = 0.0,
    delta: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """
    Two One-Sided Tests (TOST) for equivalence: one-sample.
    H0: |mean - mu| >= delta  vs  H1: |mean - mu| < delta
    """
    data = df[col].drop_nulls().to_numpy().astype(float)
    n = len(data)
    x_bar = data.mean()
    se = data.std(ddof=1) / np.sqrt(n)

    # Lower test: H0: mean <= mu - delta
    t_lo = (x_bar - (mu - delta)) / se
    p_lo = float(sp_stats.t.sf(t_lo, df=n - 1))  # one-sided upper

    # Upper test: H0: mean >= mu + delta
    t_hi = (x_bar - (mu + delta)) / se
    p_hi = float(sp_stats.t.cdf(t_hi, df=n - 1))  # one-sided lower

    p_tost = max(p_lo, p_hi)
    equivalent = p_tost < alpha

    return {
        "test": "TOST Equivalence (one-sample)",
        "col": col,
        "n_obs": n,
        "mean": float(x_bar),
        "mu": mu,
        "delta": delta,
        "alpha": alpha,
        "t_lower": float(t_lo),
        "t_upper": float(t_hi),
        "p_lower": p_lo,
        "p_upper": p_hi,
        "p_tost": p_tost,
        "equivalent_at_alpha": equivalent,
    }


def tost_twomeans(
    df: pl.DataFrame,
    col: str,
    by: str,
    delta: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """TOST for equivalence of two independent group means."""
    groups = df[by].drop_nulls().unique().sort().to_list()
    if len(groups) != 2:
        raise ValueError(f"tost_twomeans requires exactly 2 groups, got {len(groups)}")
    g1 = df.filter(pl.col(by) == groups[0])[col].drop_nulls().to_numpy().astype(float)
    g2 = df.filter(pl.col(by) == groups[1])[col].drop_nulls().to_numpy().astype(float)

    diff = float(g1.mean() - g2.mean())
    se = float(np.sqrt(g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2)))
    df_welch = int((g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2))**2 /
                   ((g1.var(ddof=1) / len(g1))**2 / (len(g1) - 1) +
                    (g2.var(ddof=1) / len(g2))**2 / (len(g2) - 1)))

    t_lo = (diff - (-delta)) / se
    t_hi = (diff - delta) / se
    p_lo = float(sp_stats.t.sf(t_lo, df=df_welch))
    p_hi = float(sp_stats.t.cdf(t_hi, df=df_welch))
    p_tost = max(p_lo, p_hi)

    return {
        "test": "TOST Equivalence (two-sample)",
        "col": col, "by": by,
        "groups": [str(g) for g in groups],
        "mean_diff": diff,
        "delta": delta,
        "alpha": alpha,
        "p_tost": p_tost,
        "equivalent_at_alpha": p_tost < alpha,
        "p_lower": p_lo,
        "p_upper": p_hi,
    }


def fit_tobit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    left: float | None = 0.0,
    right: float | None = None,
) -> dict:
    """
    Tobit regression for censored outcomes via MLE (scipy optimize).
    Handles left-censoring (default at 0), right-censoring, or both.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    kp = k + 1

    def neg_ll(params):
        beta = params[:kp]
        log_sigma = params[kp]
        sigma = np.exp(log_sigma)
        xb = X @ beta
        ll = np.zeros(n)

        for i in range(n):
            if left is not None and y[i] <= left:
                ll[i] = norm.logcdf((left - xb[i]) / sigma)
            elif right is not None and y[i] >= right:
                ll[i] = norm.logsf((right - xb[i]) / sigma)
            else:
                ll[i] = norm.logpdf(y[i], loc=xb[i], scale=sigma)
        return -ll.sum()

    # OLS start
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    resid0 = y - X @ beta0
    log_sigma0 = np.log(max(resid0.std(), 1e-4))
    x0 = np.concatenate([beta0, [log_sigma0]])

    try:
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
        beta_hat = res.x[:kp]
        sigma_hat = float(np.exp(res.x[kp]))
        llf = -res.fun
        aic = 2 * (kp + 1) - 2 * llf
        bic = (kp + 1) * np.log(n) - 2 * llf

        param_names = ["_cons"] + indeps
        params = {nm: float(v) for nm, v in zip(param_names, beta_hat)}

        return {
            "method": "Tobit",
            "dep": dep, "indeps": indeps,
            "left_censoring": left,
            "right_censoring": right,
            "params": params,
            "sigma": sigma_hat,
            "log_likelihood": float(llf),
            "aic": float(aic),
            "bic": float(bic),
            "n_obs": n,
            "n_censored_left": int((y <= left).sum()) if left is not None else 0,
            "n_censored_right": int((y >= right).sum()) if right is not None else 0,
        }
    except Exception as exc:
        raise RuntimeError(f"Tobit failed: {exc}") from exc
