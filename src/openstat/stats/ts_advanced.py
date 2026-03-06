"""Advanced time-series: Granger causality, VECM, Johansen cointegration, STL, tssmooth."""

from __future__ import annotations

import numpy as np
import polars as pl


def granger_causality(df: pl.DataFrame, dep: str, cause: str, maxlag: int = 4) -> dict:
    """Granger causality test: does 'cause' Granger-cause 'dep'?"""
    from statsmodels.tsa.stattools import grangercausalitytests
    sub = df.select([dep, cause]).drop_nulls()
    data = sub.to_numpy()
    results = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    # Collect F-test p-values per lag
    lag_pvals = {}
    for lag, res in results.items():
        lag_pvals[lag] = float(res[0]["ssr_ftest"][1])
    min_pval = min(lag_pvals.values())
    best_lag = min(lag_pvals, key=lag_pvals.get)
    return {
        "test": "Granger Causality",
        "dep": dep,
        "cause": cause,
        "maxlag": maxlag,
        "lag_pvalues": lag_pvals,
        "min_pvalue": min_pval,
        "best_lag": best_lag,
        "reject_null_5pct": min_pval < 0.05,
    }


def johansen_test(df: pl.DataFrame, cols: list[str], det_order: int = -1, k_ar_diff: int = 1) -> dict:
    """Johansen cointegration test."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    sub = df.select(cols).drop_nulls()
    data = sub.to_numpy()
    result = coint_johansen(data, det_order, k_ar_diff)
    trace_stat = result.lr1.tolist()
    trace_cv_90 = result.cvt[:, 0].tolist()
    trace_cv_95 = result.cvt[:, 1].tolist()
    max_stat = result.lr2.tolist()
    max_cv_95 = result.cvm[:, 1].tolist()
    n_cointegrated = int(np.sum(np.array(trace_stat) > np.array(trace_cv_95)))
    return {
        "test": "Johansen Cointegration",
        "cols": cols,
        "trace_statistics": trace_stat,
        "trace_cv_95": trace_cv_95,
        "trace_cv_90": trace_cv_90,
        "max_eigen_statistics": max_stat,
        "max_eigen_cv_95": max_cv_95,
        "n_cointegrating_vectors": n_cointegrated,
    }


def fit_vecm(df: pl.DataFrame, cols: list[str], k_ar_diff: int = 1, coint_rank: int = 1) -> dict:
    """Vector Error Correction Model."""
    from statsmodels.tsa.vector_ar.vecm import VECM
    sub = df.select(cols).drop_nulls()
    data = sub.to_numpy()
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank).fit()
    return {
        "test": "VECM",
        "cols": cols,
        "k_ar_diff": k_ar_diff,
        "coint_rank": coint_rank,
        "alpha": model.alpha.tolist(),
        "beta": model.beta.tolist(),
        "gamma": model.gamma.tolist() if hasattr(model, "gamma") else None,
        "det_coef": model.det_coef.tolist() if hasattr(model, "det_coef") else None,
        "llf": float(model.llf) if hasattr(model, "llf") else None,
        "_model": model,
    }


def stl_decompose(df: pl.DataFrame, col: str, period: int = 12) -> dict:
    """STL decomposition: trend + seasonal + residual."""
    from statsmodels.tsa.seasonal import STL
    sub = df.select([col]).drop_nulls()
    y = sub[col].to_numpy().astype(float)
    stl = STL(y, period=period).fit()
    return {
        "test": "STL Decomposition",
        "col": col,
        "period": period,
        "trend": stl.trend.tolist(),
        "seasonal": stl.seasonal.tolist(),
        "resid": stl.resid.tolist(),
        "strength_trend": float(1 - np.var(stl.resid) / np.var(stl.trend + stl.resid)),
        "strength_seasonal": float(1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)),
        "_model": stl,
    }


def tssmooth(df: pl.DataFrame, col: str, method: str = "ma", window: int = 3, alpha: float = 0.3) -> pl.DataFrame:
    """Smooth a time series: moving average (ma) or exponential smoothing (exp)."""
    series = df[col].to_numpy().astype(float)
    n = len(series)
    if method == "ma":
        smoothed = np.convolve(series, np.ones(window) / window, mode="same")
        # Fix edges
        for i in range(window // 2):
            smoothed[i] = np.mean(series[:i + window // 2 + 1])
            smoothed[n - 1 - i] = np.mean(series[n - i - window // 2 - 1:])
    elif method == "exp":
        smoothed = np.zeros(n)
        smoothed[0] = series[0]
        for t in range(1, n):
            smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t - 1]
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'ma' or 'exp'.")
    new_col = f"{col}_smooth"
    return df.with_columns(pl.Series(new_col, smoothed))
