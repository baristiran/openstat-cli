"""Time series analysis: ARIMA, VAR, ADF test, ACF/PACF, forecasting."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats

from openstat.stats.models import FitResult


def adf_test(series: np.ndarray, variable_name: str = "y") -> str:
    """Augmented Dickey-Fuller unit root test."""
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag="AIC")
    adf_stat, p_value, used_lag, nobs, crit_values, icbest = result

    lines = [
        f"Augmented Dickey-Fuller Test: {variable_name}",
        f"  ADF Statistic: {adf_stat:.4f}",
        f"  p-value: {p_value:.4f}",
        f"  Lags used: {used_lag}",
        f"  Observations: {nobs}",
        "  Critical Values:",
    ]
    for level, val in crit_values.items():
        lines.append(f"    {level}: {val:.4f}")

    if p_value < 0.05:
        lines.append("  ✓ Reject H0: Series is stationary")
    else:
        lines.append("  ⚠ Cannot reject H0: Series has a unit root (non-stationary)")

    return "\n".join(lines)


def fit_arima(
    df: pl.DataFrame,
    dep: str,
    order: tuple[int, int, int],
    exog_vars: list[str] | None = None,
    time_var: str | None = None,
) -> tuple[FitResult, object]:
    """Fit an ARIMA(p,d,q) model, optionally with exogenous variables (ARIMAX)."""
    from statsmodels.tsa.arima.model import ARIMA

    pdf = df.to_pandas()
    if time_var and time_var in pdf.columns:
        pdf = pdf.sort_values(time_var)
        try:
            pdf = pdf.set_index(time_var)
        except Exception:
            pass

    endog = pdf[dep].dropna()
    exog = pdf[exog_vars].loc[endog.index] if exog_vars else None

    model = ARIMA(endog, exog=exog, order=order)
    result = model.fit()

    # Build FitResult
    params = {name: float(val) for name, val in result.params.items()}
    bse = result.bse
    std_errors = {name: float(bse[name]) for name in params}
    z_values = {name: float(result.tvalues[name]) for name in params}
    p_values = {name: float(result.pvalues[name]) for name in params}
    ci = result.conf_int()
    conf_low = {name: float(ci.loc[name, 0]) for name in params}
    conf_high = {name: float(ci.loc[name, 1]) for name in params}

    warnings_list = [
        f"Order: ARIMA{order}",
        f"AIC: {result.aic:.1f}",
        f"BIC: {result.bic:.1f}",
    ]

    fit = FitResult(
        model_type=f"ARIMA{order}",
        formula=f"{dep} ~ ARIMA{order}" + (f" + {' + '.join(exog_vars)}" if exog_vars else ""),
        dep_var=dep,
        indep_vars=list(params.keys()),
        n_obs=int(result.nobs),
        params=params,
        std_errors=std_errors,
        t_values=z_values,
        p_values=p_values,
        conf_int_low=conf_low,
        conf_int_high=conf_high,
        aic=float(result.aic),
        bic=float(result.bic),
        log_likelihood=float(result.llf),
        warnings=warnings_list,
    )

    return fit, result


def fit_var(
    df: pl.DataFrame,
    variables: list[str],
    lags: int,
    time_var: str | None = None,
) -> tuple[str, object]:
    """Fit a Vector Autoregression (VAR) model.

    Returns summary string and raw result.
    """
    from statsmodels.tsa.api import VAR

    pdf = df.to_pandas()
    if time_var and time_var in pdf.columns:
        pdf = pdf.sort_values(time_var)

    data = pdf[variables].dropna()
    model = VAR(data)
    result = model.fit(maxlags=lags)

    summary = str(result.summary())
    return summary, result


def forecast_model(result, steps: int) -> np.ndarray:
    """Generate forecasts from a fitted time series model."""
    if hasattr(result, 'forecast'):
        # ARIMA result
        fc = result.forecast(steps=steps)
        return np.array(fc)
    elif hasattr(result, 'forecast') and hasattr(result, 'endog'):
        # VAR result
        fc = result.forecast(result.endog[-result.k_ar:], steps=steps)
        return fc
    raise ValueError("Model does not support forecasting")


def compute_irf(var_result, steps: int = 10) -> str:
    """Compute impulse response functions for VAR model."""
    irf = var_result.irf(steps)
    lines = ["Impulse Response Functions:"]
    lines.append(str(irf.summary()))
    return "\n".join(lines)
