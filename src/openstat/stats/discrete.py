"""Discrete / censored models: Tobit, Multinomial Logit, Ordered Logit/Probit."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.optimize import minimize

from openstat.stats.models import FitResult, _prepare_data, _cov_args, _model_type_suffix


# ── Tobit (censored regression) ─────────────────────────────────────

def _tobit_loglike(params, y, X, lower, upper):
    """Tobit log-likelihood for scipy.optimize."""
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    Xb = X @ beta
    resid = (y - Xb) / sigma

    ll = 0.0
    for i in range(len(y)):
        if lower is not None and y[i] <= lower:
            # Left-censored
            cdf_val = sp_stats.norm.cdf((lower - Xb[i]) / sigma)
            ll += np.log(max(cdf_val, 1e-300))
        elif upper is not None and y[i] >= upper:
            # Right-censored
            cdf_val = sp_stats.norm.sf((upper - Xb[i]) / sigma)
            ll += np.log(max(cdf_val, 1e-300))
        else:
            # Uncensored
            ll += sp_stats.norm.logpdf(resid[i]) - log_sigma

    return -ll  # minimize negative log-likelihood


def fit_tobit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Tobit (censored) regression model via MLE."""
    y, X, warnings_list, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    if lower_limit is None and upper_limit is None:
        warnings_list.append("Note: No censoring limits specified. Results are equivalent to OLS.")

    n_censored_low = int(np.sum(y <= lower_limit)) if lower_limit is not None else 0
    n_censored_high = int(np.sum(y >= upper_limit)) if upper_limit is not None else 0
    n_uncensored = len(y) - n_censored_low - n_censored_high

    if n_uncensored < len(var_names) + 2:
        raise ValueError(
            f"Too few uncensored observations ({n_uncensored}) for {len(var_names)} parameters."
        )

    # Initial values: OLS estimates + log(sigma)
    ols = sm.OLS(y, X).fit()
    init_beta = ols.params
    init_log_sigma = np.log(np.std(ols.resid))
    init_params = np.append(init_beta, init_log_sigma)

    result = minimize(
        _tobit_loglike, init_params, args=(y, X, lower_limit, upper_limit),
        method="BFGS",
    )

    if not result.success:
        warnings_list.append(f"Warning: Optimization did not fully converge: {result.message}")

    beta = result.x[:-1]
    log_sigma = result.x[-1]
    sigma = np.exp(log_sigma)

    # Standard errors from inverse Hessian
    try:
        hess_inv = result.hess_inv
        if hasattr(hess_inv, 'todense'):
            hess_inv = hess_inv.todense()
        se_all = np.sqrt(np.diag(np.abs(hess_inv)))
    except Exception:
        # Fallback: numerical Hessian
        from scipy.optimize import approx_fprime
        eps = 1e-5
        n_p = len(result.x)
        hess = np.zeros((n_p, n_p))
        for i in range(n_p):
            def grad_i(p):
                g = approx_fprime(p, _tobit_loglike, eps, y, X, lower_limit, upper_limit)
                return g[i]
            hess[i, :] = approx_fprime(result.x, grad_i, eps)
        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(hess)))
        except np.linalg.LinAlgError:
            se_all = np.full(n_p, np.nan)

    se_beta = se_all[:-1]
    se_sigma = se_all[-1]

    # Build coefficient results
    t_vals = beta / se_beta
    p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(t_vals)))
    ci_low = beta - 1.96 * se_beta
    ci_high = beta + 1.96 * se_beta

    # Add sigma as extra parameter
    all_var_names = var_names + ["sigma"]
    all_params = np.append(beta, sigma)
    all_se = np.append(se_beta, se_sigma)
    sigma_t = sigma / se_sigma if se_sigma > 0 else np.nan
    sigma_p = 2 * (1 - sp_stats.norm.cdf(np.abs(sigma_t)))
    all_t = np.append(t_vals, sigma_t)
    all_p = np.append(p_vals, sigma_p)
    all_ci_low = np.append(ci_low, sigma - 1.96 * se_sigma)
    all_ci_high = np.append(ci_high, sigma + 1.96 * se_sigma)

    suffix = _model_type_suffix(robust, groups is not None)
    ll_val = -result.fun

    censor_info = []
    if lower_limit is not None:
        censor_info.append(f"Left-censored at {lower_limit}: {n_censored_low} obs")
    if upper_limit is not None:
        censor_info.append(f"Right-censored at {upper_limit}: {n_censored_high} obs")
    censor_info.append(f"Uncensored: {n_uncensored} obs")
    warnings_list.extend(censor_info)

    fit = FitResult(
        model_type="Tobit" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=len(y),
        params=dict(zip(all_var_names, all_params)),
        std_errors=dict(zip(all_var_names, all_se)),
        t_values=dict(zip(all_var_names, all_t)),
        p_values=dict(zip(all_var_names, all_p)),
        conf_int_low=dict(zip(all_var_names, all_ci_low)),
        conf_int_high=dict(zip(all_var_names, all_ci_high)),
        log_likelihood=ll_val,
        aic=-2 * ll_val + 2 * len(result.x),
        bic=-2 * ll_val + np.log(len(y)) * len(result.x),
        warnings=warnings_list,
    )
    return fit, result


# ── Multinomial Logit ────────────────────────────────────────────────

def fit_mlogit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Multinomial Logit model."""
    y, X, warnings_list, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.MNLogit(y, X).fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds)

    # Get unique categories
    categories = sorted(np.unique(y))
    base_cat = categories[0]
    other_cats = categories[1:]

    # Flatten per-category coefficients
    params_dict = {}
    se_dict = {}
    t_dict = {}
    p_dict = {}
    ci_low_dict = {}
    ci_high_dict = {}

    ci = model.conf_int()  # shape: (n_cats-1, n_vars, 2)

    for j, cat in enumerate(other_cats):
        cat_label = f"y={int(cat)}" if cat == int(cat) else f"y={cat}"
        for i, var in enumerate(var_names):
            key = f"{var} ({cat_label})"
            params_dict[key] = float(model.params[i, j])
            se_dict[key] = float(model.bse[i, j])
            t_dict[key] = float(model.tvalues[i, j])
            p_dict[key] = float(model.pvalues[i, j])
            ci_low_dict[key] = float(ci[j, i, 0])
            ci_high_dict[key] = float(ci[j, i, 1])

    suffix = _model_type_suffix(robust, groups is not None)
    warnings_list.append(f"Base category: {int(base_cat) if base_cat == int(base_cat) else base_cat}")

    fit = FitResult(
        model_type="MNLogit" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=params_dict,
        std_errors=se_dict,
        t_values=t_dict,
        p_values=p_dict,
        conf_int_low=ci_low_dict,
        conf_int_high=ci_high_dict,
        pseudo_r2=float(model.prsquared),
        log_likelihood=float(model.llf),
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings_list,
    )
    return fit, model


# ── Ordered Logit / Probit ───────────────────────────────────────────

def fit_ordered(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    link: str = "logit",
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit an Ordered Logit or Ordered Probit model."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    y, X, warnings_list, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    # OrderedModel does not want a constant — it estimates thresholds instead
    # Remove the constant column (first column from sm.add_constant)
    X_no_const = X[:, 1:]
    var_names_no_const = var_names[1:]  # remove "_cons"

    distr = "logit" if link == "logit" else "probit"

    model = OrderedModel(y, X_no_const, distr=distr)
    result = model.fit(disp=0)

    # params is a numpy array; exog_names gives labels
    n_coefs = len(var_names_no_const)
    n_total = len(result.params)
    ci = result.conf_int()  # shape: (n_total, 2)

    # Coefficients
    params_dict = {}
    se_dict = {}
    t_dict = {}
    p_dict = {}
    ci_low_dict = {}
    ci_high_dict = {}

    for i, var in enumerate(var_names_no_const):
        params_dict[var] = float(result.params[i])
        se_dict[var] = float(result.bse[i])
        t_dict[var] = float(result.tvalues[i])
        p_dict[var] = float(result.pvalues[i])
        ci_low_dict[var] = float(ci[i, 0])
        ci_high_dict[var] = float(ci[i, 1])

    # Threshold (cut-point) parameters
    for i in range(n_coefs, n_total):
        cut_label = f"cut{i - n_coefs + 1}"
        params_dict[cut_label] = float(result.params[i])
        se_dict[cut_label] = float(result.bse[i])
        t_dict[cut_label] = float(result.tvalues[i])
        p_dict[cut_label] = float(result.pvalues[i])
        ci_low_dict[cut_label] = float(ci[i, 0])
        ci_high_dict[cut_label] = float(ci[i, 1])

    suffix = _model_type_suffix(robust, groups is not None)
    model_name = f"O{link.capitalize()}" + suffix  # OLogit or OProbit

    categories = sorted(np.unique(y))
    warnings_list.append(f"Ordered categories: {[int(c) if c == int(c) else c for c in categories]}")

    fit = FitResult(
        model_type=model_name,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(result.nobs),
        params=params_dict,
        std_errors=se_dict,
        t_values=t_dict,
        p_values=p_dict,
        conf_int_low=ci_low_dict,
        conf_int_high=ci_high_dict,
        pseudo_r2=float(result.prsquared) if hasattr(result, "prsquared") else None,
        log_likelihood=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        warnings=warnings_list,
    )
    return fit, result
