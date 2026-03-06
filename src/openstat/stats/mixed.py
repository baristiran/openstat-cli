"""Mixed / hierarchical linear models: random intercepts, random slopes, ICC."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats as sp_stats

from openstat.stats.models import FitResult


def _mixed_to_fit_result(result, dep: str, fixed: list[str], group_var: str, re_vars: list[str]) -> FitResult:
    """Convert statsmodels MixedLMResults to FitResult."""
    fe_params = result.fe_params
    bse = result.bse_fe
    tvalues = result.tvalues
    pvalues = result.pvalues

    var_names = list(fe_params.index)
    params = {name: float(fe_params[name]) for name in var_names}
    std_errors = {name: float(bse[name]) for name in var_names}
    t_vals = {name: float(tvalues[name]) for name in var_names}
    p_vals = {name: float(pvalues[name]) for name in var_names}
    ci = result.conf_int()
    conf_low = {name: float(ci.loc[name, 0]) for name in var_names}
    conf_high = {name: float(ci.loc[name, 1]) for name in var_names}

    warnings_list: list[str] = []
    warnings_list.append(f"Group variable: {group_var}")
    n_groups = result.model.n_groups if hasattr(result.model, 'n_groups') else "?"
    warnings_list.append(f"Number of groups: {n_groups}")

    # Random effects variance
    re_cov = result.cov_re
    if re_cov is not None and re_cov.size > 0:
        if hasattr(re_cov, 'iloc'):
            re_var = float(re_cov.iloc[0, 0])
        else:
            re_var = float(re_cov[0, 0]) if re_cov.ndim > 1 else float(re_cov[0])
        warnings_list.append(f"Random intercept variance: {re_var:.4f}")
        resid_var = float(result.scale)
        icc = re_var / (re_var + resid_var)
        warnings_list.append(f"ICC: {icc:.4f}")

    re_desc = "Random intercept" if not re_vars else f"Random intercept + slopes: {', '.join(re_vars)}"
    warnings_list.append(re_desc)

    return FitResult(
        model_type="Mixed LM",
        formula=f"{dep} ~ {' + '.join(fixed)} || {group_var}: {' + '.join(re_vars) if re_vars else '(intercept)'}",
        dep_var=dep,
        indep_vars=var_names,
        n_obs=int(result.nobs),
        params=params,
        std_errors=std_errors,
        t_values=t_vals,
        p_values=p_vals,
        conf_int_low=conf_low,
        conf_int_high=conf_high,
        log_likelihood=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        warnings=warnings_list,
    )


def fit_mixed(
    df: pl.DataFrame,
    dep: str,
    fixed: list[str],
    group_var: str,
    re_vars: list[str] | None = None,
) -> tuple[FitResult, object]:
    """Fit a mixed/hierarchical linear model.

    Args:
        dep: Dependent variable name.
        fixed: Fixed effect variable names.
        group_var: Grouping variable name for random effects.
        re_vars: Variables with random slopes. Empty = random intercept only.
    """
    all_cols = list(dict.fromkeys([dep] + fixed + [group_var] + (re_vars or [])))
    pdf = df.select(all_cols).to_pandas().dropna()

    endog = pdf[dep]
    exog = sm.add_constant(pdf[fixed])
    groups = pdf[group_var]

    if re_vars:
        exog_re = pdf[re_vars]
    else:
        exog_re = None  # random intercept only

    model = MixedLM(endog, exog, groups, exog_re=exog_re)
    result = model.fit(reml=True)

    fit = _mixed_to_fit_result(result, dep, ["const"] + fixed, group_var, re_vars or [])
    return fit, result


def compute_icc(result) -> float:
    """Compute Intraclass Correlation Coefficient from mixed model result."""
    re_cov = result.cov_re
    if hasattr(re_cov, 'iloc'):
        re_var = float(re_cov.iloc[0, 0])
    else:
        re_var = float(re_cov[0, 0]) if re_cov.ndim > 1 else float(re_cov[0])
    resid_var = float(result.scale)
    return re_var / (re_var + resid_var)


def lr_test(result_restricted, result_full) -> dict:
    """Likelihood ratio test between nested mixed models.

    Returns dict with statistic, df, p_value.
    """
    ll_r = result_restricted.llf
    ll_f = result_full.llf
    lr_stat = 2 * (ll_f - ll_r)
    df_r = result_restricted.df_modelwc
    df_f = result_full.df_modelwc
    df_diff = df_f - df_r
    if df_diff <= 0:
        df_diff = 1
    p_value = float(1 - sp_stats.chi2.cdf(lr_stat, df_diff))
    return {"lr_stat": float(lr_stat), "df": int(df_diff), "p_value": p_value}
