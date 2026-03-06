"""Panel data models: Fixed Effects, Random Effects, Between, Hausman test."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from rich.table import Table
from rich.console import Console

from openstat.stats.models import FitResult


def _try_import_linearmodels():
    try:
        import linearmodels  # noqa: F401
    except ImportError:
        raise ImportError(
            "Panel data models require linearmodels. "
            "Install it with: pip install openstat[panel]"
        )


def _panel_to_fit_result(result, model_type: str, dep: str, indeps: list[str]) -> FitResult:
    """Convert a linearmodels PanelResults to FitResult."""
    params = {name: float(val) for name, val in result.params.items()}
    std_errors = {name: float(val) for name, val in result.std_errors.items()}
    t_values = {name: float(val) for name, val in result.tstats.items()}
    p_values = {name: float(val) for name, val in result.pvalues.items()}
    ci = result.conf_int()
    conf_low = {name: float(ci.loc[name, "lower"]) for name in params}
    conf_high = {name: float(ci.loc[name, "upper"]) for name in params}

    warnings_list: list[str] = []

    return FitResult(
        model_type=model_type,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(result.nobs),
        params=params,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        conf_int_low=conf_low,
        conf_int_high=conf_high,
        r_squared=float(result.rsquared) if hasattr(result, "rsquared") else None,
        f_statistic=float(result.f_statistic.stat) if hasattr(result, "f_statistic") and result.f_statistic is not None else None,
        f_pvalue=float(result.f_statistic.pval) if hasattr(result, "f_statistic") and result.f_statistic is not None else None,
        warnings=warnings_list,
    )


def fit_panel_fe(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    entity_col: str,
    time_col: str,
    robust: bool = False,
    cluster: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Fixed Effects panel model."""
    _try_import_linearmodels()
    from linearmodels.panel import PanelOLS

    pdf = df.select([entity_col, time_col, dep] + indeps).to_pandas().dropna()
    pdf = pdf.set_index([entity_col, time_col])

    import statsmodels.api as sm
    y = pdf[dep]
    X = sm.add_constant(pdf[indeps])

    model = PanelOLS(y, X, entity_effects=True)

    cov_type = "unadjusted"
    cov_kwds: dict = {}
    if cluster:
        cov_type = "clustered"
        cov_kwds["cluster_entity"] = True
    elif robust:
        cov_type = "robust"

    result = model.fit(cov_type=cov_type, **cov_kwds)
    fit = _panel_to_fit_result(result, "Panel FE", dep, ["const"] + indeps)
    return fit, result


def fit_panel_re(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    entity_col: str,
    time_col: str,
    robust: bool = False,
) -> tuple[FitResult, object]:
    """Fit a Random Effects panel model."""
    _try_import_linearmodels()
    from linearmodels.panel import RandomEffects

    pdf = df.select([entity_col, time_col, dep] + indeps).to_pandas().dropna()
    pdf = pdf.set_index([entity_col, time_col])

    import statsmodels.api as sm
    y = pdf[dep]
    X = sm.add_constant(pdf[indeps])

    model = RandomEffects(y, X)
    cov_type = "robust" if robust else "unadjusted"
    result = model.fit(cov_type=cov_type)
    fit = _panel_to_fit_result(result, "Panel RE", dep, ["const"] + indeps)
    return fit, result


def fit_panel_be(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    entity_col: str,
    time_col: str,
) -> tuple[FitResult, object]:
    """Fit a Between Effects panel model."""
    _try_import_linearmodels()
    from linearmodels.panel import BetweenOLS

    pdf = df.select([entity_col, time_col, dep] + indeps).to_pandas().dropna()
    pdf = pdf.set_index([entity_col, time_col])

    import statsmodels.api as sm
    y = pdf[dep]
    X = sm.add_constant(pdf[indeps])

    model = BetweenOLS(y, X)
    result = model.fit()
    fit = _panel_to_fit_result(result, "Panel BE", dep, ["const"] + indeps)
    return fit, result


def hausman_test(fe_result, re_result) -> str:
    """Perform the Hausman test for FE vs RE.

    H0: RE is consistent and efficient (prefer RE).
    H1: RE is inconsistent (prefer FE).
    """
    b_fe = fe_result.params
    b_re = re_result.params

    # Use common coefficients (exclude const)
    common = [k for k in b_fe.index if k in b_re.index and k != "const"]
    if not common:
        return "No common coefficients for Hausman test."

    b_diff = np.array([b_fe[k] - b_re[k] for k in common])
    cov_fe = fe_result.cov.loc[common, common].values
    cov_re = re_result.cov.loc[common, common].values
    cov_diff = cov_fe - cov_re

    try:
        chi2_stat = float(b_diff @ np.linalg.inv(cov_diff) @ b_diff)
    except np.linalg.LinAlgError:
        chi2_stat = float(b_diff @ np.linalg.pinv(cov_diff) @ b_diff)

    from scipy import stats as sp_stats
    df = len(common)
    p_value = float(1 - sp_stats.chi2.cdf(chi2_stat, df))

    recommendation = "Use Fixed Effects (FE)" if p_value < 0.05 else "Use Random Effects (RE)"

    lines = [
        "Hausman Test (FE vs RE)",
        f"  H0: Random Effects model is consistent",
        f"  chi2({df}) = {chi2_stat:.4f}",
        f"  p-value = {p_value:.4f}",
        f"  Recommendation: {recommendation}",
    ]
    return "\n".join(lines)
