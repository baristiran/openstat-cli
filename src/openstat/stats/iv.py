"""Instrumental variables: 2SLS, first-stage diagnostics, overidentification tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm

from openstat.stats.models import FitResult


def _try_import_linearmodels():
    try:
        import linearmodels  # noqa: F401
    except ImportError:
        raise ImportError(
            "IV models require linearmodels. "
            "Install it with: pip install openstat[panel]"
        )


def _iv_to_fit_result(result, dep: str, exog: list[str], endog: list[str], instruments: list[str]) -> FitResult:
    """Convert linearmodels IVResults to FitResult."""
    all_vars = list(result.params.index)
    params = {name: float(val) for name, val in result.params.items()}
    std_errors = {name: float(val) for name, val in result.std_errors.items()}
    t_values = {name: float(val) for name, val in result.tstats.items()}
    p_values = {name: float(val) for name, val in result.pvalues.items()}
    ci = result.conf_int()
    conf_low = {name: float(ci.loc[name, "lower"]) for name in all_vars}
    conf_high = {name: float(ci.loc[name, "upper"]) for name in all_vars}

    warnings_list: list[str] = []
    warnings_list.append(f"Endogenous: {', '.join(endog)}")
    warnings_list.append(f"Instruments: {', '.join(instruments)}")

    return FitResult(
        model_type="IV-2SLS",
        formula=f"{dep} ~ {' + '.join(exog)} + ({' + '.join(endog)} = {' + '.join(instruments)})",
        dep_var=dep,
        indep_vars=all_vars,
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


def fit_iv_2sls(
    df: pl.DataFrame,
    dep: str,
    exog: list[str],
    endog: list[str],
    instruments: list[str],
    robust: bool = False,
) -> tuple[FitResult, object]:
    """Fit an IV model via Two-Stage Least Squares."""
    _try_import_linearmodels()
    from linearmodels.iv import IV2SLS

    all_cols = [dep] + exog + endog + instruments
    pdf = df.select(all_cols).to_pandas().dropna()

    dep_data = pdf[dep]
    exog_data = sm.add_constant(pdf[exog]) if exog else sm.add_constant(pdf[[]])
    endog_data = pdf[endog]
    instr_data = pdf[instruments]

    model = IV2SLS(dep_data, exog_data, endog_data, instr_data)
    cov_type = "robust" if robust else "unadjusted"
    result = model.fit(cov_type=cov_type)

    fit = _iv_to_fit_result(result, dep, ["const"] + exog, endog, instruments)
    return fit, result


def first_stage_diagnostics(iv_result) -> str:
    """Report first-stage regression diagnostics."""
    lines = ["First-Stage Diagnostics:"]
    try:
        fs = iv_result.first_stage
        for endog_var in fs.diagnostics:
            diag = fs.diagnostics[endog_var]
            lines.append(f"\n  Endogenous: {endog_var}")
            lines.append(f"    Partial R²: {diag.rsquared:.4f}")
            lines.append(f"    Partial F-stat: {diag.f_stat.stat:.2f} (p={diag.f_stat.pval:.4f})")
            if diag.f_stat.stat < 10:
                lines.append("    ⚠ Weak instruments (F < 10)")
    except Exception as e:
        lines.append(f"  Could not compute: {e}")
    return "\n".join(lines)


def overidentification_test(iv_result) -> str:
    """Sargan/Hansen J-test for overidentifying restrictions."""
    lines = ["Overidentification Test (Sargan/Hansen J):"]
    try:
        j_test = iv_result.sargan
        lines.append(f"  J-statistic: {j_test.stat:.4f}")
        lines.append(f"  p-value: {j_test.pval:.4f}")
        lines.append(f"  df: {j_test.df}")
        if j_test.pval < 0.05:
            lines.append("  ⚠ Reject H0: instruments may not be valid")
        else:
            lines.append("  ✓ Cannot reject H0: instruments appear valid")
    except Exception as e:
        lines.append(f"  Not available (exactly identified or error: {e})")
    return "\n".join(lines)


def endogeneity_test(iv_result) -> str:
    """Durbin-Wu-Hausman test for endogeneity."""
    lines = ["Endogeneity Test (Durbin-Wu-Hausman):"]
    try:
        wu_test = iv_result.wu_hausman()
        lines.append(f"  Statistic: {wu_test.stat:.4f}")
        lines.append(f"  p-value: {wu_test.pval:.4f}")
        if wu_test.pval < 0.05:
            lines.append("  ⚠ Reject H0: endogeneity detected — IV is appropriate")
        else:
            lines.append("  ✓ Cannot reject exogeneity — OLS may be sufficient")
    except Exception as e:
        lines.append(f"  Could not compute: {e}")
    return "\n".join(lines)
