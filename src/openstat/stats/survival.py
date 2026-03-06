"""Survival analysis: Cox PH, Kaplan-Meier, log-rank test."""

from __future__ import annotations

import numpy as np
import polars as pl

from openstat.stats.models import FitResult


def _try_import_lifelines():
    try:
        import lifelines  # noqa: F401
    except ImportError:
        raise ImportError(
            "Survival analysis requires lifelines. "
            "Install it with: pip install openstat[survival]"
        )


def fit_cox_ph(
    df: pl.DataFrame,
    time_var: str,
    event_var: str,
    covariates: list[str],
    robust: bool = False,
) -> tuple[FitResult, object]:
    """Fit a Cox Proportional Hazards model."""
    _try_import_lifelines()
    from lifelines import CoxPHFitter

    cols = [time_var, event_var] + covariates
    pdf = df.select(cols).to_pandas().dropna()

    cph = CoxPHFitter()
    cph.fit(pdf, duration_col=time_var, event_col=event_var, robust=robust)

    summary = cph.summary
    params = {name: float(summary.loc[name, "coef"]) for name in covariates}
    std_errors = {name: float(summary.loc[name, "se(coef)"]) for name in covariates}
    z_values = {name: float(summary.loc[name, "z"]) for name in covariates}
    p_values = {name: float(summary.loc[name, "p"]) for name in covariates}

    ci_cols = [c for c in summary.columns if "lower" in c.lower()]
    ci_low_col = ci_cols[0] if ci_cols else None
    ci_high_cols = [c for c in summary.columns if "upper" in c.lower()]
    ci_high_col = ci_high_cols[0] if ci_high_cols else None

    conf_low = {}
    conf_high = {}
    for name in covariates:
        conf_low[name] = float(summary.loc[name, ci_low_col]) if ci_low_col else 0.0
        conf_high[name] = float(summary.loc[name, ci_high_col]) if ci_high_col else 0.0

    warnings_list = [
        f"Concordance: {cph.concordance_index_:.4f}",
        f"Partial log-likelihood: {cph.log_likelihood_:.2f}",
    ]

    # Hazard ratios
    hr_lines = ["Hazard Ratios:"]
    for name in covariates:
        hr = float(summary.loc[name, "exp(coef)"])
        hr_lines.append(f"  {name}: {hr:.4f}")
    warnings_list.append("\n".join(hr_lines))

    fit = FitResult(
        model_type="Cox PH",
        formula=f"h(t) ~ {' + '.join(covariates)}",
        dep_var=f"{time_var} (event: {event_var})",
        indep_vars=covariates,
        n_obs=int(len(pdf)),
        params=params,
        std_errors=std_errors,
        t_values=z_values,
        p_values=p_values,
        conf_int_low=conf_low,
        conf_int_high=conf_high,
        log_likelihood=float(cph.log_likelihood_),
        warnings=warnings_list,
    )

    return fit, cph


def kaplan_meier(
    df: pl.DataFrame,
    time_var: str,
    event_var: str,
    group_var: str | None = None,
) -> tuple[str, object | list]:
    """Fit Kaplan-Meier survival curves.

    Returns summary string and fitted KMF object(s).
    """
    _try_import_lifelines()
    from lifelines import KaplanMeierFitter

    pdf = df.select([time_var, event_var] + ([group_var] if group_var else [])).to_pandas().dropna()

    if group_var is None:
        kmf = KaplanMeierFitter()
        kmf.fit(pdf[time_var], event_observed=pdf[event_var])
        median = kmf.median_survival_time_
        lines = [
            f"Kaplan-Meier Estimate (N={len(pdf)})",
            f"  Median survival time: {median:.2f}" if np.isfinite(median) else "  Median survival time: not reached",
            f"  Events: {int(pdf[event_var].sum())}",
            f"  Censored: {int(len(pdf) - pdf[event_var].sum())}",
        ]
        return "\n".join(lines), kmf
    else:
        groups = sorted(pdf[group_var].unique())
        kmfs = []
        lines = [f"Kaplan-Meier Estimates by {group_var}:"]
        for g in groups:
            mask = pdf[group_var] == g
            sub = pdf[mask]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[time_var], event_observed=sub[event_var], label=str(g))
            kmfs.append(kmf)
            median = kmf.median_survival_time_
            lines.append(
                f"\n  Group {g} (N={len(sub)}):"
                f"\n    Median survival: {median:.2f}" if np.isfinite(median)
                else f"\n  Group {g} (N={len(sub)}):\n    Median survival: not reached"
            )
            lines.append(f"    Events: {int(sub[event_var].sum())}")
        return "\n".join(lines), kmfs


def log_rank_test(
    df: pl.DataFrame,
    time_var: str,
    event_var: str,
    group_var: str,
) -> str:
    """Log-rank test comparing survival between groups."""
    _try_import_lifelines()
    from lifelines.statistics import logrank_test

    pdf = df.select([time_var, event_var, group_var]).to_pandas().dropna()
    groups = sorted(pdf[group_var].unique())

    if len(groups) < 2:
        return "Log-rank test requires at least 2 groups."

    if len(groups) == 2:
        g1 = pdf[pdf[group_var] == groups[0]]
        g2 = pdf[pdf[group_var] == groups[1]]
        result = logrank_test(
            g1[time_var], g2[time_var],
            event_observed_A=g1[event_var],
            event_observed_B=g2[event_var],
        )
        lines = [
            f"Log-Rank Test: {group_var}",
            f"  Groups: {groups[0]} vs {groups[1]}",
            f"  Test statistic: {result.test_statistic:.4f}",
            f"  p-value: {result.p_value:.4f}",
        ]
        if result.p_value < 0.05:
            lines.append("  ⚠ Significant difference in survival between groups")
        else:
            lines.append("  ✓ No significant difference in survival")
        return "\n".join(lines)
    else:
        # Pairwise for >2 groups
        lines = [f"Log-Rank Tests (pairwise): {group_var}"]
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                d1 = pdf[pdf[group_var] == g1]
                d2 = pdf[pdf[group_var] == g2]
                result = logrank_test(
                    d1[time_var], d2[time_var],
                    event_observed_A=d1[event_var],
                    event_observed_B=d2[event_var],
                )
                sig = "*" if result.p_value < 0.05 else ""
                lines.append(f"  {g1} vs {g2}: chi2={result.test_statistic:.3f}, p={result.p_value:.4f}{sig}")
        return "\n".join(lines)


def schoenfeld_test(cph_result) -> str:
    """Test proportional hazards assumption via Schoenfeld residuals."""
    try:
        ph_test = cph_result.check_assumptions(show_plots=False, p_value_threshold=1.0)
        lines = ["Proportional Hazards Test (Schoenfeld Residuals):"]
        # check_assumptions returns summary or prints it
        if ph_test is not None:
            lines.append(str(ph_test))
        else:
            lines.append("  PH assumption appears satisfied for all covariates.")
        return "\n".join(lines)
    except Exception as e:
        return f"PH test: {e}"
