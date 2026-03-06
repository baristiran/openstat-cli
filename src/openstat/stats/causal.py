"""Causal inference models: Difference-in-Differences, Propensity Score Matching."""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.spatial import KDTree

from openstat.stats.models import FitResult, _model_type_suffix


# ── Difference-in-Differences ────────────────────────────────────────

def fit_did(
    df: pl.DataFrame,
    dep: str,
    treatment_col: str,
    time_col: str,
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Difference-in-Differences model.

    Model: y = b0 + b1*treatment + b2*post + b3*(treatment*post) + e
    The DiD estimate is b3.
    """
    cols_needed = [dep, treatment_col, time_col]
    if cluster_col:
        cols_needed.append(cluster_col)
    cols_needed = list(dict.fromkeys(cols_needed))

    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    sub = df.select(cols_needed).drop_nulls()
    if sub.height == 0:
        raise ValueError("No observations after dropping missing values")

    warnings_list: list[str] = []
    n_dropped = df.height - sub.height
    if n_dropped > 0:
        warnings_list.append(f"Note: {n_dropped} observation(s) dropped due to missing values.")

    y = sub[dep].to_numpy().astype(float)
    treat = sub[treatment_col].to_numpy().astype(float)
    post = sub[time_col].to_numpy().astype(float)
    interact = treat * post

    X = np.column_stack([np.ones(len(y)), treat, post, interact])
    var_names = ["_cons", treatment_col, time_col, f"{treatment_col}:{time_col}"]

    # Cluster SE
    groups = None
    if cluster_col:
        groups = sub[cluster_col].to_numpy()

    if robust:
        cov_type = "HC1"
        cov_kwds: dict = {}
    elif groups is not None:
        cov_type = "cluster"
        cov_kwds = {"groups": groups}
    else:
        cov_type = "nonrobust"
        cov_kwds = {}

    model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    ci = model.conf_int()

    # Compute group means for diagnostics
    treat_mask = treat == 1
    control_mask = treat == 0
    pre_mask = post == 0
    post_mask = post == 1

    means = {}
    for label, t_mask, p_mask in [
        ("Control, Pre", control_mask, pre_mask),
        ("Control, Post", control_mask, post_mask),
        ("Treatment, Pre", treat_mask, pre_mask),
        ("Treatment, Post", treat_mask, post_mask),
    ]:
        mask = t_mask & p_mask
        if mask.sum() > 0:
            means[label] = float(np.mean(y[mask]))

    if means:
        warnings_list.append("Group means:")
        for label, mean in means.items():
            warnings_list.append(f"  {label}: {mean:.4f}")

    did_coef = float(model.params[3])
    did_se = float(model.bse[3])
    did_p = float(model.pvalues[3])
    warnings_list.append(f"DiD estimate: {did_coef:.4f} (SE={did_se:.4f}, p={did_p:.4f})")

    suffix = _model_type_suffix(robust, groups is not None)

    fit = FitResult(
        model_type="DiD" + suffix,
        formula=f"{dep} ~ {treatment_col} + {time_col} + {treatment_col}:{time_col}",
        dep_var=dep,
        indep_vars=[treatment_col, time_col],
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue) if model.fvalue is not None else None,
        f_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else None,
        warnings=warnings_list,
    )
    return fit, model


# ── Propensity Score Matching ────────────────────────────────────────

def fit_psm(
    df: pl.DataFrame,
    outcome: str,
    covariates: list[str],
    treatment_col: str,
    *,
    n_neighbors: int = 1,
    caliper: float | None = None,
) -> str:
    """Propensity Score Matching: estimate Average Treatment Effect on Treated (ATT).

    Steps:
    1. Logit model for propensity score P(T=1 | X)
    2. KDTree nearest-neighbor matching
    3. ATT = mean(Y_treated - Y_matched_control)
    4. Bootstrap SE
    """
    all_cols = list(dict.fromkeys([outcome, treatment_col] + covariates))
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    sub = df.select(all_cols).drop_nulls()
    if sub.height < 20:
        raise ValueError(f"Too few observations ({sub.height}) for propensity score matching.")

    y = sub[outcome].to_numpy().astype(float)
    treat = sub[treatment_col].to_numpy().astype(float)

    unique_t = set(treat)
    if not unique_t.issubset({0.0, 1.0}):
        raise ValueError(
            f"Treatment variable must be binary (0/1). Found: {sorted(unique_t)[:10]}"
        )

    X = sub.select(covariates).to_numpy().astype(float)
    X_with_const = sm.add_constant(X)

    # Step 1: Propensity score via logit
    logit_model = sm.Logit(treat, X_with_const).fit(disp=0)
    pscore = logit_model.predict(X_with_const)

    # Default caliper
    if caliper is None:
        caliper = 0.2 * np.std(pscore)

    # Step 2: KDTree matching
    treated_idx = np.where(treat == 1)[0]
    control_idx = np.where(treat == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Need both treated and control observations.")

    control_ps = pscore[control_idx].reshape(-1, 1)
    tree = KDTree(control_ps)

    matched_treated = []
    matched_control_outcomes = []
    unmatched = 0

    for t_i in treated_idx:
        ps_t = pscore[t_i]
        dists, idxs = tree.query([[ps_t]], k=n_neighbors)
        dists = dists.flatten()
        idxs = idxs.flatten()

        # Apply caliper
        valid = dists <= caliper
        if not valid.any():
            unmatched += 1
            continue

        matched_treated.append(y[t_i])
        control_outcomes = [y[control_idx[idx]] for idx, v in zip(idxs, valid) if v]
        matched_control_outcomes.append(np.mean(control_outcomes))

    if len(matched_treated) < 5:
        raise ValueError(
            f"Only {len(matched_treated)} treated units matched. "
            f"Try increasing caliper or reducing n_neighbors."
        )

    matched_treated = np.array(matched_treated)
    matched_control_outcomes = np.array(matched_control_outcomes)

    # Step 3: ATT
    att = float(np.mean(matched_treated - matched_control_outcomes))

    # Step 4: Bootstrap SE
    n_boot = 50
    rng = np.random.RandomState(42)
    boot_atts = []
    for _ in range(n_boot):
        boot_idx = rng.choice(len(matched_treated), size=len(matched_treated), replace=True)
        boot_att = float(np.mean(matched_treated[boot_idx] - matched_control_outcomes[boot_idx]))
        boot_atts.append(boot_att)

    se_att = float(np.std(boot_atts, ddof=1))
    t_stat = att / se_att if se_att > 0 else np.nan
    p_value = float(2 * (1 - sp_stats.norm.cdf(np.abs(t_stat))))

    # Balance table: mean difference before/after matching
    balance_lines = []
    for i, cov in enumerate(covariates):
        mean_t = float(np.mean(X[treated_idx, i]))
        mean_c_all = float(np.mean(X[control_idx, i]))
        # Matched controls (approximate via pscore-matched indices)
        balance_lines.append(
            f"  {cov:20s}  Treated: {mean_t:8.4f}  Control: {mean_c_all:8.4f}  "
            f"Diff: {mean_t - mean_c_all:8.4f}"
        )

    lines = [
        "Propensity Score Matching",
        f"  Treatment variable: {treatment_col}",
        f"  Outcome variable: {outcome}",
        f"  Covariates: {', '.join(covariates)}",
        f"  Neighbors: {n_neighbors}, Caliper: {caliper:.4f}",
        "",
        f"  N treated:  {len(treated_idx)}",
        f"  N control:  {len(control_idx)}",
        f"  Matched:    {len(matched_treated)}",
        f"  Unmatched:  {unmatched}",
        "",
        f"  ATT:        {att:.4f}",
        f"  SE:         {se_att:.4f}",
        f"  t-stat:     {t_stat:.4f}",
        f"  p-value:    {p_value:.4f}",
        "",
        "Covariate Balance (before matching):",
    ] + balance_lines

    return "\n".join(lines)
