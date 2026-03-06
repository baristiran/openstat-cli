"""Multiple Imputation by Chained Equations (MICE) and Rubin's rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats


@dataclass
class MIResult:
    """Combined result from multiple imputation using Rubin's rules."""

    model_type: str
    formula: str
    m: int  # number of imputations
    params: dict[str, float]
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    conf_int_low: dict[str, float]
    conf_int_high: dict[str, float]
    n_obs: int
    within_var: dict[str, float]  # U_bar
    between_var: dict[str, float]  # B
    fmi: dict[str, float]  # fraction of missing information


def _initial_fill(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Initialize missing values by sampling from observed."""
    result = series.copy()
    mask = np.isnan(result)
    observed = result[~mask]
    if len(observed) > 0 and mask.any():
        result[mask] = rng.choice(observed, size=mask.sum())
    return result


def _impute_regression(data: np.ndarray, col_idx: int, rng: np.random.Generator) -> None:
    """Impute a continuous variable using Bayesian linear regression."""
    mask = np.isnan(data[:, col_idx])
    if not mask.any():
        return

    obs_idx = ~mask
    predictors = np.delete(data, col_idx, axis=1)

    y_obs = data[obs_idx, col_idx]
    X_obs = sm.add_constant(predictors[obs_idx])

    try:
        model = sm.OLS(y_obs, X_obs).fit()
        # Draw from posterior (proper imputation)
        beta_hat = model.params
        sigma2 = model.scale
        # Draw sigma from scaled inverse chi-squared
        n = len(y_obs)
        k = len(beta_hat)
        sigma2_draw = sigma2 * (n - k) / rng.chisquare(n - k)
        # Draw beta from multivariate normal
        cov = model.cov_params() * sigma2_draw / sigma2
        beta_draw = rng.multivariate_normal(beta_hat, cov)

        # Predict missing
        X_miss = sm.add_constant(predictors[mask])
        y_pred = X_miss @ beta_draw
        y_pred += rng.normal(0, np.sqrt(sigma2_draw), size=len(y_pred))
        data[mask, col_idx] = y_pred
    except Exception:
        # Fallback: simple mean imputation
        data[mask, col_idx] = np.nanmean(data[:, col_idx])


def _impute_logit(data: np.ndarray, col_idx: int, rng: np.random.Generator) -> None:
    """Impute a binary variable using logistic regression."""
    mask = np.isnan(data[:, col_idx])
    if not mask.any():
        return

    obs_idx = ~mask
    predictors = np.delete(data, col_idx, axis=1)

    y_obs = data[obs_idx, col_idx]
    X_obs = sm.add_constant(predictors[obs_idx])

    try:
        model = sm.Logit(y_obs, X_obs).fit(disp=0)
        beta_hat = model.params
        cov = model.cov_params()
        beta_draw = rng.multivariate_normal(beta_hat, cov)

        X_miss = sm.add_constant(predictors[mask])
        logits = X_miss @ beta_draw
        probs = 1 / (1 + np.exp(-logits))
        data[mask, col_idx] = (rng.random(size=len(probs)) < probs).astype(float)
    except Exception:
        data[mask, col_idx] = np.round(np.nanmean(data[:, col_idx]))


def _impute_pmm(data: np.ndarray, col_idx: int, rng: np.random.Generator, k: int = 5) -> None:
    """Impute using Predictive Mean Matching."""
    mask = np.isnan(data[:, col_idx])
    if not mask.any():
        return

    obs_idx = ~mask
    predictors = np.delete(data, col_idx, axis=1)

    y_obs = data[obs_idx, col_idx]
    X_obs = sm.add_constant(predictors[obs_idx])

    try:
        model = sm.OLS(y_obs, X_obs).fit()
        y_hat_obs = model.predict(X_obs)

        beta_draw = rng.multivariate_normal(model.params, model.cov_params())
        X_miss = sm.add_constant(predictors[mask])
        y_hat_miss = X_miss @ beta_draw

        # For each missing, find k nearest donors and sample
        for i, pred in enumerate(y_hat_miss):
            distances = np.abs(y_hat_obs - pred)
            donor_indices = np.argsort(distances)[:k]
            chosen = rng.choice(donor_indices)
            data[mask, col_idx] = y_obs.iloc[chosen] if hasattr(y_obs, 'iloc') else y_obs[chosen]
    except Exception:
        data[mask, col_idx] = np.nanmean(data[:, col_idx])


def mice_impute(
    df: pl.DataFrame,
    specs: list[tuple[str, str]],
    m: int = 5,
    max_iter: int = 10,
    seed: int = 42,
) -> list[pl.DataFrame]:
    """Run MICE (Multiple Imputation by Chained Equations).

    Args:
        df: Input DataFrame with missing values.
        specs: List of (method, column) tuples.
            method: "regress", "logit", "pmm"
        m: Number of imputed datasets.
        max_iter: Number of MICE iterations.
        seed: Random seed.

    Returns list of m imputed DataFrames.
    """
    rng = np.random.default_rng(seed)
    col_names = [col for _, col in specs]
    other_cols = [c for c in df.columns if c not in col_names]
    all_cols = col_names + other_cols

    imputed_datasets: list[pl.DataFrame] = []

    for imp in range(m):
        # Convert to numpy for fast computation
        data = df.select(all_cols).to_numpy().astype(float)
        n_imp_cols = len(col_names)

        # Initialize missing with random draws from observed
        for i in range(n_imp_cols):
            data[:, i] = _initial_fill(data[:, i], rng)

        # Iterate chained equations
        for _ in range(max_iter):
            for i, (method, _col) in enumerate(specs):
                # Temporarily set imputed values back to NaN for this col
                orig = df[_col].to_numpy().astype(float)
                was_missing = np.isnan(orig)
                save = data[was_missing, i].copy()
                data[was_missing, i] = np.nan

                if method == "regress":
                    _impute_regression(data, i, rng)
                elif method == "logit":
                    _impute_logit(data, i, rng)
                elif method == "pmm":
                    _impute_pmm(data, i, rng)
                else:
                    # Default to regression
                    _impute_regression(data, i, rng)

        # Convert back to Polars
        imputed_df = pl.DataFrame({
            col: data[:, i] for i, col in enumerate(all_cols)
        })
        imputed_datasets.append(imputed_df)

    return imputed_datasets


def rubins_rules(
    estimates: list[dict[str, float]],
    std_errors: list[dict[str, float]],
    n_obs: int,
) -> MIResult:
    """Combine estimates from m imputed datasets using Rubin's rules.

    Args:
        estimates: List of param dicts from each imputed dataset.
        std_errors: List of SE dicts from each imputed dataset.
        n_obs: Number of observations.

    Returns MIResult with combined estimates.
    """
    m = len(estimates)
    var_names = list(estimates[0].keys())

    combined_params: dict[str, float] = {}
    combined_se: dict[str, float] = {}
    combined_t: dict[str, float] = {}
    combined_p: dict[str, float] = {}
    combined_ci_low: dict[str, float] = {}
    combined_ci_high: dict[str, float] = {}
    within_var: dict[str, float] = {}
    between_var: dict[str, float] = {}
    fmi_dict: dict[str, float] = {}

    for var in var_names:
        # Point estimate: average across imputations
        q_vals = np.array([est[var] for est in estimates])
        q_bar = float(np.mean(q_vals))

        # Within-imputation variance
        u_vals = np.array([se[var] ** 2 for se in std_errors])
        u_bar = float(np.mean(u_vals))

        # Between-imputation variance
        b = float(np.var(q_vals, ddof=1))

        # Total variance
        t = u_bar + (1 + 1 / m) * b

        # Degrees of freedom (Barnard-Rubin)
        if b > 0 and u_bar > 0:
            r = (1 + 1 / m) * b / u_bar
            df_old = (m - 1) * (1 + 1 / r) ** 2
            df_obs = (n_obs - len(var_names) + 1) / (n_obs - len(var_names) + 3) * (n_obs - len(var_names)) * (1 - r)
            if df_obs > 0:
                df = (df_old * df_obs) / (df_old + df_obs)
            else:
                df = df_old
            fmi = (r + 2 / (df + 3)) / (r + 1)
        else:
            df = max(n_obs - len(var_names), 1)
            fmi = 0.0

        se = np.sqrt(t)
        t_val = q_bar / se if se > 0 else 0.0
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_val), df))) if df > 0 else 1.0
        ci_low = q_bar - 1.96 * se
        ci_high = q_bar + 1.96 * se

        combined_params[var] = q_bar
        combined_se[var] = float(se)
        combined_t[var] = t_val
        combined_p[var] = p_val
        combined_ci_low[var] = ci_low
        combined_ci_high[var] = ci_high
        within_var[var] = u_bar
        between_var[var] = b
        fmi_dict[var] = fmi

    return MIResult(
        model_type="MI",
        formula="",
        m=m,
        params=combined_params,
        std_errors=combined_se,
        t_values=combined_t,
        p_values=combined_p,
        conf_int_low=combined_ci_low,
        conf_int_high=combined_ci_high,
        n_obs=n_obs,
        within_var=within_var,
        between_var=between_var,
        fmi=fmi_dict,
    )
