"""Bayesian linear regression via scipy (conjugate prior, no PyMC required)."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats


def bayes_ols(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    prior_scale: float = 10.0,
    n_samples: int = 4000,
    credible_interval: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bayesian linear regression using conjugate Normal-Inverse-Gamma prior.

    Analytically exact posterior — no MCMC required.

    prior_scale: scale of the diffuse Normal(0, prior_scale²) prior on coefficients.
    """
    rng = np.random.default_rng(seed)

    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape

    # Add intercept
    X = np.column_stack([np.ones(n), X_raw])
    param_names = ["_cons"] + indeps
    kp = k + 1

    # ── Conjugate prior: β | σ² ~ N(0, prior_scale² I), σ² ~ IG(a0, b0)
    a0 = 0.001
    b0 = 0.001
    V0_inv = np.eye(kp) / prior_scale**2

    # ── Posterior parameters (Normal-Inverse-Gamma)
    XtX = X.T @ X
    Xty = X.T @ y
    Vn_inv = XtX + V0_inv
    Vn = np.linalg.inv(Vn_inv)
    beta_n = Vn @ Xty  # posterior mean of β

    an = a0 + n / 2
    bn = b0 + 0.5 * (y @ y - beta_n @ Vn_inv @ beta_n)

    # ── Draw from posterior
    sigma2_draws = 1.0 / rng.gamma(an, 1.0 / max(bn, 1e-10), size=n_samples)
    beta_draws = np.array([
        rng.multivariate_normal(beta_n, s2 * Vn)
        for s2 in sigma2_draws
    ])

    # ── Summary
    alpha = 1 - credible_interval
    lo, hi = alpha / 2, 1 - alpha / 2

    post_mean = beta_draws.mean(axis=0)
    post_std = beta_draws.std(axis=0)
    post_lo = np.quantile(beta_draws, lo, axis=0)
    post_hi = np.quantile(beta_draws, hi, axis=0)

    # P(β > 0)
    prob_positive = (beta_draws > 0).mean(axis=0)

    # Posterior predictive R²
    y_pred = X @ post_mean
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    coefficients = {}
    for i, name in enumerate(param_names):
        coefficients[name] = {
            "mean": float(post_mean[i]),
            "std": float(post_std[i]),
            f"ci_{int(credible_interval*100)}_lo": float(post_lo[i]),
            f"ci_{int(credible_interval*100)}_hi": float(post_hi[i]),
            "prob_positive": float(prob_positive[i]),
        }

    return {
        "model": "Bayesian OLS (conjugate Normal-IG prior)",
        "dep": dep,
        "indeps": indeps,
        "n_obs": n,
        "n_samples": n_samples,
        "prior_scale": prior_scale,
        "credible_interval": credible_interval,
        "r_squared": r2,
        "sigma_mean": float(np.sqrt(sigma2_draws.mean())),
        "sigma_std": float(np.sqrt(sigma2_draws).std()),
        "coefficients": coefficients,
        "_beta_draws": beta_draws,
        "_sigma2_draws": sigma2_draws,
    }
