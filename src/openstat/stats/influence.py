"""Influence diagnostics: leverage, Cook's D, DFBETAs, outlier detection."""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_influence(df: pl.DataFrame, dep: str, indeps: list[str]) -> dict:
    """Compute OLS influence statistics: leverage, Cook's D, DFBETAs, studentized residuals."""
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    kp = k + 1

    # OLS fit
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat
    mse = (resid @ resid) / (n - kp)

    # Hat matrix diagonal (leverage)
    H = X @ XtX_inv @ X.T
    leverage = np.diag(H)

    # Studentized residuals (internal)
    sigma = np.sqrt(mse)
    std_resid = resid / (sigma * np.sqrt(1 - leverage + 1e-10))

    # Cook's Distance
    cooks_d = (std_resid ** 2 * leverage) / (kp * (1 - leverage + 1e-10))

    # DFBETAs per coefficient
    dfbetas = {}
    se_beta = np.sqrt(mse * np.diag(XtX_inv))
    for j in range(kp):
        name = "_cons" if j == 0 else indeps[j - 1]
        # Approximation: dfbeta_j = h_j * resid_j / (se_beta_j * (1-lev_j))
        dfb = (X[:, j] * resid) / ((n - kp - 1) * se_beta[j] + 1e-10)
        dfbetas[name] = dfb.tolist()

    # Mahalanobis distance on X for outlier detection
    X_centered = X_raw - X_raw.mean(axis=0)
    try:
        cov_inv = np.linalg.pinv(np.cov(X_raw.T))
        mahal = np.array([float(x @ cov_inv @ x) for x in X_centered])
    except Exception:
        mahal = np.zeros(n)

    return {
        "n_obs": n,
        "n_params": kp,
        "leverage": leverage.tolist(),
        "cooks_d": cooks_d.tolist(),
        "std_residuals": std_resid.tolist(),
        "mahalanobis": mahal.tolist(),
        "dfbetas": dfbetas,
        "high_leverage_threshold": 2 * kp / n,
        "high_cooks_threshold": 4 / n,
        "n_high_leverage": int((leverage > 2 * kp / n).sum()),
        "n_high_cooks": int((cooks_d > 4 / n).sum()),
    }


def detect_outliers(df: pl.DataFrame, dep: str, indeps: list[str], threshold: float = 3.0) -> dict:
    """Identify outliers by studentized residuals > threshold."""
    inf = compute_influence(df, dep, indeps)
    std_resid = np.array(inf["std_residuals"])
    outlier_idx = np.where(np.abs(std_resid) > threshold)[0].tolist()
    return {
        "outlier_indices": outlier_idx,
        "n_outliers": len(outlier_idx),
        "threshold": threshold,
        "std_residuals": inf["std_residuals"],
    }
