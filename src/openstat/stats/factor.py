"""Factor Analysis and PCA."""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    from sklearn.decomposition import PCA as _SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── PCA ────────────────────────────────────────────────────────────────────

def fit_pca(
    df: pl.DataFrame,
    cols: list[str],
    n_components: int | None = None,
) -> dict:
    """
    Fit PCA using numpy SVD (no sklearn required).

    Returns:
        eigenvalues, loadings, explained_variance_ratio, cumulative_var,
        scores (component scores for each observation), n_components, cols
    """
    X = df.select(cols).to_numpy().astype(float)
    # standardise
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

    n, p = X.shape
    if n_components is None:
        n_components = min(n, p)
    n_components = min(n_components, min(n, p))

    cov = X.T @ X / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    total_var = eigvals.sum()
    if total_var <= 0:
        total_var = 1.0
    evr = eigvals / p  # proportion of total variance (eigenvalue / p)
    cum_var = np.cumsum(eigvals / p)

    scores = X @ eigvecs

    return {
        "eigenvalues": eigvals.tolist(),
        "loadings": eigvecs.tolist(),        # shape (p, n_components)
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "scores": scores.tolist(),
        "n_components": n_components,
        "cols": cols,
    }


# ── Varimax rotation ───────────────────────────────────────────────────────

def varimax_rotation(loadings: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """Orthogonal varimax rotation of factor loadings."""
    p, k = loadings.shape
    rotation = np.eye(k)

    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                x = loadings @ rotation
                u = x[:, i] ** 2 - x[:, j] ** 2
                v = 2 * x[:, i] * x[:, j]
                A = v.sum()
                B = u.sum()
                C = (v**2 - u**2).sum()
                D = 2 * (u * v).sum()
                num = D - 2 * A * B / p
                den = C - (A**2 - B**2) / p
                if abs(den) < 1e-15:
                    continue
                theta = 0.25 * np.arctan2(num, den)
                c, s = np.cos(theta), np.sin(theta)
                rot2 = np.eye(k)
                rot2[i, i] = c
                rot2[j, j] = c
                rot2[i, j] = -s
                rot2[j, i] = s
                rotation = rotation @ rot2

        if np.max(np.abs(rotation - old_rotation)) < tol:
            break

    return loadings @ rotation


# ── Factor Analysis ────────────────────────────────────────────────────────

def fit_factor(
    df: pl.DataFrame,
    cols: list[str],
    n_factors: int = 2,
    method: str = "pc",
    rotate: bool = True,
) -> dict:
    """
    Fit a factor analysis model.

    method: 'pc' (principal components extraction) or 'ml' (max likelihood via statsmodels if available)
    rotate: apply varimax rotation when True

    Returns:
        loadings, communalities, uniqueness, n_factors, cols
    """
    X = df.select(cols).to_numpy().astype(float)
    n, p = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)
    n_factors = min(n_factors, p)

    if method == "ml":
        try:
            import statsmodels.multivariate.factor as sm_fa  # type: ignore[import]
            fa = sm_fa.Factor(X, n_factor=n_factors, method="ml")
            res = fa.fit()
            loadings = np.array(res.loadings)
            if rotate and n_factors > 1:
                loadings = varimax_rotation(loadings)
            communalities = (loadings**2).sum(axis=1)
            uniqueness = 1 - communalities
            return {
                "method": "ml",
                "loadings": loadings.tolist(),
                "communalities": communalities.tolist(),
                "uniqueness": uniqueness.tolist(),
                "n_factors": n_factors,
                "cols": cols,
            }
        except Exception:
            pass  # fall through to PC method

    # Principal components extraction
    cov = X.T @ X / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_k = eigvals[:n_factors]
    eigvecs_k = eigvecs[:, :n_factors]

    loadings = eigvecs_k * np.sqrt(np.maximum(eigvals_k, 0))

    if rotate and n_factors > 1:
        loadings = varimax_rotation(loadings)

    communalities = (loadings**2).sum(axis=1)
    uniqueness = 1 - communalities

    return {
        "method": "pc",
        "eigenvalues": eigvals.tolist(),
        "loadings": loadings.tolist(),
        "communalities": communalities.tolist(),
        "uniqueness": uniqueness.tolist(),
        "n_factors": n_factors,
        "cols": cols,
    }
