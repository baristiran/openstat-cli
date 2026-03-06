"""Clustering, MDS, and discriminant analysis."""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering  # type: ignore[import]
    from sklearn.manifold import MDS  # type: ignore[import]
    from sklearn.discriminant_analysis import (  # type: ignore[import]
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore[import]
    from sklearn.metrics import (  # type: ignore[import]
        silhouette_score, calinski_harabasz_score, accuracy_score,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _require_sklearn():
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for clustering commands.\n"
            "Install: pip install scikit-learn"
        )


def _std(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    X = df.select(cols).drop_nulls().to_numpy().astype(float)
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X), X


# ── K-Means ────────────────────────────────────────────────────────────────

def fit_kmeans(
    df: pl.DataFrame,
    cols: list[str],
    *,
    k: int = 3,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> dict:
    """K-means clustering."""
    _require_sklearn()
    X_s, X_raw = _std(df, cols)
    n = len(X_s)

    model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
    labels = model.fit_predict(X_s)

    sil = float(silhouette_score(X_s, labels)) if k > 1 else float("nan")
    ch = float(calinski_harabasz_score(X_s, labels)) if k > 1 else float("nan")
    inertia = float(model.inertia_)

    cluster_sizes = {int(i): int((labels == i).sum()) for i in range(k)}

    # Cluster centroids (in original scale)
    centroids_std = model.cluster_centers_
    # Back-transform using per-column stats
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0) + 1e-15
    centroids_orig = centroids_std * stds + means

    return {
        "method": "K-Means",
        "cols": cols,
        "k": k,
        "n_obs": n,
        "inertia": inertia,
        "silhouette_score": sil,
        "calinski_harabasz": ch,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids_orig.tolist(),
        "labels": labels.tolist(),
        "_model": model,
    }


# ── Hierarchical (Agglomerative) ───────────────────────────────────────────

def fit_hierarchical(
    df: pl.DataFrame,
    cols: list[str],
    *,
    k: int = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> dict:
    """Agglomerative hierarchical clustering."""
    _require_sklearn()
    X_s, _ = _std(df, cols)
    n = len(X_s)

    link = linkage if linkage != "ward" or metric == "euclidean" else "average"
    model = AgglomerativeClustering(n_clusters=k, linkage=link)
    labels = model.fit_predict(X_s)

    sil = float(silhouette_score(X_s, labels)) if k > 1 else float("nan")
    ch = float(calinski_harabasz_score(X_s, labels)) if k > 1 else float("nan")
    cluster_sizes = {int(i): int((labels == i).sum()) for i in range(k)}

    return {
        "method": "Hierarchical",
        "cols": cols,
        "k": k,
        "linkage": linkage,
        "n_obs": n,
        "silhouette_score": sil,
        "calinski_harabasz": ch,
        "cluster_sizes": cluster_sizes,
        "labels": labels.tolist(),
        "_model": model,
    }


# ── MDS ────────────────────────────────────────────────────────────────────

def fit_mds(
    df: pl.DataFrame,
    cols: list[str],
    *,
    n_components: int = 2,
    metric: bool = True,
    random_state: int = 42,
) -> dict:
    """Multidimensional Scaling."""
    _require_sklearn()
    X_s, _ = _std(df, cols)

    model = MDS(
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        normalized_stress="auto",
    )
    coords = model.fit_transform(X_s)
    stress = float(model.stress_)

    return {
        "method": "MDS",
        "cols": cols,
        "n_components": n_components,
        "metric": metric,
        "stress": stress,
        "n_obs": len(X_s),
        "coordinates": coords.tolist(),
        "_model": model,
    }


# ── Discriminant Analysis ──────────────────────────────────────────────────

def fit_discriminant(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    method: str = "lda",
) -> dict:
    """Linear or Quadratic Discriminant Analysis."""
    _require_sklearn()
    sub = df.select([dep] + indeps).drop_nulls()
    y_raw = sub[dep].to_numpy()
    X = sub.select(indeps).to_numpy().astype(float)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    if method.lower() == "qda":
        model = QuadraticDiscriminantAnalysis()
    else:
        model = LinearDiscriminantAnalysis()

    model.fit(X, y)
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))

    classes = le.classes_.tolist()
    prior = model.priors_.tolist() if hasattr(model, "priors_") else []

    result = {
        "method": method.upper(),
        "dep": dep,
        "indeps": indeps,
        "classes": classes,
        "n_classes": len(classes),
        "priors": prior,
        "accuracy": acc,
        "n_obs": len(y),
        "_model": model,
        "_le": le,
    }

    # LDA-specific: discriminant function coefficients
    if method.lower() == "lda" and hasattr(model, "coef_"):
        result["coefficients"] = {
            cls: dict(zip(indeps, model.coef_[i].tolist()))
            for i, cls in enumerate(classes[1:])  # k-1 functions
        }

    return result
