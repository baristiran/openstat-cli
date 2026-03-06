"""Advanced ML: RandomForest, GradientBoosting, SVM, t-SNE."""

from __future__ import annotations

import numpy as np
import polars as pl


def fit_random_forest(df: pl.DataFrame, dep: str, indeps: list[str],
                      n_estimators: int = 100, max_depth: int | None = None,
                      task: str = "regression", seed: int = 42) -> dict:
    """Random Forest regressor or classifier."""
    sklearn = __import__("sklearn.ensemble", fromlist=["RandomForestRegressor", "RandomForestClassifier"])
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy()
    X = sub.select(indeps).to_numpy().astype(float)
    if task == "classification":
        clf = sklearn.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
        clf.fit(X, y)
        score = float(clf.score(X, y))
        metric = "accuracy"
    else:
        clf = sklearn.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
        clf.fit(X, y.astype(float))
        score = float(clf.score(X, y.astype(float)))
        metric = "r_squared"
    feat_imp = {col: float(imp) for col, imp in zip(indeps, clf.feature_importances_)}
    return {
        "method": f"Random Forest ({task})",
        "dep": dep, "indeps": indeps,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        metric: score,
        "feature_importances": feat_imp,
        "n_obs": len(y),
        "_model": clf,
    }


def fit_gradient_boosting(df: pl.DataFrame, dep: str, indeps: list[str],
                           n_estimators: int = 100, learning_rate: float = 0.1,
                           max_depth: int = 3, task: str = "regression", seed: int = 42) -> dict:
    """Gradient Boosting regressor or classifier."""
    sklearn = __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor", "GradientBoostingClassifier"])
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy()
    X = sub.select(indeps).to_numpy().astype(float)
    if task == "classification":
        clf = sklearn.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                   max_depth=max_depth, random_state=seed)
        clf.fit(X, y)
        score = float(clf.score(X, y))
        metric = "accuracy"
    else:
        clf = sklearn.GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                                 max_depth=max_depth, random_state=seed)
        clf.fit(X, y.astype(float))
        score = float(clf.score(X, y.astype(float)))
        metric = "r_squared"
    feat_imp = {col: float(imp) for col, imp in zip(indeps, clf.feature_importances_)}
    return {
        "method": f"Gradient Boosting ({task})",
        "dep": dep, "indeps": indeps,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        metric: score,
        "feature_importances": feat_imp,
        "n_obs": len(y),
        "_model": clf,
    }


def fit_svm(df: pl.DataFrame, dep: str, indeps: list[str],
            kernel: str = "rbf", C: float = 1.0,
            task: str = "regression", seed: int = 42) -> dict:
    """Support Vector Machine regressor or classifier."""
    sklearn_svm = __import__("sklearn.svm", fromlist=["SVR", "SVC"])
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy()
    X = sub.select(indeps).to_numpy().astype(float)
    if task == "classification":
        clf = sklearn_svm.SVC(kernel=kernel, C=C, random_state=seed)
        clf.fit(X, y)
        score = float(clf.score(X, y))
        metric = "accuracy"
    else:
        clf = sklearn_svm.SVR(kernel=kernel, C=C)
        clf.fit(X, y.astype(float))
        score = float(clf.score(X, y.astype(float)))
        metric = "r_squared"
    return {
        "method": f"SVM ({task}, kernel={kernel})",
        "dep": dep, "indeps": indeps,
        "kernel": kernel, "C": C,
        metric: score,
        "n_obs": len(y),
        "_model": clf,
    }


def fit_tsne(df: pl.DataFrame, cols: list[str], n_components: int = 2,
             perplexity: float = 30.0, seed: int = 42) -> dict:
    """t-SNE dimensionality reduction."""
    from sklearn.manifold import TSNE
    sub = df.select(cols).drop_nulls()
    X = sub.to_numpy().astype(float)
    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(X) - 1),
                random_state=seed)
    embedding = tsne.fit_transform(X)
    return {
        "method": "t-SNE",
        "cols": cols,
        "n_components": n_components,
        "perplexity": perplexity,
        "embedding": embedding.tolist(),
        "n_obs": len(X),
    }
