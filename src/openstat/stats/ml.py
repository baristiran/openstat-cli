"""Machine learning / penalized regression and decision trees."""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    from sklearn.linear_model import (  # type: ignore[import]
        Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # type: ignore[import]
    from sklearn.model_selection import cross_val_score, KFold  # type: ignore[import]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import]
    from sklearn.metrics import r2_score, mean_squared_error  # type: ignore[import]
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _require_sklearn():
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for ML commands.\n"
            "Install: pip install scikit-learn"
        )


def _prep(df: pl.DataFrame, dep: str, indeps: list[str]):
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X = sub.select(indeps).to_numpy().astype(float)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    return y, X_s, X, scaler


# ── Lasso ─────────────────────────────────────────────────────────────────

def fit_lasso(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    alpha: float | None = None,
    cv: int = 5,
) -> dict:
    """Lasso regression with optional cross-validated alpha selection."""
    _require_sklearn()
    y, X_s, X_raw, scaler = _prep(df, dep, indeps)

    if alpha is None:
        model = LassoCV(cv=cv, max_iter=10000)
        model.fit(X_s, y)
        alpha = float(model.alpha_)
    else:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_s, y)

    coef = model.coef_
    y_pred = model.predict(X_s)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    n_nonzero = int(np.sum(coef != 0))

    return {
        "method": "Lasso",
        "dep": dep,
        "indeps": indeps,
        "alpha": alpha,
        "coefficients": dict(zip(indeps, coef.tolist())),
        "intercept": float(model.intercept_),
        "r_squared": r2,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n_obs": len(y),
        "n_nonzero": n_nonzero,
        "n_zeroed": len(indeps) - n_nonzero,
        "_model": model,
        "_scaler": scaler,
        "_indeps": indeps,
    }


# ── Ridge ─────────────────────────────────────────────────────────────────

def fit_ridge(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    alpha: float | None = None,
    cv: int = 5,
) -> dict:
    """Ridge regression with optional cross-validated alpha selection."""
    _require_sklearn()
    y, X_s, X_raw, scaler = _prep(df, dep, indeps)

    if alpha is None:
        alphas = np.logspace(-3, 5, 50)
        model = RidgeCV(alphas=alphas, cv=cv)
        model.fit(X_s, y)
        alpha = float(model.alpha_)
    else:
        model = Ridge(alpha=alpha)
        model.fit(X_s, y)

    coef = model.coef_
    y_pred = model.predict(X_s)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))

    return {
        "method": "Ridge",
        "dep": dep,
        "indeps": indeps,
        "alpha": alpha,
        "coefficients": dict(zip(indeps, coef.tolist())),
        "intercept": float(model.intercept_),
        "r_squared": r2,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n_obs": len(y),
        "_model": model,
        "_scaler": scaler,
        "_indeps": indeps,
    }


# ── Elastic Net ────────────────────────────────────────────────────────────

def fit_elasticnet(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    alpha: float | None = None,
    l1_ratio: float = 0.5,
    cv: int = 5,
) -> dict:
    """Elastic Net regression."""
    _require_sklearn()
    y, X_s, X_raw, scaler = _prep(df, dep, indeps)

    if alpha is None:
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=cv, max_iter=10000)
        model.fit(X_s, y)
        alpha = float(model.alpha_)
        l1_ratio = float(model.l1_ratio_)
    else:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_s, y)

    coef = model.coef_
    y_pred = model.predict(X_s)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))

    return {
        "method": "ElasticNet",
        "dep": dep,
        "indeps": indeps,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "coefficients": dict(zip(indeps, coef.tolist())),
        "intercept": float(model.intercept_),
        "r_squared": r2,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n_obs": len(y),
        "_model": model,
        "_scaler": scaler,
        "_indeps": indeps,
    }


# ── Decision Tree ──────────────────────────────────────────────────────────

def fit_cart(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    task: str = "regression",
    max_depth: int | None = 5,
    min_samples_leaf: int = 5,
) -> dict:
    """CART: decision tree for regression or classification."""
    _require_sklearn()
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy()
    X = sub.select(indeps).to_numpy().astype(float)

    if task == "classification":
        y = y.astype(str)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model.fit(X, y)
        score = float(model.score(X, y))
        metric_name = "accuracy"
    else:
        y = y.astype(float)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model.fit(X, y)
        y_pred = model.predict(X)
        score = float(r2_score(y, y_pred))
        metric_name = "r_squared"

    importances = dict(zip(indeps, model.feature_importances_.tolist()))

    return {
        "method": "CART",
        "task": task,
        "dep": dep,
        "indeps": indeps,
        "max_depth": max_depth,
        "n_leaves": int(model.get_n_leaves()),
        "n_obs": len(y),
        metric_name: score,
        "feature_importances": importances,
        "_model": model,
        "_indeps": indeps,
    }


# ── Cross-validation ───────────────────────────────────────────────────────

def cross_validate_model(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    method: str = "ols",
    k: int = 5,
    alpha: float = 1.0,
    scoring: str = "r2",
) -> dict:
    """K-fold cross-validation for various models."""
    _require_sklearn()
    y, X_s, X_raw, scaler = _prep(df, dep, indeps)

    method_lower = method.lower()
    if method_lower == "lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
        X_fit = X_s
    elif method_lower == "ridge":
        model = Ridge(alpha=alpha)
        X_fit = X_s
    elif method_lower == "elasticnet":
        model = ElasticNet(alpha=alpha, max_iter=10000)
        X_fit = X_s
    elif method_lower == "cart":
        model = DecisionTreeRegressor(max_depth=5)
        X_fit = X_raw
    else:
        # OLS via sklearn-compatible wrapper
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X_fit = X_s

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_fit, y, cv=kf, scoring=scoring)

    return {
        "method": method,
        "dep": dep,
        "indeps": indeps,
        "k_folds": k,
        "scoring": scoring,
        "scores": scores.tolist(),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "n_obs": len(y),
    }
