"""Advanced ML: SHAP, hyperopt, learning curve, cross-validation, PLS/PCR."""

from __future__ import annotations

import numpy as np

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ── PLS / PCR ────────────────────────────────────────────────────────────────

@command("pls", usage="pls <y> <x1> [x2 ...] [--components=2]")
def cmd_pls(session: Session, args: str) -> str:
    """Partial Least Squares regression (PLS1/PLS2).

    Handles multicollinearity by projecting predictors into latent components.
    Useful when n_features >> n_samples or predictors are highly correlated.

    Options:
      --components=N   number of latent components (default: 2)
      --cv=K           k-fold cross-validation (default: 5)

    Examples:
      pls y x1 x2 x3 x4 x5 --components=3
      pls outcome pred1 pred2 pred3 --cv=10
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    ca = CommandArgs(args)
    preds = [p for p in ca.positional if not p.startswith("-")]
    if len(preds) < 2:
        return "Usage: pls <y> <x1> [x2 ...] [--components=N]"

    y_col = preds[0]
    x_cols = preds[1:]
    n_components = int(ca.options.get("components", 2))
    cv_k = int(ca.options.get("cv", 5))

    try:
        df = session.require_data()
        sub = df.select([y_col] + x_cols).drop_nulls()
        y = sub[y_col].to_numpy().astype(float).reshape(-1, 1)
        X = sub.select(x_cols).to_numpy().astype(float)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_sc, y)

        y_pred = pls.predict(X_sc).ravel()
        y_flat = y.ravel()
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # CV score
        cv_scores = cross_val_score(pls, X_sc, y_flat, cv=min(cv_k, len(y_flat)), scoring="r2")

        lines = [
            f"Partial Least Squares Regression: {y_col} ~ {' + '.join(x_cols)}",
            f"N={sub.height}  Components={n_comp}",
            "=" * 55,
            f"  R²          : {r2:.4f}",
            f"  CV R² (k={min(cv_k, len(y_flat))}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "",
            "  X Loadings (first component):",
        ]
        x_load = pls.x_loadings_[:, 0]
        for name, load in sorted(zip(x_cols, x_load), key=lambda t: -abs(t[1])):
            lines.append(f"    {name:<20} {load:9.4f}")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "pls")


@command("pcr", usage="pcr <y> <x1> [x2 ...] [--components=2]")
def cmd_pcr(session: Session, args: str) -> str:
    """Principal Component Regression (PCR).

    First runs PCA on predictors, then regresses outcome on components.

    Options:
      --components=N   number of PC components to keep (default: 2)
      --cv=K           k-fold cross-validation (default: 5)

    Examples:
      pcr y x1 x2 x3 x4 x5 --components=3
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    ca = CommandArgs(args)
    preds = [p for p in ca.positional if not p.startswith("-")]
    if len(preds) < 2:
        return "Usage: pcr <y> <x1> [x2 ...] [--components=N]"

    y_col = preds[0]
    x_cols = preds[1:]
    n_components = int(ca.options.get("components", 2))
    cv_k = int(ca.options.get("cv", 5))

    try:
        df = session.require_data()
        sub = df.select([y_col] + x_cols).drop_nulls()
        y = sub[y_col].to_numpy().astype(float)
        X = sub.select(x_cols).to_numpy().astype(float)

        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp)),
            ("reg", LinearRegression()),
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        pca = pipe.named_steps["pca"]
        var_exp = pca.explained_variance_ratio_

        cv_scores = cross_val_score(pipe, X, y, cv=min(cv_k, len(y)), scoring="r2")

        lines = [
            f"Principal Component Regression: {y_col} ~ {' + '.join(x_cols)}",
            f"N={sub.height}  Components={n_comp}",
            "=" * 55,
            f"  R²          : {r2:.4f}",
            f"  CV R² (k={min(cv_k, len(y))}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "",
            "  PCA Components — Variance Explained:",
        ]
        cum = 0.0
        for i, ve in enumerate(var_exp):
            cum += ve
            lines.append(f"    PC{i+1}: {ve*100:.1f}%  (cumulative: {cum*100:.1f}%)")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "pcr")


# ── Cross-validation ─────────────────────────────────────────────────────────

@command("crossval", usage="crossval [--folds=5] [--metric=r2|rmse|mae|accuracy|auc]")
def cmd_crossval(session: Session, args: str) -> str:
    """K-fold cross-validation on the last fitted model.

    Evaluates model generalization using the dataset in the current session.

    Options:
      --folds=K         number of folds (default: 5)
      --metric=<name>   scoring metric: r2, rmse, mae, accuracy, auc (default: r2)
      --seed=N          random seed

    Examples:
      ols income educ age
      crossval --folds=10 --metric=rmse

      logit employed educ age female
      crossval --metric=auc
    """
    try:
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    ca = CommandArgs(args)
    k = int(ca.options.get("folds", 5))
    metric = ca.options.get("metric", "r2").lower()
    seed = int(ca.options.get("seed", getattr(session, "_repro_seed", 42) or 42))

    if session._last_model is None or session._last_model_vars is None:
        return "No model fitted. Run ols/logit/etc. first."

    dep, indeps = session._last_model_vars
    try:
        df = session.require_data()
        sub = df.select([dep] + indeps).drop_nulls()
        y = sub[dep].to_numpy().astype(float)
        X = sub.select(indeps).to_numpy().astype(float)

        # Rebuild sklearn model from statsmodels fit
        model_name = type(session._last_model.model).__name__.lower()

        if "logit" in model_name or "probit" in model_name or "mnlogit" in model_name:
            from sklearn.linear_model import LogisticRegression
            sk_model = LogisticRegression(max_iter=500, random_state=seed)
            cv = StratifiedKFold(n_splits=min(k, len(y)), shuffle=True, random_state=seed)
            metric_map = {
                "accuracy": "accuracy", "auc": "roc_auc",
                "r2": "accuracy",  # fallback
            }
            sk_metric = metric_map.get(metric, "accuracy")
        else:
            from sklearn.linear_model import LinearRegression
            sk_model = LinearRegression()
            cv = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=seed)
            metric_map = {
                "r2": "r2", "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
            }
            sk_metric = metric_map.get(metric, "r2")

        scores = cross_val_score(sk_model, X, y, cv=cv, scoring=sk_metric)
        # Negate for neg_* metrics
        if sk_metric.startswith("neg_"):
            scores = -scores
            display_metric = metric.upper()
        else:
            display_metric = metric.upper()

        lines = [
            f"Cross-Validation: {dep} ~ {' + '.join(indeps)}",
            f"Folds={min(k, len(y))}  Metric={display_metric}  N={len(y)}",
            "=" * 45,
            f"  Mean:   {scores.mean():.4f}",
            f"  Std:    {scores.std():.4f}",
            f"  Min:    {scores.min():.4f}",
            f"  Max:    {scores.max():.4f}",
            "",
            "  Per-fold scores:",
        ]
        for i, s in enumerate(scores):
            lines.append(f"    Fold {i+1}: {s:.4f}")
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "crossval")


# ── Hyperparameter Optimization ──────────────────────────────────────────────

@command("hyperopt", usage="hyperopt <y> <x1> [x2 ...] --model=rf|gb|svm|logit [--cv=5]")
def cmd_hyperopt(session: Session, args: str) -> str:
    """Hyperparameter optimization via GridSearch / RandomSearch.

    Finds optimal hyperparameters for ML models using cross-validation.

    Models: rf (Random Forest), gb (Gradient Boosting), svm, logit, ridge, lasso

    Options:
      --model=<name>    model to optimize (required)
      --cv=K            cross-validation folds (default: 5)
      --n_iter=N        number of random search iterations (default: 20)
      --metric=<name>   scoring metric (default: r2 or accuracy)
      --task=reg|class  regression or classification (auto-detected)

    Examples:
      hyperopt income educ age --model=rf
      hyperopt employed educ age female --model=gb --task=class --cv=10
    """
    try:
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    ca = CommandArgs(args)
    preds = [p for p in ca.positional if not p.startswith("-")]
    if len(preds) < 2:
        return "Usage: hyperopt <y> <x1> [x2 ...] --model=rf|gb|svm|logit"

    y_col = preds[0]
    x_cols = preds[1:]
    model_name = ca.options.get("model", "rf").lower()
    cv_k = int(ca.options.get("cv", 5))
    n_iter = int(ca.options.get("n_iter", 20))
    seed = int(ca.options.get("seed", getattr(session, "_repro_seed", 42) or 42))

    try:
        df = session.require_data()
        sub = df.select([y_col] + x_cols).drop_nulls()
        y = sub[y_col].to_numpy().astype(float)
        X = sub.select(x_cols).to_numpy().astype(float)

        # Auto-detect task
        task = ca.options.get("task", "")
        if not task:
            n_uniq = len(set(y))
            task = "class" if n_uniq <= 10 and (y == y.astype(int)).all() else "reg"

        is_clf = task.startswith("class")
        metric = ca.options.get("metric", "accuracy" if is_clf else "r2")

        # Model + param grid
        from scipy.stats import randint, uniform
        if model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            mdl = RandomForestClassifier(random_state=seed) if is_clf else RandomForestRegressor(random_state=seed)
            param_dist = {"n_estimators": randint(50, 300), "max_depth": [None, 3, 5, 10, 20],
                          "min_samples_split": randint(2, 10), "max_features": ["sqrt", "log2", None]}
        elif model_name == "gb":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            mdl = GradientBoostingClassifier(random_state=seed) if is_clf else GradientBoostingRegressor(random_state=seed)
            param_dist = {"n_estimators": randint(50, 300), "learning_rate": uniform(0.01, 0.3),
                          "max_depth": randint(2, 8), "subsample": uniform(0.6, 0.4)}
        elif model_name == "svm":
            from sklearn.svm import SVC, SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            scaler = StandardScaler()
            base = SVC(random_state=seed, probability=True) if is_clf else SVR()
            mdl = Pipeline([("sc", scaler), ("svm", base)])
            param_dist = {"svm__C": uniform(0.01, 100), "svm__kernel": ["rbf", "linear"],
                          "svm__gamma": ["scale", "auto"]}
        elif model_name in ("logit", "ridge"):
            from sklearn.linear_model import LogisticRegression, Ridge, Lasso
            if is_clf:
                mdl = LogisticRegression(max_iter=500, random_state=seed)
                param_dist = {"C": uniform(0.001, 10), "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]}
            else:
                mdl = Ridge(random_state=seed)
                param_dist = {"alpha": uniform(0.001, 10)}
        elif model_name == "lasso":
            from sklearn.linear_model import Lasso
            mdl = Lasso(random_state=seed)
            param_dist = {"alpha": uniform(0.001, 10)}
        else:
            return f"Unknown model: {model_name}. Use rf, gb, svm, logit, ridge, lasso."

        search = RandomizedSearchCV(
            mdl, param_distributions=param_dist,
            n_iter=n_iter, cv=min(cv_k, len(y)),
            scoring=metric, random_state=seed, n_jobs=-1,
        )
        search.fit(X, y)

        best_params = search.best_params_
        lines = [
            f"Hyperparameter Optimization: {y_col} ~ {' + '.join(x_cols)}",
            f"Model: {model_name.upper()}  Task: {'Classification' if is_clf else 'Regression'}",
            f"Search: RandomSearch({n_iter} iterations)  CV={min(cv_k, len(y))}  Metric={metric}",
            "=" * 60,
            f"  Best score: {search.best_score_:.4f}",
            "",
            "  Best parameters:",
        ]
        for k_p, v in sorted(best_params.items()):
            lines.append(f"    {k_p:<30} {v}")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "hyperopt")


# ── SHAP values ──────────────────────────────────────────────────────────────

@command("shap", usage="shap <y> <x1> [x2 ...] [--model=rf|gb|linear] [--plot]")
def cmd_shap(session: Session, args: str) -> str:
    """SHAP (SHapley Additive exPlanations) feature importance.

    Computes SHAP values to explain model predictions.
    Works with tree models (RF, GB) and linear models.

    Options:
      --model=rf|gb|linear   model type (default: rf)
      --plot                 save SHAP summary plot
      --n_samples=N          max samples for SHAP (default: 500)

    Examples:
      shap income educ age female --model=rf --plot
      shap y x1 x2 x3 --model=gb
    """
    try:
        import shap as _shap
    except ImportError:
        return "shap required. Install: pip install shap"
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    ca = CommandArgs(args)
    preds = [p for p in ca.positional if not p.startswith("-")]
    if len(preds) < 2:
        return "Usage: shap <y> <x1> [x2 ...] [--model=rf]"

    y_col = preds[0]
    x_cols = preds[1:]
    model_name = ca.options.get("model", "rf").lower()
    make_plot = "--plot" in args
    n_samples = int(ca.options.get("n_samples", 500))
    seed = int(ca.options.get("seed", getattr(session, "_repro_seed", 42) or 42))

    try:
        df = session.require_data()
        sub = df.select([y_col] + x_cols).drop_nulls()
        y = sub[y_col].to_numpy().astype(float)
        X = sub.select(x_cols).to_numpy().astype(float)

        # Use subset for SHAP if large
        if len(X) > n_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X_shap, y_shap = X[idx], y[idx]
        else:
            X_shap, y_shap = X, y

        if model_name == "rf":
            mdl = RandomForestRegressor(n_estimators=100, random_state=seed)
            mdl.fit(X, y)
            explainer = _shap.TreeExplainer(mdl)
        elif model_name == "gb":
            mdl = GradientBoostingRegressor(n_estimators=100, random_state=seed)
            mdl.fit(X, y)
            explainer = _shap.TreeExplainer(mdl)
        else:  # linear
            mdl = LinearRegression()
            mdl.fit(X, y)
            explainer = _shap.LinearExplainer(mdl, X_shap)

        shap_values = explainer.shap_values(X_shap)
        mean_abs = np.abs(shap_values).mean(axis=0)

        lines = [
            f"SHAP Feature Importance: {y_col} ~ {' + '.join(x_cols)}",
            f"Model: {model_name.upper()}  N={len(X_shap)} samples",
            "=" * 50,
            f"  {'Feature':<25} {'Mean |SHAP|':>12}",
            "-" * 50,
        ]
        for name, val in sorted(zip(x_cols, mean_abs), key=lambda t: -t[1]):
            bar = "█" * int(val / mean_abs.max() * 20)
            lines.append(f"  {name:<25} {val:12.4f}  {bar}")

        if make_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, max(4, len(x_cols) * 0.5 + 1)))
            order = np.argsort(mean_abs)
            ax.barh([x_cols[i] for i in order], mean_abs[order], color="#4C72B0")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance ({model_name.upper()})")
            fig.tight_layout()
            from pathlib import Path
            session.output_dir.mkdir(parents=True, exist_ok=True)
            path = session.output_dir / "shap_importance.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            session.plot_paths.append(str(path))
            lines.append(f"\nSHAP plot saved: {path}")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "shap")


# ── Learning curve ───────────────────────────────────────────────────────────

@command("learncurve", usage="learncurve <y> <x1> [x2 ...] [--model=ols|rf|logit]")
def cmd_learncurve(session: Session, args: str) -> str:
    """Plot learning curve: training and CV score vs training set size.

    Helps diagnose bias/variance tradeoff.

    Options:
      --model=ols|rf|logit|gb  model (default: ols for continuous y, logit for binary)
      --cv=K                   cross-validation folds (default: 5)
      --steps=N                number of training size steps (default: 10)

    Examples:
      learncurve income educ age
      learncurve employed educ age female --model=logit
    """
    try:
        from sklearn.model_selection import learning_curve
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ca = CommandArgs(args)
    preds = [p for p in ca.positional if not p.startswith("-")]
    if len(preds) < 2:
        return "Usage: learncurve <y> <x1> [x2 ...]"

    y_col = preds[0]
    x_cols = preds[1:]
    cv_k = int(ca.options.get("cv", 5))
    steps = int(ca.options.get("steps", 10))
    seed = int(ca.options.get("seed", getattr(session, "_repro_seed", 42) or 42))

    try:
        df = session.require_data()
        sub = df.select([y_col] + x_cols).drop_nulls()
        y = sub[y_col].to_numpy().astype(float)
        X = sub.select(x_cols).to_numpy().astype(float)

        model_opt = ca.options.get("model", "")
        n_uniq = len(set(y))
        is_clf = (n_uniq <= 10 and (y == y.astype(int)).all()) if not model_opt else ("logit" in model_opt or "class" in model_opt)

        mdl_map = {
            "ols": LinearRegression(), "linear": LinearRegression(),
            "logit": LogisticRegression(max_iter=500, random_state=seed),
            "rf": RandomForestClassifier(random_state=seed) if is_clf else RandomForestRegressor(random_state=seed),
            "gb": GradientBoostingRegressor(random_state=seed),
        }
        mdl = mdl_map.get(model_opt, LogisticRegression(max_iter=500, random_state=seed) if is_clf else LinearRegression())
        metric = "accuracy" if is_clf else "r2"

        train_sizes = np.linspace(0.1, 1.0, steps)
        ts, train_scores, cv_scores = learning_curve(
            mdl, X, y, cv=min(cv_k, len(y)), scoring=metric,
            train_sizes=train_sizes, random_state=seed,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(ts, train_scores.mean(1) - train_scores.std(1),
                        train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#4C72B0")
        ax.plot(ts, train_scores.mean(1), "o-", color="#4C72B0", label="Train")
        ax.fill_between(ts, cv_scores.mean(1) - cv_scores.std(1),
                        cv_scores.mean(1) + cv_scores.std(1), alpha=0.15, color="#DD8452")
        ax.plot(ts, cv_scores.mean(1), "s-", color="#DD8452", label="CV")
        ax.set_xlabel("Training set size")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Learning Curve: {y_col}")
        ax.legend()
        fig.tight_layout()

        session.output_dir.mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        path = session.output_dir / "learning_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        session.plot_paths.append(str(path))

        final_cv = cv_scores.mean(1)[-1]
        final_train = train_scores.mean(1)[-1]
        gap = final_train - final_cv
        diagnosis = (
            "Possible overfitting (high variance)" if gap > 0.1 else
            "Possible underfitting (high bias)" if final_cv < 0.5 else
            "Good fit"
        )

        return (
            f"Learning Curve: {y_col} ~ {' + '.join(x_cols)}\n"
            f"Final train {metric}: {final_train:.4f}  |  "
            f"CV {metric}: {final_cv:.4f}  |  Gap: {gap:.4f}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Plot saved: {path}"
        )
    except Exception as e:
        return friendly_error(e, "learncurve")
