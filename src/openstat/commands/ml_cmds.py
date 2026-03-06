"""Machine learning commands: lasso, ridge, elasticnet, cart, crossval."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


def _parse_varlist(args: str, df_cols: list[str]) -> tuple[str, list[str], dict]:
    """Parse 'dep indeps [, opts]' from raw args."""
    positional, opts = _stata_opts(args)
    dep = positional[0] if positional else ""
    indeps = [c for c in positional[1:] if c in df_cols]
    return dep, indeps, opts


def _ml_table(result: dict) -> str:
    method = result.get("method", "")
    dep = result.get("dep", "")
    lines = [f"\n{method}: {dep}", "=" * 55]
    coef = result.get("coefficients", {})
    if coef:
        lines.append(f"  {'Variable':<25}  {'Coefficient':>12}")
        lines.append("  " + "-" * 40)
        for var, val in coef.items():
            lines.append(f"  {var:<25}  {val:>12.6f}")
        lines.append("")
    for key in ("intercept", "alpha", "l1_ratio", "r_squared", "mse", "rmse",
                "n_obs", "n_nonzero", "n_zeroed", "accuracy"):
        if key in result:
            v = result[key]
            lines.append(f"  {key:<25}  {v!s:>12}" if not isinstance(v, float)
                         else f"  {key:<25}  {v:>12.6f}")
    lines.append("=" * 55)
    return "\n".join(lines)


@command("lasso", usage="lasso depvar indepvars [, alpha(0.1) cv(5)]")
def cmd_lasso(session: Session, args: str) -> str:
    """Lasso regression with optional CV-tuned penalty."""
    df = session.require_data()
    dep, indeps, opts = _parse_varlist(args, df.columns)
    if not dep or not indeps:
        return "Usage: lasso depvar indepvar1 indepvar2 ... [, alpha(0.1) cv(5)]"
    alpha_raw = opts.get("alpha")
    alpha = float(alpha_raw) if alpha_raw else None
    cv = int(opts.get("cv", 5))
    try:
        from openstat.stats.ml import fit_lasso
        result = fit_lasso(df, dep, indeps, alpha=alpha, cv=cv)
        session._last_model = result
        return _ml_table(result)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"lasso error: {exc}"


@command("ridge", usage="ridge depvar indepvars [, alpha(1.0) cv(5)]")
def cmd_ridge(session: Session, args: str) -> str:
    """Ridge regression with optional CV-tuned penalty."""
    df = session.require_data()
    dep, indeps, opts = _parse_varlist(args, df.columns)
    if not dep or not indeps:
        return "Usage: ridge depvar indepvar1 ... [, alpha(1.0) cv(5)]"
    alpha_raw = opts.get("alpha")
    alpha = float(alpha_raw) if alpha_raw else None
    cv = int(opts.get("cv", 5))
    try:
        from openstat.stats.ml import fit_ridge
        result = fit_ridge(df, dep, indeps, alpha=alpha, cv=cv)
        session._last_model = result
        return _ml_table(result)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"ridge error: {exc}"


@command("elasticnet", usage="elasticnet depvar indepvars [, alpha(1.0) l1ratio(0.5)]")
def cmd_elasticnet(session: Session, args: str) -> str:
    """Elastic Net regression."""
    df = session.require_data()
    dep, indeps, opts = _parse_varlist(args, df.columns)
    if not dep or not indeps:
        return "Usage: elasticnet depvar indepvar1 ... [, alpha(1.0) l1ratio(0.5)]"
    alpha_raw = opts.get("alpha")
    alpha = float(alpha_raw) if alpha_raw else None
    l1 = float(opts.get("l1ratio", 0.5))
    cv = int(opts.get("cv", 5))
    try:
        from openstat.stats.ml import fit_elasticnet
        result = fit_elasticnet(df, dep, indeps, alpha=alpha, l1_ratio=l1, cv=cv)
        session._last_model = result
        return _ml_table(result)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"elasticnet error: {exc}"


@command("cart", usage="cart depvar indepvars [, depth(5) task(regression|classification)]")
def cmd_cart(session: Session, args: str) -> str:
    """CART decision tree (regression or classification)."""
    df = session.require_data()
    dep, indeps, opts = _parse_varlist(args, df.columns)
    if not dep or not indeps:
        return "Usage: cart depvar indepvar1 ... [, depth(5) task(regression)]"
    depth_raw = opts.get("depth")
    max_depth = int(depth_raw) if depth_raw else 5
    task = opts.get("task", "regression")
    min_leaf = int(opts.get("minleaf", 5))
    try:
        from openstat.stats.ml import fit_cart
        result = fit_cart(df, dep, indeps, task=task, max_depth=max_depth,
                          min_samples_leaf=min_leaf)
        session._last_model = result
        lines = [f"\nCART ({task}): {dep}", "=" * 55]
        lines.append(f"  {'max_depth':<25}  {max_depth!s:>12}")
        lines.append(f"  {'n_leaves':<25}  {result['n_leaves']:>12}")
        lines.append(f"  {'n_obs':<25}  {result['n_obs']:>12}")
        metric = "r_squared" if task == "regression" else "accuracy"
        lines.append(f"  {metric:<25}  {result[metric]:>12.4f}")
        lines.append("\nFeature Importances:")
        for feat, imp in sorted(result["feature_importances"].items(),
                                key=lambda x: -x[1]):
            lines.append(f"  {feat:<25}  {imp:>12.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"cart error: {exc}"


@command("crossval", usage="crossval depvar indepvars [, method(ols) k(5) scoring(r2)]")
def cmd_crossval(session: Session, args: str) -> str:
    """K-fold cross-validation for regression models."""
    df = session.require_data()
    dep, indeps, opts = _parse_varlist(args, df.columns)
    if not dep or not indeps:
        return "Usage: crossval depvar indepvar1 ... [, method(ols) k(5) scoring(r2)]"
    method = opts.get("method", "ols")
    k = int(opts.get("k", 5))
    alpha = float(opts.get("alpha", 1.0))
    scoring = opts.get("scoring", "r2")
    try:
        from openstat.stats.ml import cross_validate_model
        result = cross_validate_model(df, dep, indeps, method=method, k=k,
                                      alpha=alpha, scoring=scoring)
        lines = [f"\nCross-Validation ({k}-fold): {method}", "=" * 55]
        lines.append(f"  {'Dependent':<25}  {dep}")
        lines.append(f"  {'Scoring':<25}  {scoring}")
        lines.append(f"  {'Mean score':<25}  {result['mean_score']:>12.4f}")
        lines.append(f"  {'Std score':<25}  {result['std_score']:>12.4f}")
        lines.append(f"  {'Min score':<25}  {result['min_score']:>12.4f}")
        lines.append(f"  {'Max score':<25}  {result['max_score']:>12.4f}")
        lines.append(f"  {'N obs':<25}  {result['n_obs']:>12}")
        lines.append("\nFold scores:")
        for i, s in enumerate(result["scores"], 1):
            lines.append(f"  Fold {i:<3}  {s:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"crossval error: {exc}"
