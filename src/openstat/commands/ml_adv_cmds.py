"""Advanced ML commands: randomforest, gbm, svm, tsne."""

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


def _fmt_imp(feat_imp: dict, top: int = 10) -> str:
    sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])[:top]
    lines = ["\n  Feature Importances:"]
    for feat, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        lines.append(f"    {feat:<20} {imp:.4f} {bar}")
    return "\n".join(lines)


@command("randomforest", usage="randomforest dep var1 var2 ... [n(100) depth(none) task(regression)]")
def cmd_randomforest(session: Session, args: str) -> str:
    """Random Forest regression or classification."""
    try:
        import sklearn
    except ImportError:
        return "sklearn not installed. Run: pip install scikit-learn"
    from openstat.stats.ml_advanced import fit_random_forest
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: randomforest dep var1 [var2 ...] [n(100) depth(5) task(regression)]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    n_est = int(opts.get("n", 100))
    max_depth = int(opts["depth"]) if "depth" in opts else None
    task = opts.get("task", "regression")
    try:
        r = fit_random_forest(df, dep, indeps, n_estimators=n_est, max_depth=max_depth, task=task)
        session._last_model = r
        metric = "r_squared" if task == "regression" else "accuracy"
        lines = [f"\nRandom Forest ({task})", "=" * 50]
        lines.append(f"  Dep: {dep}, N: {r['n_obs']}, Trees: {n_est}")
        lines.append(f"  {metric.replace('_', ' ').title()}: {r.get(metric, 'N/A'):.4f}")
        lines.append(_fmt_imp(r["feature_importances"]))
        return "\n".join(lines)
    except Exception as exc:
        return f"randomforest error: {exc}"


@command("gbm", usage="gbm dep var1 var2 ... [n(100) lr(0.1) depth(3) task(regression)]")
def cmd_gbm(session: Session, args: str) -> str:
    """Gradient Boosting Machine."""
    try:
        import sklearn
    except ImportError:
        return "sklearn not installed. Run: pip install scikit-learn"
    from openstat.stats.ml_advanced import fit_gradient_boosting
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: gbm dep var1 [var2 ...] [n(100) lr(0.1) depth(3) task(regression)]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    n_est = int(opts.get("n", 100))
    lr = float(opts.get("lr", 0.1))
    depth = int(opts.get("depth", 3))
    task = opts.get("task", "regression")
    try:
        r = fit_gradient_boosting(df, dep, indeps, n_estimators=n_est, learning_rate=lr, max_depth=depth, task=task)
        session._last_model = r
        metric = "r_squared" if task == "regression" else "accuracy"
        lines = [f"\nGradient Boosting ({task})", "=" * 50]
        lines.append(f"  Dep: {dep}, N: {r['n_obs']}, Trees: {n_est}, LR: {lr}")
        lines.append(f"  {metric.replace('_', ' ').title()}: {r.get(metric, 'N/A'):.4f}")
        lines.append(_fmt_imp(r["feature_importances"]))
        return "\n".join(lines)
    except Exception as exc:
        return f"gbm error: {exc}"


@command("svm", usage="svm dep var1 var2 ... [kernel(rbf) C(1.0) task(regression)]")
def cmd_svm(session: Session, args: str) -> str:
    """Support Vector Machine."""
    try:
        import sklearn
    except ImportError:
        return "sklearn not installed. Run: pip install scikit-learn"
    from openstat.stats.ml_advanced import fit_svm
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: svm dep var1 [var2 ...] [kernel(rbf) C(1.0) task(regression)]"
    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    kernel = opts.get("kernel", "rbf")
    C = float(opts.get("c", 1.0))
    task = opts.get("task", "regression")
    try:
        r = fit_svm(df, dep, indeps, kernel=kernel, C=C, task=task)
        session._last_model = r
        metric = "r_squared" if task == "regression" else "accuracy"
        lines = [f"\nSVM ({task}, kernel={kernel})", "=" * 50]
        lines.append(f"  Dep: {dep}, N: {r['n_obs']}, C: {C}")
        lines.append(f"  {metric.replace('_', ' ').title()}: {r.get(metric, 'N/A'):.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"svm error: {exc}"


# Backward-compat alias — cmd_tsne moved to dimreduce_cmds
from openstat.commands.dimreduce_cmds import cmd_tsne  # noqa: F401
