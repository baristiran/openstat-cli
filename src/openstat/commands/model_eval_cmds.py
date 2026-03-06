"""Model evaluation commands: roc, confmatrix, calibration, shap."""

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


@command("roc", usage="roc outcome score_col")
def cmd_roc(session: Session, args: str) -> str:
    """ROC curve and AUC for binary classification."""
    from openstat.stats.model_eval import roc_auc
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: roc outcome score_col"
    outcome, score = positional[0], positional[1]
    for v in (outcome, score):
        if v not in df.columns:
            return f"Column '{v}' not found."
    try:
        r = roc_auc(df, outcome, score)
        session._last_model = r
        lines = ["\nROC Analysis", "=" * 50]
        lines.append(f"  {'Outcome':<30} {outcome}")
        lines.append(f"  {'Score':<30} {score}")
        lines.append(f"  {'AUC':<30} {r['auc']:.4f}")
        lines.append(f"  {'Optimal threshold':<30} {r['optimal_threshold']:.4f}")
        lines.append(f"  {'N observations':<30} {r['n_obs']}")
        lines.append(f"  {'Prevalence':<30} {r['prevalence']:.4f}")
        lines.append(f"\n  Interpretation:")
        auc = r['auc']
        if auc >= 0.9:
            interp = "Excellent (≥0.90)"
        elif auc >= 0.8:
            interp = "Good (0.80–0.89)"
        elif auc >= 0.7:
            interp = "Fair (0.70–0.79)"
        elif auc >= 0.6:
            interp = "Poor (0.60–0.69)"
        else:
            interp = "Fail (<0.60)"
        lines.append(f"  {'AUC interpretation':<30} {interp}")
        return "\n".join(lines)
    except Exception as exc:
        return f"roc error: {exc}"


@command("confmatrix", usage="confmatrix outcome predicted [threshold(0.5)]")
def cmd_confmatrix(session: Session, args: str) -> str:
    """Confusion matrix and classification metrics."""
    from openstat.stats.model_eval import confusion_matrix
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: confmatrix outcome predicted [threshold(0.5)]"
    outcome, predicted = positional[0], positional[1]
    for v in (outcome, predicted):
        if v not in df.columns:
            return f"Column '{v}' not found."
    threshold = float(opts.get("threshold", 0.5))
    try:
        r = confusion_matrix(df, outcome, predicted, threshold=threshold)
        lines = ["\nConfusion Matrix", "=" * 50]
        lines.append(f"\n  Predicted →    Positive   Negative")
        lines.append(f"  Actual Positive  {r['tp']:>8}   {r['fn']:>8}")
        lines.append(f"  Actual Negative  {r['fp']:>8}   {r['tn']:>8}")
        lines.append("\n  Metrics:")
        for k in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "npv", "mcc"]:
            lines.append(f"  {k.replace('_', ' ').title():<30} {r[k]:.4f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"confmatrix error: {exc}"


@command("calibration", usage="calibration outcome score [bins(10)]")
def cmd_calibration(session: Session, args: str) -> str:
    """Calibration curve (reliability diagram) and Brier score."""
    from openstat.stats.model_eval import calibration_curve
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: calibration outcome score [bins(10)]"
    outcome, score = positional[0], positional[1]
    for v in (outcome, score):
        if v not in df.columns:
            return f"Column '{v}' not found."
    n_bins = int(opts.get("bins", 10))
    try:
        r = calibration_curve(df, outcome, score, n_bins=n_bins)
        lines = ["\nCalibration Analysis", "=" * 55]
        lines.append(f"  {'Brier score':<35} {r['brier_score']:.6f}")
        lines.append(f"  {'N observations':<35} {r['n_obs']}")
        lines.append(f"\n  {'Bin Center':>10}  {'Mean Pred':>10}  {'Frac Pos':>10}")
        lines.append("  " + "-" * 35)
        for bc, mp, fp in zip(r["bin_centers"], r["mean_predicted"], r["fraction_positive"]):
            lines.append(f"  {bc:>10.3f}  {mp:>10.3f}  {fp:>10.3f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"calibration error: {exc}"


# Backward-compat alias — cmd_shap moved to advanced_ml_cmds
from openstat.commands.advanced_ml_cmds import cmd_shap  # noqa: F401
