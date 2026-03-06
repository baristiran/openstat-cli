"""Model evaluation: ROC/AUC, confusion matrix, calibration, SHAP approximation."""

from __future__ import annotations

import numpy as np
import polars as pl


def roc_auc(
    df: pl.DataFrame,
    outcome: str,
    score: str,
) -> dict:
    """Compute ROC curve and AUC (trapezoidal rule)."""
    sub = df.select([outcome, score]).drop_nulls()
    y_true = sub[outcome].to_numpy().astype(int)
    y_score = sub[score].to_numpy().astype(float)

    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr_list = []
    fpr_list = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))

    fpr_arr = np.array([0.0] + fpr_list + [1.0])
    tpr_arr = np.array([0.0] + tpr_list + [1.0])
    auc = float(np.trapezoid(tpr_arr, fpr_arr))

    # Youden J statistic → optimal threshold
    j = tpr_arr - fpr_arr
    opt_idx = int(np.argmax(j))
    opt_threshold = float(thresholds[max(opt_idx - 1, 0)]) if opt_idx > 0 else float(thresholds[0])

    return {
        "test": "ROC / AUC",
        "outcome": outcome,
        "score": score,
        "auc": auc,
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
        "thresholds": thresholds.tolist(),
        "optimal_threshold": opt_threshold,
        "n_obs": len(y_true),
        "prevalence": float(y_true.mean()),
    }


def confusion_matrix(
    df: pl.DataFrame,
    outcome: str,
    predicted: str,
    threshold: float = 0.5,
) -> dict:
    """Compute confusion matrix and classification metrics."""
    sub = df.select([outcome, predicted]).drop_nulls()
    y_true = sub[outcome].to_numpy().astype(int)
    y_score = sub[predicted].to_numpy().astype(float)

    # If predicted is already binary (0/1), don't threshold
    if set(np.unique(y_score)).issubset({0, 1, 0.0, 1.0}):
        y_pred = y_score.astype(int)
    else:
        y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    npv = tn / max(tn + fn, 1)
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = float(mcc_num / mcc_den)

    return {
        "test": "Confusion Matrix",
        "outcome": outcome, "predicted": predicted,
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "npv": float(npv),
        "mcc": mcc,
        "n_obs": len(y_true),
    }


def calibration_curve(
    df: pl.DataFrame,
    outcome: str,
    score: str,
    n_bins: int = 10,
) -> dict:
    """Calibration curve (reliability diagram) + Brier score."""
    sub = df.select([outcome, score]).drop_nulls()
    y_true = sub[outcome].to_numpy().astype(int)
    y_score = sub[score].to_numpy().astype(float)

    brier = float(np.mean((y_score - y_true) ** 2))

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    mean_predicted = []
    fraction_positive = []

    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_score >= bins[i]) & (y_score <= bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
            mean_predicted.append(float(y_score[mask].mean()))
            fraction_positive.append(float(y_true[mask].mean()))

    # Hosmer-Lemeshow test approximation
    expected = np.array(mean_predicted) * np.array([
        int(((y_score >= bins[i]) & (y_score < bins[i + 1])).sum())
        for i in range(n_bins)
        if ((y_score >= bins[i]) & (y_score < bins[i + 1])).sum() > 0
    ])

    return {
        "test": "Calibration Curve",
        "outcome": outcome, "score": score,
        "brier_score": brier,
        "n_bins": n_bins,
        "bin_centers": bin_centers,
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "n_obs": len(y_true),
    }


def compute_shap_linear(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
) -> dict:
    """
    Linear SHAP values for OLS regression.
    SHAP_i(x) = beta_i * (x_i - E[x_i])
    Exact for linear models.
    """
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    x_means = X_raw.mean(axis=0)

    # SHAP values per observation per feature
    shap_vals = (X_raw - x_means) * beta[1:]  # shape (n, k)

    mean_abs_shap = {col: float(np.abs(shap_vals[:, i]).mean()) for i, col in enumerate(indeps)}
    sorted_imp = sorted(mean_abs_shap.items(), key=lambda x: -x[1])

    return {
        "method": "Linear SHAP",
        "dep": dep,
        "indeps": indeps,
        "n_obs": n,
        "mean_abs_shap": mean_abs_shap,
        "feature_ranking": [f for f, _ in sorted_imp],
        "shap_values": shap_vals.tolist(),
        "coefficients": {col: float(beta[i + 1]) for i, col in enumerate(indeps)},
        "intercept": float(beta[0]),
    }
