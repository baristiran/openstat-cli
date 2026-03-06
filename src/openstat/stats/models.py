"""Statistical models: OLS, Logistic regression, hypothesis tests.

Includes diagnostics:
- Multicollinearity detection (condition number)
- Convergence check (logit)
- Minimum observations check
- Stepwise variable selection (forward/backward)
- Residual diagnostics

Hypothesis tests:
- t-test (one-sample, two-sample, paired)
- Chi-square test of independence
- One-way ANOVA
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as sp_stats
from rich.table import Table
from rich.console import Console

from openstat.config import get_config


@dataclass
class FitResult:
    """Holds a fitted model result for reporting."""

    model_type: str  # "OLS" or "Logit"
    formula: str
    dep_var: str
    indep_vars: list[str]
    n_obs: int
    params: dict[str, float]
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    conf_int_low: dict[str, float]
    conf_int_high: dict[str, float]
    r_squared: float | None = None  # OLS only
    adj_r_squared: float | None = None  # OLS only
    f_statistic: float | None = None  # OLS only
    f_pvalue: float | None = None  # OLS only
    aic: float | None = None
    bic: float | None = None
    pseudo_r2: float | None = None  # Logit/Probit only
    log_likelihood: float | None = None
    dispersion: float | None = None  # Negative Binomial alpha
    warnings: list[str] = field(default_factory=list)

    def summary_table(self) -> str:
        """Return a Rich-formatted summary table as string."""
        console = Console(file=io.StringIO(), width=100, record=True)
        table = Table(title=f"{self.model_type}: {self.formula}")
        table.add_column("Variable", style="cyan")
        table.add_column("Coef", justify="right")
        table.add_column("Std.Err", justify="right")
        table.add_column("t/z", justify="right")
        table.add_column("P>|t|", justify="right")
        table.add_column("[95% CI Low]", justify="right")
        table.add_column("[95% CI High]", justify="right")

        for var in self.params:
            sig = ""
            pv = self.p_values[var]
            if pv < 0.001:
                sig = " ***"
            elif pv < 0.01:
                sig = " **"
            elif pv < 0.05:
                sig = " *"

            table.add_row(
                var,
                f"{self.params[var]:.4f}",
                f"{self.std_errors[var]:.4f}",
                f"{self.t_values[var]:.3f}",
                f"{pv:.4f}{sig}",
                f"{self.conf_int_low[var]:.4f}",
                f"{self.conf_int_high[var]:.4f}",
            )

        console.print(table)
        header = f"N = {self.n_obs}"
        if self.r_squared is not None:
            header += f"  |  R² = {self.r_squared:.4f}"
        if self.adj_r_squared is not None:
            header += f"  |  Adj.R² = {self.adj_r_squared:.4f}"
        if self.f_statistic is not None and self.f_pvalue is not None:
            k = len(self.indep_vars)
            header += f"  |  F({k}, {self.n_obs - k - 1}) = {self.f_statistic:.2f} (p={self.f_pvalue:.4f})"
        if self.aic is not None:
            header += f"  |  AIC = {self.aic:.1f}"
        if self.bic is not None:
            header += f"  |  BIC = {self.bic:.1f}"
        if self.pseudo_r2 is not None:
            header += f"  |  Pseudo R² = {self.pseudo_r2:.4f}"
        if self.log_likelihood is not None:
            header += f"  |  LL = {self.log_likelihood:.1f}"
        if self.dispersion is not None:
            header += f"  |  alpha = {self.dispersion:.4f}"
        console.print(header)
        console.print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
        return console.export_text()

    def to_markdown(self) -> str:
        """Return a Markdown-formatted summary."""
        lines = [
            f"### {self.model_type}: {self.formula}",
            "",
            f"N = {self.n_obs}",
        ]
        if self.r_squared is not None:
            lines.append(f"R² = {self.r_squared:.4f}")
        if self.adj_r_squared is not None:
            lines.append(f"Adj. R² = {self.adj_r_squared:.4f}")
        if self.f_statistic is not None and self.f_pvalue is not None:
            lines.append(f"F-statistic = {self.f_statistic:.2f} (p = {self.f_pvalue:.4f})")
        if self.aic is not None:
            lines.append(f"AIC = {self.aic:.1f}")
        if self.bic is not None:
            lines.append(f"BIC = {self.bic:.1f}")
        if self.pseudo_r2 is not None:
            lines.append(f"Pseudo R² = {self.pseudo_r2:.4f}")
        if self.log_likelihood is not None:
            lines.append(f"Log-Likelihood = {self.log_likelihood:.1f}")
        if self.dispersion is not None:
            lines.append(f"Dispersion (alpha) = {self.dispersion:.4f}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"> {w}")
        lines.append("")
        lines.append("| Variable | Coef | Std.Err | t/z | P>\\|t\\| | 95% CI |")
        lines.append("|----------|------|---------|-----|---------|--------|")
        for var in self.params:
            ci = f"[{self.conf_int_low[var]:.4f}, {self.conf_int_high[var]:.4f}]"
            lines.append(
                f"| {var} | {self.params[var]:.4f} | {self.std_errors[var]:.4f} "
                f"| {self.t_values[var]:.3f} | {self.p_values[var]:.4f} | {ci} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Return a LaTeX-formatted regression table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.model_type}: {self.formula}}}",
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"Variable & Coef & Std.Err & t/z & P$>|$t$|$ & [95\% CI Low] & [95\% CI High] \\",
            r"\hline",
        ]
        for var in self.params:
            sig = ""
            pv = self.p_values[var]
            if pv < 0.001:
                sig = "$^{***}$"
            elif pv < 0.01:
                sig = "$^{**}$"
            elif pv < 0.05:
                sig = "$^{*}$"
            # Escape underscores in variable names for LaTeX
            var_tex = var.replace("_", r"\_")
            lines.append(
                f"{var_tex} & {self.params[var]:.4f}{sig} & "
                f"{self.std_errors[var]:.4f} & {self.t_values[var]:.3f} & "
                f"{self.p_values[var]:.4f} & {self.conf_int_low[var]:.4f} & "
                f"{self.conf_int_high[var]:.4f} \\\\"
            )
        lines.append(r"\hline")
        footer = f"N = {self.n_obs}"
        if self.r_squared is not None:
            footer += f", $R^2$ = {self.r_squared:.4f}"
        if self.adj_r_squared is not None:
            footer += f", Adj.$R^2$ = {self.adj_r_squared:.4f}"
        if self.f_statistic is not None and self.f_pvalue is not None:
            footer += f", F = {self.f_statistic:.2f} (p = {self.f_pvalue:.4f})"
        if self.aic is not None:
            footer += f", AIC = {self.aic:.1f}"
        if self.bic is not None:
            footer += f", BIC = {self.bic:.1f}"
        if self.pseudo_r2 is not None:
            footer += f", Pseudo $R^2$ = {self.pseudo_r2:.4f}"
        if self.log_likelihood is not None:
            footer += f", LL = {self.log_likelihood:.1f}"
        if self.dispersion is not None:
            footer += f", $\\alpha$ = {self.dispersion:.4f}"
        lines.append(f"\\multicolumn{{7}}{{l}}{{{footer}}} \\\\")
        lines.append(r"\hline")
        lines.append(r"\multicolumn{7}{l}{\footnotesize $^{*}$p$<$0.05, $^{**}$p$<$0.01, $^{***}$p$<$0.001} \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def to_html(self) -> str:
        """Return HTML-formatted summary for Jupyter display."""
        console = Console(file=io.StringIO(), width=120, record=True)
        table = Table(title=f"{self.model_type}: {self.formula}")
        table.add_column("Variable", style="cyan")
        table.add_column("Coef", justify="right")
        table.add_column("Std.Err", justify="right")
        table.add_column("t/z", justify="right")
        table.add_column("P>|t|", justify="right")
        table.add_column("[95% CI Low]", justify="right")
        table.add_column("[95% CI High]", justify="right")
        for var in self.params:
            sig = ""
            pv = self.p_values[var]
            if pv < 0.001:
                sig = " ***"
            elif pv < 0.01:
                sig = " **"
            elif pv < 0.05:
                sig = " *"
            table.add_row(
                var,
                f"{self.params[var]:.4f}",
                f"{self.std_errors[var]:.4f}",
                f"{self.t_values[var]:.3f}",
                f"{pv:.4f}{sig}",
                f"{self.conf_int_low[var]:.4f}",
                f"{self.conf_int_high[var]:.4f}",
            )
        console.print(table)
        return console.export_html(inline_styles=True)


def _prepare_data(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    cluster_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray | None]:
    """Extract y and X (with constant) as numpy arrays.

    Handles interaction terms (e.g. ``"x1:x2"`` in *indeps*) by creating
    product columns on the fly.

    Returns ``(y, X, warnings, var_names, cluster_groups)``.
    *var_names* is ``["_cons"] + expanded_indeps``.
    *cluster_groups* is ``None`` when *cluster_col* is not given.
    """
    # Separate base variables from interaction terms
    base_vars: list[str] = []
    interactions: list[str] = []
    for v in indeps:
        if ":" in v:
            interactions.append(v)
        else:
            base_vars.append(v)

    # Collect all raw columns needed
    all_base: set[str] = set(base_vars)
    for inter in interactions:
        all_base.update(inter.split(":"))

    cols_needed = [dep] + sorted(all_base)
    if cluster_col:
        if cluster_col not in df.columns:
            raise ValueError(f"Cluster column not found: {cluster_col}")
        cols_needed = list(dict.fromkeys(cols_needed + [cluster_col]))

    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    sub = df.select(cols_needed).drop_nulls()
    if sub.height == 0:
        raise ValueError("No observations after dropping missing values")

    n_dropped = df.height - sub.height
    warnings: list[str] = []
    if n_dropped > 0:
        warnings.append(
            f"Note: {n_dropped} observation(s) dropped due to missing values."
        )

    y = sub[dep].to_numpy().astype(float)

    # Build X: base vars first, then interaction columns
    X_parts = sub.select(base_vars).to_numpy().astype(float) if base_vars else np.empty((sub.height, 0))
    var_names_x: list[str] = list(base_vars)

    for inter in interactions:
        parts = inter.split(":")
        col = np.ones(sub.height)
        for p in parts:
            col = col * sub[p].to_numpy().astype(float)
        X_parts = np.column_stack([X_parts, col]) if X_parts.size > 0 else col.reshape(-1, 1)
        var_names_x.append(inter)

    # Check minimum observations
    n_params = len(var_names_x) + 1  # +1 for constant
    if sub.height < n_params + 2:
        raise ValueError(
            f"Too few observations ({sub.height}) for {n_params} parameters. "
            f"Need at least {n_params + 2}."
        )

    cfg = get_config()
    if sub.height < n_params * cfg.min_obs_per_predictor:
        warnings.append(
            f"Warning: Low observations-to-predictors ratio "
            f"({sub.height} obs / {n_params} params = {sub.height / n_params:.1f}). "
            f"Results may be unreliable."
        )

    X = sm.add_constant(X_parts)

    # Check multicollinearity via condition number
    try:
        cond = np.linalg.cond(X)
        if cond > cfg.condition_threshold:
            warnings.append(
                f"Warning: Possible multicollinearity detected "
                f"(condition number = {cond:.0f}, threshold = {cfg.condition_threshold}). "
                f"Consider removing correlated predictors."
            )
    except np.linalg.LinAlgError:
        warnings.append("Warning: Could not compute condition number.")

    # Cluster groups
    groups: np.ndarray | None = None
    if cluster_col:
        groups = sub[cluster_col].to_numpy()

    var_names = ["_cons"] + var_names_x
    return y, X, warnings, var_names, groups


def _cov_args(
    robust: bool, groups: np.ndarray | None
) -> tuple[str, dict]:
    """Return (cov_type, cov_kwds) for statsmodels fit()."""
    if groups is not None:
        return "cluster", {"groups": groups}
    if robust:
        return "HC1", {}
    return "nonrobust", {}


def _model_type_suffix(robust: bool, cluster: bool) -> str:
    if cluster:
        return " (cluster-robust)"
    if robust:
        return " (robust)"
    return ""


def fit_ols(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit an OLS regression. Returns (FitResult, raw_model)."""
    y, X, warnings, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )
    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    ci = model.conf_int()

    # Heteroscedasticity check (Breusch-Pagan)
    if not robust and groups is None:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
            if bp_pval < 0.05:
                warnings.append(
                    f"Warning: Heteroscedasticity detected (Breusch-Pagan p={bp_pval:.4f}). "
                    f"Consider using --robust for heteroscedasticity-robust standard errors."
                )
        except Exception:
            pass  # diagnostic failure should not block results

    # Autocorrelation check (Durbin-Watson)
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(model.resid)
        if dw < 1.5 or dw > 2.5:
            warnings.append(
                f"Warning: Possible autocorrelation (Durbin-Watson = {dw:.3f}). "
                f"Values far from 2.0 suggest serial correlation in residuals."
            )
    except Exception:
        pass

    suffix = _model_type_suffix(robust, groups is not None)
    result = FitResult(
        model_type="OLS" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue) if hasattr(model, "fvalue") and model.fvalue is not None else None,
        f_pvalue=float(model.f_pvalue) if hasattr(model, "f_pvalue") and model.f_pvalue is not None else None,
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings,
    )
    return result, model


def fit_logit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a logistic regression (binary). Returns (FitResult, raw_model)."""
    y, X, warnings, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    unique_y = set(y)
    if not unique_y.issubset({0.0, 1.0}):
        raise ValueError(
            f"Logit requires binary (0/1) dependent variable. "
            f"Found values: {sorted(unique_y)[:10]}"
        )

    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.Logit(y, X).fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds)

    # Check convergence
    if hasattr(model, "mle_retvals") and not model.mle_retvals.get("converged", True):
        warnings.append(
            "WARNING: Model did NOT converge. Results may be unreliable. "
            "Consider rescaling variables or reducing predictors."
        )

    ci = model.conf_int()
    suffix = _model_type_suffix(robust, groups is not None)

    result = FitResult(
        model_type="Logit" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        pseudo_r2=float(model.prsquared),
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings,
    )
    return result, model


def fit_probit(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a probit regression (binary). Returns (FitResult, raw_model)."""
    y, X, warnings, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    unique_y = set(y)
    if not unique_y.issubset({0.0, 1.0}):
        raise ValueError(
            f"Probit requires binary (0/1) dependent variable. "
            f"Found values: {sorted(unique_y)[:10]}"
        )

    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.Probit(y, X).fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds)

    if hasattr(model, "mle_retvals") and not model.mle_retvals.get("converged", True):
        warnings.append(
            "WARNING: Model did NOT converge. Results may be unreliable."
        )

    ci = model.conf_int()
    suffix = _model_type_suffix(robust, groups is not None)

    result = FitResult(
        model_type="Probit" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        pseudo_r2=float(model.prsquared),
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings,
    )
    return result, model


def fit_poisson(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
    exposure_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Poisson regression. Returns (FitResult, raw_model)."""
    # Pre-filter: drop rows where exposure is null so _prepare_data and
    # offset computation use the same subset of rows.
    if exposure_col:
        if exposure_col not in df.columns:
            raise ValueError(f"Exposure column not found: {exposure_col}")
        df = df.filter(pl.col(exposure_col).is_not_null())

    y, X, warnings, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    # Handle exposure (offset = log(exposure))
    offset = None
    if exposure_col:
        # Reconstruct the same subset as _prepare_data (exposure nulls already
        # removed above, so drop_nulls here matches _prepare_data exactly).
        base_vars = [v for v in indeps if ":" not in v]
        all_base: set[str] = set(base_vars)
        for v in indeps:
            if ":" in v:
                all_base.update(v.split(":"))
        cols_needed = [dep] + sorted(all_base) + [exposure_col]
        if cluster_col:
            cols_needed = list(dict.fromkeys(cols_needed + [cluster_col]))
        sub = df.select(list(dict.fromkeys(cols_needed))).drop_nulls()
        offset = np.log(sub[exposure_col].to_numpy().astype(float))

    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.Poisson(y, X, offset=offset).fit(
        disp=0, cov_type=cov_type, cov_kwds=cov_kwds,
    )

    if hasattr(model, "mle_retvals") and not model.mle_retvals.get("converged", True):
        warnings.append("WARNING: Model did NOT converge. Results may be unreliable.")

    ci = model.conf_int()
    suffix = _model_type_suffix(robust, groups is not None)

    result = FitResult(
        model_type="Poisson" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        pseudo_r2=float(model.prsquared) if hasattr(model, "prsquared") else None,
        log_likelihood=float(model.llf),
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings,
    )
    return result, model


def fit_negbin(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    robust: bool = False,
    cluster_col: str | None = None,
) -> tuple[FitResult, object]:
    """Fit a Negative Binomial regression. Returns (FitResult, raw_model)."""
    y, X, warnings, var_names, groups = _prepare_data(
        df, dep, indeps, cluster_col=cluster_col,
    )

    cov_type, cov_kwds = _cov_args(robust, groups)
    model = sm.NegativeBinomial(y, X).fit(
        disp=0, cov_type=cov_type, cov_kwds=cov_kwds,
    )

    if hasattr(model, "mle_retvals") and not model.mle_retvals.get("converged", True):
        warnings.append("WARNING: Model did NOT converge. Results may be unreliable.")

    ci = model.conf_int()
    suffix = _model_type_suffix(robust, groups is not None)

    # NegativeBinomial includes alpha as the last parameter; exclude it from coef table
    n_coefs = len(var_names)
    params_arr = model.params[:n_coefs]
    bse_arr = model.bse[:n_coefs]
    tvals_arr = model.tvalues[:n_coefs]
    pvals_arr = model.pvalues[:n_coefs]
    ci_arr = ci[:n_coefs]

    result = FitResult(
        model_type="NegBin" + suffix,
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, params_arr)),
        std_errors=dict(zip(var_names, bse_arr)),
        t_values=dict(zip(var_names, tvals_arr)),
        p_values=dict(zip(var_names, pvals_arr)),
        conf_int_low=dict(zip(var_names, ci_arr[:, 0])),
        conf_int_high=dict(zip(var_names, ci_arr[:, 1])),
        pseudo_r2=float(model.prsquared) if hasattr(model, "prsquared") else None,
        log_likelihood=float(model.llf),
        dispersion=float(model.params[-1]),  # alpha
        aic=float(model.aic),
        bic=float(model.bic),
        warnings=warnings,
    )
    return result, model


def fit_quantreg(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    *,
    tau: float = 0.5,
) -> tuple[FitResult, object]:
    """Fit a quantile regression. Returns (FitResult, raw_model)."""
    if not (0 < tau < 1):
        raise ValueError(f"tau must be between 0 and 1 (exclusive), got {tau}")
    y, X, warnings, var_names, _ = _prepare_data(df, dep, indeps)

    model = sm.QuantReg(y, X).fit(q=tau)
    ci = model.conf_int()

    result = FitResult(
        model_type=f"QuantReg(tau={tau})",
        formula=f"{dep} ~ {' + '.join(indeps)}",
        dep_var=dep,
        indep_vars=indeps,
        n_obs=int(model.nobs),
        params=dict(zip(var_names, model.params)),
        std_errors=dict(zip(var_names, model.bse)),
        t_values=dict(zip(var_names, model.tvalues)),
        p_values=dict(zip(var_names, model.pvalues)),
        conf_int_low=dict(zip(var_names, ci[:, 0])),
        conf_int_high=dict(zip(var_names, ci[:, 1])),
        pseudo_r2=float(model.prsquared) if hasattr(model, "prsquared") else None,
        warnings=warnings,
    )
    return result, model


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------

@dataclass
class MarginalEffectsResult:
    """Holds marginal effects for a discrete-choice model."""

    method: str  # "at_means" or "average"
    effects: dict[str, float]
    std_errors: dict[str, float]
    z_values: dict[str, float]
    p_values: dict[str, float]
    conf_int_low: dict[str, float]
    conf_int_high: dict[str, float]

    def summary_table(self) -> str:
        console = Console(file=io.StringIO(), width=100, record=True)
        table = Table(title=f"Marginal Effects ({self.method})")
        table.add_column("Variable", style="cyan")
        table.add_column("dy/dx", justify="right")
        table.add_column("Std.Err", justify="right")
        table.add_column("z", justify="right")
        table.add_column("P>|z|", justify="right")
        table.add_column("[95% CI Low]", justify="right")
        table.add_column("[95% CI High]", justify="right")

        for var in self.effects:
            pv = self.p_values[var]
            sig = ""
            if pv < 0.001:
                sig = " ***"
            elif pv < 0.01:
                sig = " **"
            elif pv < 0.05:
                sig = " *"
            table.add_row(
                var,
                f"{self.effects[var]:.6f}",
                f"{self.std_errors[var]:.6f}",
                f"{self.z_values[var]:.3f}",
                f"{pv:.4f}{sig}",
                f"{self.conf_int_low[var]:.6f}",
                f"{self.conf_int_high[var]:.6f}",
            )

        console.print(table)
        return console.export_text()


def compute_margins(
    raw_model: object, var_names: list[str], method: str = "average"
) -> MarginalEffectsResult:
    """Compute marginal effects from a logit/probit model.

    *var_names* should be the list of predictor names (excluding ``_cons``).
    *method*: ``"average"`` for average marginal effects, ``"means"`` for at-means.
    """
    at_map = {"average": "overall", "means": "mean"}
    at = at_map.get(method, "overall")

    mfx = raw_model.get_margeff(at=at)  # type: ignore[attr-defined]
    margeff = mfx.margeff
    margeff_se = mfx.margeff_se
    z_vals = margeff / margeff_se
    p_vals = mfx.pvalues
    ci = mfx.conf_int()

    # var_names without _cons
    names = [v for v in var_names if v != "_cons"]

    return MarginalEffectsResult(
        method=method,
        effects=dict(zip(names, margeff)),
        std_errors=dict(zip(names, margeff_se)),
        z_values=dict(zip(names, z_vals)),
        p_values=dict(zip(names, p_vals)),
        conf_int_low=dict(zip(names, ci[:, 0])),
        conf_int_high=dict(zip(names, ci[:, 1])),
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Holds bootstrap confidence interval results."""

    original_params: dict[str, float]
    boot_means: dict[str, float]
    boot_std: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    n_boot: int
    ci_level: float
    n_failed: int = 0

    def summary_table(self) -> str:
        console = Console(file=io.StringIO(), width=100, record=True)
        table = Table(title=f"Bootstrap Confidence Intervals ({self.n_boot} replications, {self.ci_level}%)")
        table.add_column("Variable", style="cyan")
        table.add_column("Original", justify="right")
        table.add_column("Boot.Mean", justify="right")
        table.add_column("Boot.SE", justify="right")
        table.add_column(f"[{self.ci_level}% CI Low]", justify="right")
        table.add_column(f"[{self.ci_level}% CI High]", justify="right")

        for var in self.original_params:
            table.add_row(
                var,
                f"{self.original_params[var]:.4f}",
                f"{self.boot_means[var]:.4f}",
                f"{self.boot_std[var]:.4f}",
                f"{self.ci_low[var]:.4f}",
                f"{self.ci_high[var]:.4f}",
            )

        console.print(table)
        if self.n_failed > 0:
            console.print(f"Note: {self.n_failed} bootstrap iteration(s) failed and were skipped.")
        return console.export_text()


def _boot_one_iter(args):
    """Run a single bootstrap iteration (for parallel execution)."""
    df, dep, indeps, fit_fn, fit_kwargs, seed = args
    sample = df.sample(n=df.height, with_replacement=True, seed=seed)
    try:
        r, _ = fit_fn(sample, dep, indeps, **fit_kwargs)
        return dict(r.params)
    except Exception:
        return None


def bootstrap_model(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    fit_fn,
    n_boot: int = 1000,
    ci: float = 95.0,
    **fit_kwargs,
) -> BootstrapResult:
    """Generic bootstrap for any model fit function.

    *fit_fn* must have signature ``(df, dep, indeps, **kwargs) -> (FitResult, raw_model)``.
    Uses thread-pool parallelism for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    original_result, _ = fit_fn(df, dep, indeps, **fit_kwargs)

    boot_params: dict[str, list[float]] = {var: [] for var in original_result.params}
    n_failed = 0

    max_workers = min(os.cpu_count() or 4, 8)

    # For small n_boot, serial is faster due to thread overhead
    if n_boot <= 100:
        for i in range(n_boot):
            result = _boot_one_iter((df, dep, indeps, fit_fn, fit_kwargs, i))
            if result is None:
                n_failed += 1
            else:
                for var in boot_params:
                    if var in result:
                        boot_params[var].append(result[var])
    else:
        tasks = [(df, dep, indeps, fit_fn, fit_kwargs, i) for i in range(n_boot)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_boot_one_iter, t) for t in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    n_failed += 1
                else:
                    for var in boot_params:
                        if var in result:
                            boot_params[var].append(result[var])

    alpha = (100 - ci) / 2
    boot_means: dict[str, float] = {}
    boot_std: dict[str, float] = {}
    ci_low: dict[str, float] = {}
    ci_high: dict[str, float] = {}

    for var in original_result.params:
        arr = np.array(boot_params[var])
        if len(arr) > 0:
            boot_means[var] = float(np.mean(arr))
            boot_std[var] = float(np.std(arr, ddof=1))
            ci_low[var] = float(np.percentile(arr, alpha))
            ci_high[var] = float(np.percentile(arr, 100 - alpha))
        else:
            boot_means[var] = float("nan")
            boot_std[var] = float("nan")
            ci_low[var] = float("nan")
            ci_high[var] = float("nan")

    return BootstrapResult(
        original_params=dict(original_result.params),
        boot_means=boot_means,
        boot_std=boot_std,
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=n_boot,
        ci_level=ci,
        n_failed=n_failed,
    )


def _build_X_from_indeps(df: pl.DataFrame, indeps: list[str]) -> np.ndarray:
    """Build an X matrix from *indeps*, handling interaction terms (``":"``).

    Does NOT add a constant. Drops nulls from the relevant columns first.
    Returns the numpy array for the rows present in *df*.
    """
    base_vars = [v for v in indeps if ":" not in v]
    interactions = [v for v in indeps if ":" in v]

    X_parts = df.select(base_vars).to_numpy().astype(float) if base_vars else np.empty((df.height, 0))
    for inter in interactions:
        parts = inter.split(":")
        col = np.ones(df.height)
        for p in parts:
            col = col * df[p].to_numpy().astype(float)
        X_parts = np.column_stack([X_parts, col]) if X_parts.size > 0 else col.reshape(-1, 1)
    return X_parts


def compute_vif(df: pl.DataFrame, indeps: list[str]) -> list[tuple[str, float]]:
    """Compute Variance Inflation Factor for each predictor.

    Handles interaction terms (e.g. ``"x1:x2"`` in *indeps*).
    """
    # Collect all base columns needed
    all_base: set[str] = set()
    for v in indeps:
        if ":" in v:
            all_base.update(v.split(":"))
        else:
            all_base.add(v)

    missing = [c for c in all_base if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    cols_needed = sorted(all_base)
    sub = df.select(cols_needed).drop_nulls()

    # Build full X matrix (handles interactions)
    X_full = _build_X_from_indeps(sub, indeps)
    if X_full.shape[0] < X_full.shape[1] + 1:
        raise ValueError("Too few observations for VIF calculation")

    vifs = []
    for i, var in enumerate(indeps):
        y_i = X_full[:, i]
        X_i = np.delete(X_full, i, axis=1)
        X_i = sm.add_constant(X_i)
        r2 = sm.OLS(y_i, X_i).fit().rsquared
        vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vifs.append((var, vif_val))
    return vifs


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Holds a hypothesis test result."""

    test_name: str
    statistic: float
    p_value: float
    df: float | int | None = None
    details: dict[str, object] = field(default_factory=dict)
    interpretation: str = ""

    def summary_table(self) -> str:
        console = Console(file=io.StringIO(), width=100, record=True)
        table = Table(title=self.test_name)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Test statistic", f"{self.statistic:.4f}")
        if self.df is not None:
            table.add_row("Degrees of freedom", str(self.df))
        table.add_row("p-value", f"{self.p_value:.6f}")

        for k, v in self.details.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))

        console.print(table)

        sig = "significant" if self.p_value < 0.05 else "not significant"
        console.print(f"Result: {sig} at alpha = 0.05")
        if self.interpretation:
            console.print(self.interpretation)
        return console.export_text().rstrip()


def run_ttest(
    df: pl.DataFrame,
    col: str,
    *,
    by: str | None = None,
    mu: float = 0.0,
    paired_col: str | None = None,
) -> TestResult:
    """Run a t-test.

    - One-sample: test col mean against mu
    - Two-sample: split col by a binary grouping variable `by`
    - Paired: test difference between col and paired_col
    """
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")

    if paired_col is not None:
        # Paired t-test
        if paired_col not in df.columns:
            raise ValueError(f"Column not found: {paired_col}")
        sub = df.select([col, paired_col]).drop_nulls()
        a = sub[col].to_numpy().astype(float)
        b = sub[paired_col].to_numpy().astype(float)
        t_stat, p_val = sp_stats.ttest_rel(a, b)
        return TestResult(
            test_name=f"Paired t-test: {col} vs {paired_col}",
            statistic=float(t_stat),
            p_value=float(p_val),
            df=len(a) - 1,
            details={
                "N (pairs)": len(a),
                "Mean difference": float(np.mean(a - b)),
                f"Mean({col})": float(np.mean(a)),
                f"Mean({paired_col})": float(np.mean(b)),
            },
        )

    if by is not None:
        # Two-sample t-test
        if by not in df.columns:
            raise ValueError(f"Column not found: {by}")
        sub = df.select([col, by]).drop_nulls()
        groups = sub[by].unique().sort().to_list()
        if len(groups) != 2:
            raise ValueError(
                f"Two-sample t-test requires exactly 2 groups in '{by}', "
                f"found {len(groups)}: {groups[:5]}"
            )
        g1 = sub.filter(pl.col(by) == groups[0])[col].to_numpy().astype(float)
        g2 = sub.filter(pl.col(by) == groups[1])[col].to_numpy().astype(float)
        t_stat, p_val = sp_stats.ttest_ind(g1, g2, equal_var=False)
        # Welch-Satterthwaite degrees of freedom
        n1, n2 = len(g1), len(g2)
        v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        numerator = (v1 / n1 + v2 / n2) ** 2
        denominator = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        welch_df = numerator / denominator if denominator > 0 else n1 + n2 - 2
        return TestResult(
            test_name=f"Two-sample t-test: {col} by {by} (Welch)",
            statistic=float(t_stat),
            p_value=float(p_val),
            df=round(float(welch_df), 2),
            details={
                f"N({groups[0]})": len(g1),
                f"N({groups[1]})": len(g2),
                f"Mean({groups[0]})": float(np.mean(g1)),
                f"Mean({groups[1]})": float(np.mean(g2)),
            },
        )

    # One-sample t-test
    sub = df[col].drop_nulls().to_numpy().astype(float)
    t_stat, p_val = sp_stats.ttest_1samp(sub, mu)
    return TestResult(
        test_name=f"One-sample t-test: {col} (H0: mu = {mu})",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=len(sub) - 1,
        details={
            "N": len(sub),
            "Sample mean": float(np.mean(sub)),
            "Sample SD": float(np.std(sub, ddof=1)),
            "H0 mean": mu,
        },
    )


def run_chi2(df: pl.DataFrame, col1: str, col2: str) -> TestResult:
    """Run a chi-square test of independence (cross-tabulation)."""
    for c in (col1, col2):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    sub = df.select([col1, col2]).drop_nulls()

    # Build contingency table
    ct = sub.group_by([col1, col2]).len().rename({"len": "count"})
    rows = sorted(sub[col1].unique().to_list(), key=str)
    cols = sorted(sub[col2].unique().to_list(), key=str)

    table = np.zeros((len(rows), len(cols)), dtype=int)
    row_idx = {v: i for i, v in enumerate(rows)}
    col_idx = {v: i for i, v in enumerate(cols)}
    for r in ct.iter_rows(named=True):
        table[row_idx[r[col1]], col_idx[r[col2]]] = r["count"]

    chi2, p_val, dof, expected = sp_stats.chi2_contingency(table)

    return TestResult(
        test_name=f"Chi-square test: {col1} x {col2}",
        statistic=float(chi2),
        p_value=float(p_val),
        df=int(dof),
        details={
            "N": int(sub.height),
            f"Unique({col1})": len(rows),
            f"Unique({col2})": len(cols),
            "Cramér's V": float(np.sqrt(chi2 / (sub.height * (min(len(rows), len(cols)) - 1))))
            if min(len(rows), len(cols)) > 1 else 0.0,
        },
    )


def run_anova(df: pl.DataFrame, col: str, by: str) -> TestResult:
    """Run one-way ANOVA (F-test)."""
    for c in (col, by):
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    sub = df.select([col, by]).drop_nulls()
    groups = sub[by].unique().sort().to_list()

    if len(groups) < 2:
        raise ValueError(f"ANOVA requires at least 2 groups, found {len(groups)}")

    samples = []
    group_stats: list[tuple[str, int, float, float]] = []
    for g in groups:
        vals = sub.filter(pl.col(by) == g)[col].to_numpy().astype(float)
        samples.append(vals)
        group_stats.append((str(g), len(vals), float(np.mean(vals)), float(np.std(vals, ddof=1))))

    f_stat, p_val = sp_stats.f_oneway(*samples)
    k = len(groups)
    n = sub.height

    details: dict[str, object] = {
        "N (total)": n,
        "Groups": k,
        "df (between)": k - 1,
        "df (within)": n - k,
    }
    for name, cnt, mean, sd in group_stats:
        details[f"  {name}: N={cnt}"] = f"mean={mean:.4f}, sd={sd:.4f}"

    return TestResult(
        test_name=f"One-way ANOVA: {col} by {by}",
        statistic=float(f_stat),
        p_value=float(p_val),
        df=k - 1,
        details=details,
    )


# ---------------------------------------------------------------------------
# Stepwise regression
# ---------------------------------------------------------------------------

@dataclass
class StepwiseResult:
    """Holds stepwise selection result."""

    direction: str  # "forward" or "backward"
    selected: list[str]
    dropped: list[str]
    steps: list[dict[str, object]]
    final_fit: FitResult

    def summary(self) -> str:
        """Human-readable summary of variable selection."""
        lines = [f"Stepwise ({self.direction}) variable selection"]
        lines.append(f"Final model: {self.final_fit.formula}")
        lines.append(f"Selected {len(self.selected)} variable(s): {', '.join(self.selected)}")
        if self.dropped:
            lines.append(f"Dropped: {', '.join(self.dropped)}")
        lines.append("")
        for step in self.steps:
            lines.append(f"  Step {step['step']}: {step['action']} '{step['variable']}' "
                         f"(AIC={step['aic']:.2f})")
        lines.append("")
        lines.append(self.final_fit.summary_table())
        return "\n".join(lines)


def stepwise_ols(
    df: pl.DataFrame,
    dep: str,
    candidates: list[str],
    *,
    direction: str = "forward",
    p_enter: float = 0.05,
    p_remove: float = 0.10,
) -> StepwiseResult:
    """Run stepwise OLS regression.

    direction: "forward" or "backward"
    p_enter: p-value threshold to add a variable (forward)
    p_remove: p-value threshold to remove a variable (backward)
    """
    # Collect all base columns needed (interaction components resolved)
    all_base: set[str] = {dep}
    for v in candidates:
        if ":" in v:
            all_base.update(v.split(":"))
        else:
            all_base.add(v)
    missing = [c for c in all_base if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    steps: list[dict[str, object]] = []
    step_num = 0

    if direction == "forward":
        selected: list[str] = []
        remaining = list(candidates)

        while remaining:
            best_var = None
            best_pval = 1.0
            best_aic = float("inf")

            for var in remaining:
                try:
                    trial = selected + [var]
                    result, model = fit_ols(df, dep, trial)
                    var_idx = trial.index(var) + 1  # +1 for constant
                    pval = float(model.pvalues[var_idx])
                    aic = float(model.aic)
                    if pval < best_pval:
                        best_var = var
                        best_pval = pval
                        best_aic = aic
                except Exception:
                    continue

            if best_var is None or best_pval > p_enter:
                break

            selected.append(best_var)
            remaining.remove(best_var)
            step_num += 1
            steps.append({
                "step": step_num, "action": "add", "variable": best_var,
                "p_value": best_pval, "aic": best_aic,
            })

        dropped = [c for c in candidates if c not in selected]

    else:  # backward
        selected = list(candidates)

        while len(selected) > 1:
            result, model = fit_ols(df, dep, selected)
            # Find variable with highest p-value (excluding constant at index 0)
            pvals = model.pvalues[1:]  # skip constant
            worst_idx = int(np.argmax(pvals))
            worst_pval = float(pvals[worst_idx])
            worst_var = selected[worst_idx]

            if worst_pval <= p_remove:
                break

            selected.remove(worst_var)
            step_num += 1
            steps.append({
                "step": step_num, "action": "remove", "variable": worst_var,
                "p_value": worst_pval, "aic": float(model.aic),
            })

        dropped = [c for c in candidates if c not in selected]

    if not selected:
        raise ValueError("No variables selected. Consider relaxing p_enter threshold.")

    final_result, _ = fit_ols(df, dep, selected)
    return StepwiseResult(
        direction=direction,
        selected=selected,
        dropped=dropped,
        steps=steps,
        final_fit=final_result,
    )


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def compute_residuals(
    model: object, df: pl.DataFrame, dep: str, indeps: list[str]
) -> dict[str, np.ndarray]:
    """Compute residual diagnostics from a fitted OLS model.

    Returns dict with: residuals, fitted, std_residuals (internally studentized),
    leverage (hat matrix diagonal).
    """
    # Collect all raw columns needed (including interaction components)
    all_base: set[str] = set()
    for v in indeps:
        if ":" in v:
            all_base.update(v.split(":"))
        else:
            all_base.add(v)
    cols_needed = [dep] + sorted(all_base)
    sub = df.select(cols_needed).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X = _build_X_from_indeps(sub, indeps)
    X = sm.add_constant(X)

    fitted = model.predict(X)
    resid = y - fitted

    # Internally studentized residuals using leverage (hat matrix)
    leverage = np.zeros_like(resid)
    try:
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        leverage = np.diag(H)
        n = len(resid)
        p = X.shape[1]
        # MSE (mean squared error of residuals)
        mse = np.sum(resid ** 2) / (n - p)
        s = np.sqrt(mse)
        denom = s * np.sqrt(np.maximum(1 - leverage, 1e-10))
        resid_std = resid / denom
    except np.linalg.LinAlgError:
        # Fallback to simple standardization if hat matrix fails
        s = np.std(resid, ddof=1)
        resid_std = resid / s if s > 0 else resid

    return {
        "residuals": resid,
        "fitted": fitted,
        "std_residuals": resid_std,
        "leverage": leverage,
        "y": y,
    }
