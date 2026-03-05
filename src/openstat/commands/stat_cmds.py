"""Statistics commands: summarize, tabulate, groupby, corr, ols, logit, ttest, chi2, anova."""

from __future__ import annotations

import re

import polars as pl
from rich.console import Console
from rich.table import Table

from openstat.session import Session, ModelResult
from openstat.config import get_config
from openstat.dsl.parser import parse_formula, ParseError
from openstat.stats.models import (
    fit_ols, fit_logit, fit_probit, fit_poisson, fit_negbin, fit_quantreg,
    compute_margins, bootstrap_model,
    run_ttest, run_chi2, run_anova,
    compute_vif, stepwise_ols, compute_residuals,
)
from openstat.commands.base import command, CommandArgs, rich_to_str, friendly_error
from openstat.types import NUMERIC_DTYPES


def _store_model(
    session: Session, result, raw_model, dep: str, indeps: list[str],
    fit_kwargs: dict | None = None,
) -> str:
    """Store model in session state, return summary output."""
    session._last_model = raw_model
    session._last_model_vars = (dep, indeps)
    session._last_fit_result = result
    session._last_fit_kwargs = fit_kwargs or {}
    md = result.to_markdown()
    details: dict = {
        "n_obs": result.n_obs,
        "params": dict(result.params),
        "std_errors": dict(result.std_errors),
        "aic": result.aic,
        "bic": result.bic,
    }
    if result.r_squared is not None:
        details["r_squared"] = result.r_squared
    if result.adj_r_squared is not None:
        details["adj_r_squared"] = result.adj_r_squared
    if result.pseudo_r2 is not None:
        details["pseudo_r2"] = result.pseudo_r2
    if result.log_likelihood is not None:
        details["log_likelihood"] = result.log_likelihood
    if result.dispersion is not None:
        details["dispersion"] = result.dispersion
    session.results.append(ModelResult(
        name=result.model_type, formula=result.formula,
        table=md, details=details,
    ))
    output = result.summary_table()
    if result.warnings:
        output += "\n" + "\n".join(result.warnings)
    return output


def _parse_agg(token: str) -> tuple[str, str | None]:
    m = re.match(r"(\w+)\((\w*)\)", token)
    if not m:
        raise ValueError(f"Invalid aggregation: {token}. Use e.g. mean(col), count()")
    func, col = m.group(1), m.group(2) or None
    return func, col


@command("summarize", usage="summarize [col1 col2 ...]")
def cmd_summarize(session: Session, args: str) -> str:
    """Compute summary statistics for numeric columns (SD = sample, ddof=1)."""
    df = session.require_data()
    cols = args.split() if args.strip() else None

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}"
        num_cols = [c for c in cols if df[c].dtype in NUMERIC_DTYPES]
    else:
        num_cols = [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]

    if not num_cols:
        return "No numeric columns to summarize."

    def render(console: Console) -> None:
        table = Table(title="Summary Statistics")
        table.add_column("Variable", style="cyan")
        for stat in ["N", "Mean", "SD (sample)", "Min", "P25", "P50", "P75", "Max"]:
            table.add_column(stat, justify="right")

        for c in num_cols:
            col = df[c].drop_nulls()
            n = col.len()
            if n == 0:
                table.add_row(c, "0", *["—"] * 7)
                continue
            sd_val = col.std() if n > 1 else 0.0
            table.add_row(
                c, str(n),
                f"{col.mean():.4f}", f"{sd_val:.4f}", f"{col.min():.4f}",
                f"{col.quantile(0.25):.4f}", f"{col.quantile(0.50):.4f}",
                f"{col.quantile(0.75):.4f}", f"{col.max():.4f}",
            )
        console.print(table)

    return rich_to_str(render)


@command("tabulate", usage="tabulate <column>")
def cmd_tabulate(session: Session, args: str) -> str:
    """Show frequency table for a column (top 50 values by default)."""
    df = session.require_data()
    col = args.strip()
    if not col:
        return "Usage: tabulate <column>"
    if col not in df.columns:
        return f"Column not found: {col}"

    tab_limit = get_config().tabulate_limit
    counts = (
        df.group_by(col).len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )
    total = counts["count"].sum()
    total_unique = counts.height
    truncated = total_unique > tab_limit

    if truncated:
        counts = counts.head(tab_limit)

    counts = counts.with_columns(
        (pl.col("count") / total * 100).round(1).alias("percent")
    )

    def render(console: Console) -> None:
        title = f"Frequency: {col}"
        if truncated:
            title += f" (top {tab_limit} of {total_unique} unique values)"
        table = Table(title=title)
        table.add_column(col, style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percent", justify="right")

        for row in counts.iter_rows(named=True):
            table.add_row(str(row[col]), str(row["count"]), f"{row['percent']:.1f}%")
        table.add_row("Total", str(total), "100.0%", style="bold")
        console.print(table)

    return rich_to_str(render)


@command("corr", usage="corr [col1 col2 ...]")
def cmd_corr(session: Session, args: str) -> str:
    """Show correlation matrix for numeric columns."""
    df = session.require_data()
    cols = args.split() if args.strip() else None

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}"
        num_cols = [c for c in cols if df[c].dtype in NUMERIC_DTYPES]
    else:
        num_cols = [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]

    if len(num_cols) < 2:
        return "Need at least 2 numeric columns for correlation."

    sub = df.select(num_cols).drop_nulls()
    # Compute pairwise correlation
    corr_data: dict[str, list[float]] = {}
    for c1 in num_cols:
        row = []
        for c2 in num_cols:
            r = sub.select(pl.corr(c1, c2)).item()
            row.append(r if r is not None else 0.0)
        corr_data[c1] = row

    def render(console: Console) -> None:
        table = Table(title="Correlation Matrix (Pearson)")
        table.add_column("", style="cyan")
        for c in num_cols:
            table.add_column(c, justify="right")
        for i, c1 in enumerate(num_cols):
            vals = []
            for j, c2 in enumerate(num_cols):
                r = corr_data[c1][j]
                # Highlight strong correlations
                if i == j:
                    vals.append("1.0000")
                elif abs(r) > 0.7:
                    vals.append(f"[bold]{r:.4f}[/bold]")
                else:
                    vals.append(f"{r:.4f}")
            table.add_row(c1, *vals)
        console.print(table)

    return rich_to_str(render)


@command("groupby", usage="groupby <cols> summarize <agg(col)> ...")
def cmd_groupby(session: Session, args: str) -> str:
    """Group-by and summarize."""
    df = session.require_data()

    ca = CommandArgs(args)
    summarize_rest = ca.rest_after("summarize")
    if summarize_rest is None:
        return "Usage: groupby <col1> <col2> summarize mean(x) sd(x) count()"

    # Group cols are positional tokens before "summarize"
    group_cols = []
    for p in ca.positional:
        if p.lower() == "summarize":
            break
        group_cols.append(p)
    agg_tokens = summarize_rest.split()

    if not group_cols:
        return "No grouping columns specified."
    if not agg_tokens:
        return "No aggregation functions specified."

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    AGG_MAP = {
        "mean": lambda c: pl.col(c).mean().alias(f"mean_{c}"),
        "sd": lambda c: pl.col(c).std().alias(f"sd_{c}"),
        "sum": lambda c: pl.col(c).sum().alias(f"sum_{c}"),
        "min": lambda c: pl.col(c).min().alias(f"min_{c}"),
        "max": lambda c: pl.col(c).max().alias(f"max_{c}"),
        "median": lambda c: pl.col(c).median().alias(f"median_{c}"),
        "count": lambda _: pl.len().alias("count"),
    }

    agg_exprs = []
    for tok in agg_tokens:
        func_name, col_name = _parse_agg(tok)
        if func_name not in AGG_MAP:
            return f"Unknown aggregation: {func_name}. Available: {', '.join(AGG_MAP)}"
        if func_name != "count" and col_name is None:
            return f"{func_name}() requires a column name, e.g. {func_name}(col)"
        if col_name and col_name not in df.columns:
            return f"Column not found: {col_name}"
        agg_exprs.append(AGG_MAP[func_name](col_name))

    result = df.group_by(group_cols).agg(agg_exprs).sort(group_cols)

    def render(console: Console) -> None:
        table = Table(title="Group Summary")
        for col_name in result.columns:
            table.add_column(col_name, justify="right" if col_name not in group_cols else "left")
        for row in result.iter_rows():
            table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
        console.print(table)

    return rich_to_str(render)


@command("ols", usage="ols y ~ x1 + x2 [--robust] [--cluster=col]")
def cmd_ols(session: Session, args: str) -> str:
    """Fit OLS regression."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: ols y ~ x1 + x2 [--robust] [--cluster=col]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_ols(df, dep, indeps, robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "OLS error")


@command("logit", usage="logit y ~ x1 + x2 [--robust] [--cluster=col]")
def cmd_logit(session: Session, args: str) -> str:
    """Fit logistic regression (binary dependent variable)."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: logit y ~ x1 + x2 [--robust] [--cluster=col]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_logit(df, dep, indeps, robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Logit error")


@command("ttest", usage="ttest <col> [by <group>] [mu=<value>] [paired <col2>]")
def cmd_ttest(session: Session, args: str) -> str:
    """T-test: one-sample, two-sample (Welch), or paired."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return (
            "Usage:\n"
            "  ttest <col>              One-sample (H0: mean=0)\n"
            "  ttest <col> mu=5         One-sample (H0: mean=5)\n"
            "  ttest <col> by <group>   Two-sample (Welch)\n"
            "  ttest <col> paired <col2> Paired t-test"
        )

    col = ca.positional[0]
    mu = ca.get_option_float("mu", 0.0)

    by = None
    by_rest = ca.rest_after("by")
    if by_rest:
        by = by_rest.split()[0]

    paired_col = None
    paired_rest = ca.rest_after("paired")
    if paired_rest:
        paired_col = paired_rest.split()[0]

    try:
        result = run_ttest(df, col, by=by, mu=mu, paired_col=paired_col)
        return result.summary_table()
    except Exception as e:
        return friendly_error(e, "T-test error")


@command("chi2", usage="chi2 <col1> <col2>")
def cmd_chi2(session: Session, args: str) -> str:
    """Chi-square test of independence between two categorical columns."""
    df = session.require_data()
    parts = args.split()
    if len(parts) < 2:
        return "Usage: chi2 <col1> <col2>"

    try:
        result = run_chi2(df, parts[0], parts[1])
        return result.summary_table()
    except Exception as e:
        return friendly_error(e, "Chi-square error")


@command("anova", usage="anova <col> by <group>")
def cmd_anova(session: Session, args: str) -> str:
    """One-way ANOVA: test if group means differ."""
    df = session.require_data()
    ca = CommandArgs(args)
    by_str = ca.rest_after("by")
    if not by_str:
        return "Usage: anova <col> by <group_col>"
    col = ca.positional[0] if ca.positional else ""
    by_col = by_str.split()[0]
    if not col or not by_col:
        return "Usage: anova <col> by <group_col>"
    try:
        result = run_anova(df, col, by_col)
        return result.summary_table()
    except Exception as e:
        return friendly_error(e, "ANOVA error")


@command("crosstab", usage="crosstab <row_col> <col_col>")
def cmd_crosstab(session: Session, args: str) -> str:
    """Two-way frequency table (contingency table) with row percentages."""
    df = session.require_data()
    parts = args.split()
    if len(parts) < 2:
        return "Usage: crosstab <row_col> <col_col>"

    row_col, col_col = parts[0], parts[1]
    for c in (row_col, col_col):
        if c not in df.columns:
            return f"Column not found: {c}"

    sub = df.select([row_col, col_col]).drop_nulls()
    ct = sub.group_by([row_col, col_col]).len().rename({"len": "count"})

    rows = sorted(sub[row_col].unique().to_list(), key=str)
    cols = sorted(sub[col_col].unique().to_list(), key=str)

    # Build count matrix
    count_map: dict[tuple, int] = {}
    for r in ct.iter_rows(named=True):
        count_map[(r[row_col], r[col_col])] = r["count"]

    def render(console: Console) -> None:
        table = Table(title=f"Cross-tabulation: {row_col} x {col_col}")
        table.add_column(row_col, style="cyan")
        for c in cols:
            table.add_column(str(c), justify="right")
        table.add_column("Total", justify="right", style="bold")

        for row_val in rows:
            row_total = sum(count_map.get((row_val, c), 0) for c in cols)
            cells = []
            for c in cols:
                cnt = count_map.get((row_val, c), 0)
                pct = cnt / row_total * 100 if row_total > 0 else 0
                cells.append(f"{cnt} ({pct:.0f}%)")
            table.add_row(str(row_val), *cells, str(row_total))

        # Total row
        col_totals = [sum(count_map.get((r, c), 0) for r in rows) for c in cols]
        grand_total = sum(col_totals)
        table.add_row(
            "Total",
            *[str(t) for t in col_totals],
            str(grand_total),
            style="bold",
        )
        console.print(table)

    return rich_to_str(render)


@command("probit", usage="probit y ~ x1 + x2 [--robust] [--cluster=col]")
def cmd_probit(session: Session, args: str) -> str:
    """Fit probit regression (binary dependent variable)."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: probit y ~ x1 + x2 [--robust] [--cluster=col]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_probit(df, dep, indeps, robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Probit error")


@command("predict", usage="predict [<col_name>]")
def cmd_predict(session: Session, args: str) -> str:
    """Generate predictions from the last fitted model, add as a new column."""
    import statsmodels.api as sm
    from openstat.stats.models import _build_X_from_indeps

    df = session.require_data()
    if session._last_model is None or session._last_model_vars is None:
        return "No model fitted yet. Run ols, logit, or probit first."

    col_name = args.strip() or "yhat"
    dep, indeps = session._last_model_vars

    # Collect all base columns needed (including interaction components)
    all_base: set[str] = set()
    for v in indeps:
        if ":" in v:
            all_base.update(v.split(":"))
        else:
            all_base.add(v)
    missing = [c for c in all_base if c not in df.columns]
    if missing:
        return f"Predictor columns not found in current data: {', '.join(missing)}"

    model = session._last_model
    X = _build_X_from_indeps(df, indeps)
    X = sm.add_constant(X)
    preds = model.predict(X)

    session.snapshot()
    session.df = df.with_columns(pl.Series(col_name, preds.tolist()).cast(pl.Float64))
    return f"Predictions added as '{col_name}'. {session.shape_str}. Use 'undo' to revert."


@command("vif", usage="vif")
def cmd_vif(session: Session, args: str) -> str:
    """Show Variance Inflation Factor for the last fitted model's predictors."""
    df = session.require_data()
    if session._last_model_vars is None:
        return "No model fitted yet. Run ols first, then vif."

    dep, indeps = session._last_model_vars
    if len(indeps) < 2:
        return "VIF requires at least 2 predictors."

    try:
        vifs = compute_vif(df, indeps)

        def render(console: Console) -> None:
            table = Table(title="Variance Inflation Factor")
            table.add_column("Variable", style="cyan")
            table.add_column("VIF", justify="right")
            table.add_column("Status", style="green")

            for var, vif_val in vifs:
                if vif_val > 10:
                    status = "[red]HIGH[/red]"
                elif vif_val > 5:
                    status = "[yellow]moderate[/yellow]"
                else:
                    status = "ok"
                table.add_row(var, f"{vif_val:.2f}", status)
            console.print(table)
            console.print("Rule of thumb: VIF > 10 indicates serious multicollinearity")

        return rich_to_str(render)
    except Exception as e:
        return friendly_error(e, "VIF error")


@command("stepwise", usage="stepwise y ~ x1 + x2 + x3 [--backward] [--p_enter=0.05] [--p_remove=0.10]")
def cmd_stepwise(session: Session, args: str) -> str:
    """Run stepwise OLS regression for variable selection."""
    df = session.require_data()
    if not args.strip():
        return (
            "Usage: stepwise y ~ x1 + x2 + x3 [--backward]\n"
            "Options: --p_enter=0.05  --p_remove=0.10"
        )

    ca = CommandArgs(args)
    direction = "backward" if ca.has_flag("--backward") else "forward"
    p_enter = ca.get_option_float("p_enter", 0.05)
    p_remove = ca.get_option_float("p_remove", 0.10)
    formula_str = ca.strip_flags_and_options()
    try:
        dep, indeps = parse_formula(formula_str)
        result = stepwise_ols(
            df, dep, indeps, direction=direction,
            p_enter=p_enter, p_remove=p_remove,
        )
        session._last_model = None  # stepwise doesn't store a single model
        session._last_model_vars = (dep, result.selected)
        return result.summary()
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Stepwise error")


@command("residuals", usage="residuals [<col_name>]")
def cmd_residuals(session: Session, args: str) -> str:
    """Add residuals from the last model as a new column. Generates diagnostic plots."""
    df = session.require_data()
    if session._last_model is None or session._last_model_vars is None:
        return "No model fitted yet. Run ols first, then residuals."

    col_name = args.strip() or "residuals"
    dep, indeps = session._last_model_vars

    try:
        diag = compute_residuals(session._last_model, df, dep, indeps)
    except Exception as e:
        return friendly_error(e, "Residuals error")

    session.snapshot()
    session.df = df.with_columns(
        pl.Series(col_name, diag["residuals"].tolist()).cast(pl.Float64)
    )

    # Generate diagnostic plots
    from openstat.plots.plotter import plot_residuals_vs_fitted, plot_qq, plot_scale_location
    paths = []
    try:
        paths.append(plot_residuals_vs_fitted(diag["fitted"], diag["residuals"], session.output_dir))
        paths.append(plot_qq(diag["std_residuals"], session.output_dir))
        paths.append(plot_scale_location(diag["fitted"], diag["std_residuals"], session.output_dir))
        session.plot_paths.extend(str(p) for p in paths)
    except Exception:
        pass  # plots are optional

    lines = [f"Residuals added as '{col_name}'. {session.shape_str}."]
    if paths:
        lines.append("Diagnostic plots saved:")
        for p in paths:
            lines.append(f"  {p}")
    lines.append("Use 'undo' to revert.")
    return "\n".join(lines)


@command("latex", usage="latex [<path.tex>]")
def cmd_latex(session: Session, args: str) -> str:
    """Export the last model result as a LaTeX table."""
    if session._last_fit_result is None:
        return "No model results to export. Run ols, logit, or probit first."

    from openstat.stats.models import FitResult
    result: FitResult = session._last_fit_result  # type: ignore[assignment]
    latex_str = result.to_latex()

    path = args.strip()
    if path:
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(latex_str, encoding="utf-8")
        return f"LaTeX table saved to {p}"
    return latex_str


@command("poisson", usage="poisson y ~ x1 + x2 [--robust] [--cluster=col] [--exposure=col]")
def cmd_poisson(session: Session, args: str) -> str:
    """Fit Poisson regression for count data."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    exposure_col = ca.get_option("exposure")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: poisson y ~ x1 + x2 [--robust] [--exposure=col]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_poisson(
            df, dep, indeps, robust=robust,
            cluster_col=cluster_col, exposure_col=exposure_col,
        )
        kw = {"exposure_col": exposure_col} if exposure_col else {}
        return _store_model(session, result, raw_model, dep, indeps, fit_kwargs=kw)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Poisson error")


@command("negbin", usage="negbin y ~ x1 + x2 [--robust] [--cluster=col]")
def cmd_negbin(session: Session, args: str) -> str:
    """Fit Negative Binomial regression for overdispersed count data."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: negbin y ~ x1 + x2 [--robust]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_negbin(
            df, dep, indeps, robust=robust, cluster_col=cluster_col,
        )
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "NegBin error")


@command("quantreg", usage="quantreg y ~ x1 + x2 [tau=0.5]")
def cmd_quantreg(session: Session, args: str) -> str:
    """Fit quantile regression (default: median, tau=0.5)."""
    df = session.require_data()
    ca = CommandArgs(args)
    tau = ca.get_option_float("tau", 0.5)
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: quantreg y ~ x1 + x2 [tau=0.5]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_quantreg(df, dep, indeps, tau=tau)
        return _store_model(session, result, raw_model, dep, indeps, fit_kwargs={"tau": tau})
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "QuantReg error")


@command("margins", usage="margins [--at=means|average]")
def cmd_margins(session: Session, args: str) -> str:
    """Compute marginal effects after logit or probit."""
    if session._last_model is None:
        return "No model fitted. Run logit or probit first."
    if not hasattr(session._last_model, "get_margeff"):
        return "Marginal effects only available for logit/probit models."

    ca = CommandArgs(args)
    method = ca.get_option("at", "average") or "average"

    try:
        # Build var_names from last fit result
        fit_result = session._last_fit_result
        var_names = list(fit_result.params.keys()) if fit_result else []  # type: ignore[union-attr]
        result = compute_margins(session._last_model, var_names, method)
        return result.summary_table()
    except Exception as e:
        return friendly_error(e, "Margins error")


@command("bootstrap", usage="bootstrap [n=1000] [ci=95]")
def cmd_bootstrap(session: Session, args: str) -> str:
    """Bootstrap confidence intervals for the last fitted model."""
    if session._last_fit_result is None:
        return "No model fitted. Run a model command first."
    if session._last_model_vars is None:
        return "No model fitted. Run a model command first."

    ca = CommandArgs(args)
    n_boot = int(ca.get_option_float("n", float(get_config().bootstrap_iterations)))
    ci = ca.get_option_float("ci", 95.0)

    dep, indeps = session._last_model_vars

    # Determine which fit function to use from last model type
    fit_fn_map = {
        "OLS": fit_ols,
        "Logit": fit_logit,
        "Probit": fit_probit,
        "Poisson": fit_poisson,
        "NegBin": fit_negbin,
    }
    model_type = session._last_fit_result.model_type.split()[0]  # type: ignore[union-attr]
    # Handle QuantReg(tau=0.5) format
    if model_type.startswith("QuantReg"):
        fit_fn = fit_quantreg
    else:
        fit_fn = fit_fn_map.get(model_type)  # type: ignore[assignment]

    if fit_fn is None:
        return f"Bootstrap not supported for model type: {model_type}"

    try:
        result = bootstrap_model(
            session.require_data(), dep, indeps, fit_fn, n_boot, ci,
            **session._last_fit_kwargs,
        )
        return result.summary_table()
    except Exception as e:
        return friendly_error(e, "Bootstrap error")


# ---------------------------------------------------------------------------
# estat — post-estimation diagnostics
# ---------------------------------------------------------------------------

@command("estat", usage="estat <subcommand>  (hettest | ovtest | linktest | ic | all)")
def cmd_estat(session: Session, args: str) -> str:
    """Post-estimation diagnostics (Stata-style).

    Subcommands:
      hettest   — Breusch-Pagan / Cook-Weisberg heteroscedasticity test
      ovtest    — Ramsey RESET specification test
      linktest  — link test for model specification
      ic        — Information criteria (AIC, BIC, log-likelihood)
      all       — Run all diagnostics
    """
    import statsmodels.api as sm
    from openstat.stats.models import _build_X_from_indeps

    if session._last_model is None or session._last_model_vars is None:
        return "No model fitted. Run ols, logit, or probit first."

    sub_cmd = args.strip().lower()
    if not sub_cmd:
        return (
            "Usage: estat <subcommand>\n"
            "  hettest   Breusch-Pagan heteroscedasticity test\n"
            "  ovtest    Ramsey RESET specification test\n"
            "  linktest  Link test for model specification\n"
            "  ic        Information criteria (AIC, BIC)\n"
            "  all       Run all diagnostics"
        )

    model = session._last_model
    dep, indeps = session._last_model_vars
    df = session.require_data()

    # Build data aligned with model
    all_base: set[str] = set()
    for v in indeps:
        if ":" in v:
            all_base.update(v.split(":"))
        else:
            all_base.add(v)
    cols_needed = [dep] + sorted(all_base)
    sub_df = df.select(cols_needed).drop_nulls()
    y = sub_df[dep].to_numpy().astype(float)
    X = _build_X_from_indeps(sub_df, indeps)
    X = sm.add_constant(X)

    results: list[str] = []

    # ── hettest ────────────────────────────────────────────────────────
    if sub_cmd in ("hettest", "all"):
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            resid = y - model.predict(X)
            bp_stat, bp_pval, f_stat, f_pval = het_breuschpagan(resid, X)

            def render_het(console: Console) -> None:
                t = Table(title="Breusch-Pagan / Cook-Weisberg Test for Heteroscedasticity")
                t.add_column("Metric", style="cyan")
                t.add_column("Value", justify="right")
                t.add_row("LM statistic", f"{bp_stat:.4f}")
                t.add_row("LM p-value", f"{bp_pval:.6f}")
                t.add_row("F statistic", f"{f_stat:.4f}")
                t.add_row("F p-value", f"{f_pval:.6f}")
                console.print(t)
                sig = "Heteroscedasticity detected" if bp_pval < 0.05 else "No significant heteroscedasticity"
                console.print(f"Result: {sig} at alpha = 0.05")

            results.append(rich_to_str(render_het))
        except Exception as e:
            results.append(f"hettest failed: {e}")

    # ── ovtest (Ramsey RESET) ──────────────────────────────────────────
    if sub_cmd in ("ovtest", "all"):
        try:
            from statsmodels.stats.diagnostic import linear_reset
            fitted_model = sm.OLS(y, X).fit()
            reset_test = linear_reset(fitted_model, power=3, use_f=True)

            def render_ov(console: Console) -> None:
                t = Table(title="Ramsey RESET Test (powers 2-3)")
                t.add_column("Metric", style="cyan")
                t.add_column("Value", justify="right")
                t.add_row("F statistic", f"{reset_test.fvalue:.4f}")
                t.add_row("p-value", f"{reset_test.pvalue:.6f}")
                t.add_row("df", f"({int(reset_test.df_num)}, {int(reset_test.df_denom)})")
                console.print(t)
                sig = "Specification error detected" if reset_test.pvalue < 0.05 else "No specification error"
                console.print(f"Result: {sig} at alpha = 0.05")

            results.append(rich_to_str(render_ov))
        except Exception as e:
            results.append(f"ovtest failed: {e}")

    # ── linktest ───────────────────────────────────────────────────────
    if sub_cmd in ("linktest", "all"):
        try:
            yhat = model.predict(X)
            yhat_sq = yhat ** 2
            import numpy as _np
            X_link = sm.add_constant(_np.column_stack([yhat, yhat_sq]))
            link_model = sm.OLS(y, X_link).fit()
            p_hatsq = float(link_model.pvalues[2])

            def render_link(console: Console) -> None:
                t = Table(title="Link Test for Model Specification")
                t.add_column("Variable", style="cyan")
                t.add_column("Coef", justify="right")
                t.add_column("Std.Err", justify="right")
                t.add_column("t", justify="right")
                t.add_column("P>|t|", justify="right")
                names = ["_cons", "_hat", "_hatsq"]
                for i, name in enumerate(names):
                    t.add_row(
                        name,
                        f"{link_model.params[i]:.4f}",
                        f"{link_model.bse[i]:.4f}",
                        f"{link_model.tvalues[i]:.3f}",
                        f"{link_model.pvalues[i]:.4f}",
                    )
                console.print(t)
                if p_hatsq < 0.05:
                    console.print("Note: _hatsq is significant — possible specification error.")
                else:
                    console.print("Note: _hatsq is not significant — model appears well-specified.")

            results.append(rich_to_str(render_link))
        except Exception as e:
            results.append(f"linktest failed: {e}")

    # ── ic (information criteria) ──────────────────────────────────────
    if sub_cmd in ("ic", "all"):
        try:
            def render_ic(console: Console) -> None:
                t = Table(title="Information Criteria")
                t.add_column("Criterion", style="cyan")
                t.add_column("Value", justify="right")
                if hasattr(model, "aic"):
                    t.add_row("AIC", f"{model.aic:.2f}")
                if hasattr(model, "bic"):
                    t.add_row("BIC", f"{model.bic:.2f}")
                if hasattr(model, "llf"):
                    t.add_row("Log-Likelihood", f"{model.llf:.2f}")
                if hasattr(model, "nobs"):
                    t.add_row("N", str(int(model.nobs)))
                if hasattr(model, "df_model"):
                    t.add_row("df (model)", str(int(model.df_model)))
                console.print(t)

            results.append(rich_to_str(render_ic))
        except Exception as e:
            results.append(f"ic failed: {e}")

    if not results:
        return f"Unknown estat subcommand: {sub_cmd}. Use: hettest, ovtest, linktest, ic, all"

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# estimates table — model comparison
# ---------------------------------------------------------------------------

@command("estimates", usage="estimates table")
def cmd_estimates(session: Session, args: str) -> str:
    """Compare stored model results side-by-side."""
    sub = args.strip().lower()
    if sub != "table":
        return "Usage: estimates table"

    model_results = [r for r in session.results if hasattr(r, "formula")]
    if len(model_results) < 2:
        return "Need at least 2 stored model results. Run multiple models first."

    # Collect all variable names across all models (preserving order)
    all_vars: list[str] = []
    seen_vars: set[str] = set()
    for mr in model_results:
        params = mr.details.get("params", {})
        for var in params:
            if var not in seen_vars:
                seen_vars.add(var)
                all_vars.append(var)

    def render(console: Console) -> None:
        table = Table(title="Model Comparison")
        table.add_column("", style="cyan")
        for i, mr in enumerate(model_results):
            label = f"({i + 1}) {mr.name}"
            table.add_column(label, justify="right")

        # Coefficient rows with SE in parentheses
        for var in all_vars:
            vals = []
            for mr in model_results:
                params = mr.details.get("params", {})
                se = mr.details.get("std_errors", {})
                if var in params:
                    coef = params[var]
                    se_val = se.get(var)
                    cell = f"{coef:.4f}"
                    if se_val is not None:
                        cell += f"\n({se_val:.4f})"
                    vals.append(cell)
                else:
                    vals.append("—")
            table.add_row(var, *vals)

        # Separator
        table.add_section()

        # Model statistics
        stat_keys = [
            ("n_obs", "N"), ("r_squared", "R²"), ("adj_r_squared", "Adj. R²"),
            ("pseudo_r2", "Pseudo R²"), ("log_likelihood", "Log-Lik."),
            ("aic", "AIC"), ("bic", "BIC"), ("dispersion", "Dispersion (α)"),
        ]

        for key, label in stat_keys:
            vals = []
            any_present = False
            for mr in model_results:
                v = mr.details.get(key)
                if v is not None:
                    any_present = True
                    if isinstance(v, float):
                        if key == "n_obs":
                            vals.append(str(int(v)))
                        else:
                            vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                else:
                    vals.append("—")
            if any_present:
                table.add_row(label, *vals)

        console.print(table)
        console.print("Standard errors in parentheses")

    return rich_to_str(render)
