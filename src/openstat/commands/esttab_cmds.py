"""esttab and tabstat commands for multi-model comparison tables."""

from __future__ import annotations

import re

import polars as pl

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("esttab", usage="esttab [stats(coef,se,pval)] [stars]")
def cmd_esttab(session: Session, args: str) -> str:
    """Display a publication-style comparison table of all stored regression results."""
    positional, opts = _stata_opts(args)
    stats_req = opts.get("stats", "coef,se").split(",")
    show_stars = "stars" in args

    raw_results = session.results
    if not raw_results:
        return "No stored results. Run regression commands first."

    # Normalize: ModelResult objects → dict via .details
    results = []
    for res in raw_results:
        if isinstance(res, dict):
            results.append(res)
        elif hasattr(res, "details"):
            results.append(res.details)

    # Collect all parameter names across models
    all_params: list[str] = []
    for res in results:
        if "params" in res:
            for p in res["params"]:
                if p not in all_params:
                    all_params.append(p)
        elif "coefficients" in res:
            for p in res["coefficients"]:
                if p not in all_params:
                    all_params.append(p)

    if not all_params:
        return "No regression results with coefficients found."

    col_w = 14

    def _get_coef(res, param):
        if "coefficients" in res:
            c = res["coefficients"].get(param, {})
            return c.get("mean", float("nan"))
        if "params" in res:
            return res["params"].get(param, float("nan"))
        return float("nan")

    def _get_se(res, param):
        if "coefficients" in res:
            c = res["coefficients"].get(param, {})
            return c.get("std", float("nan"))
        if "std_errors" in res:
            return res["std_errors"].get(param, float("nan"))
        return float("nan")

    def _get_pval(res, param):
        if "p_values" in res:
            return res["p_values"].get(param, float("nan"))
        if "coefficients" in res:
            c = res["coefficients"].get(param, {})
            return c.get("prob_positive", float("nan"))
        return float("nan")

    def _stars(p):
        if p != p: return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    model_list = [r for r in results if "params" in r or "coefficients" in r]
    header = f"{'':25}" + "".join(f"  {'('+str(i+1)+')':>{col_w}}" for i in range(len(model_list)))
    sep = "-" * (25 + (col_w + 2) * len(model_list))
    lines = ["\nesttab — Regression Comparison", sep, header, sep]

    for param in all_params:
        coef_row = f"{param:<25}"
        se_row = f"{'':25}"
        for res in model_list:
            coef = _get_coef(res, param)
            se = _get_se(res, param)
            pval = _get_pval(res, param)
            stars = _stars(pval) if show_stars else ""
            if coef != coef:
                coef_row += f"  {'':>{col_w}}"
                se_row += f"  {'':>{col_w}}"
            else:
                coef_str = f"{coef:.4f}{stars}"
                coef_row += f"  {coef_str:>{col_w}}"
                se_str = f"({se:.4f})" if se == se else ""
                se_row += f"  {se_str:>{col_w}}"
        lines.append(coef_row)
        if "se" in stats_req:
            lines.append(se_row)

    lines.append(sep)
    # Model-level stats
    n_row = f"{'N':25}"
    r2_row = f"{'R-squared':25}"
    for res in model_list:
        n = res.get("n_obs", "")
        r2 = res.get("r_squared", res.get("pseudo_r2", ""))
        n_row += f"  {str(n):>{col_w}}"
        r2_row += f"  {(f'{r2:.4f}' if isinstance(r2, float) else ''):>{col_w}}"
    lines.append(n_row)
    lines.append(r2_row)
    lines.append(sep)
    if show_stars:
        lines.append("* p<0.05  ** p<0.01  *** p<0.001")
    return "\n".join(lines)


@command("tabstat", usage="tabstat var1 [var2 ...] [, stats(mean sd min max n) by(groupvar)]")
def cmd_tabstat(session: Session, args: str) -> str:
    """Display summary statistics table (enhanced version of summarize)."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if not cols:
        return "No valid numeric variables found."
    stats_req = [s.strip() for s in opts.get("stats", "mean,sd,min,max,n").split(",")]
    by_raw = opts.get("by", "")
    by_var = by_raw.strip() if by_raw.strip() in df.columns else None

    def _compute_stats(series: pl.Series, stats: list[str]) -> dict:
        res = {}
        if "n" in stats: res["N"] = series.drop_nulls().len()
        if "mean" in stats: res["Mean"] = float(series.mean()) if series.len() else float("nan")
        if "sd" in stats: res["Std Dev"] = float(series.std()) if series.len() > 1 else float("nan")
        if "min" in stats: res["Min"] = float(series.min()) if series.len() else float("nan")
        if "max" in stats: res["Max"] = float(series.max()) if series.len() else float("nan")
        if "median" in stats or "p50" in stats: res["Median"] = float(series.median()) if series.len() else float("nan")
        if "sum" in stats: res["Sum"] = float(series.sum()) if series.len() else float("nan")
        if "var" in stats: res["Variance"] = float(series.var()) if series.len() > 1 else float("nan")
        return res

    stat_labels = []
    for s in stats_req:
        lbl = {"n": "N", "mean": "Mean", "sd": "Std Dev", "min": "Min", "max": "Max",
               "median": "Median", "p50": "Median", "sum": "Sum", "var": "Variance"}.get(s, s)
        if lbl not in stat_labels:
            stat_labels.append(lbl)

    col_w = 12
    lines = ["\ntabstat", "=" * (22 + col_w * len(stat_labels))]
    header = f"{'Variable':<20}" + "".join(f"  {s:>{col_w}}" for s in stat_labels)
    lines.append(header)
    lines.append("-" * (22 + col_w * len(stat_labels)))

    def _add_rows(data: pl.DataFrame, prefix: str = ""):
        for col in cols:
            try:
                s = data[col].cast(pl.Float64)
            except Exception:
                continue
            stat_vals = _compute_stats(s, stats_req)
            row = f"{prefix + col:<20}"
            for lbl in stat_labels:
                val = stat_vals.get(lbl, float("nan"))
                if isinstance(val, float) and val != val:
                    row += f"  {'':>{col_w}}"
                elif isinstance(val, int):
                    row += f"  {val:>{col_w}}"
                else:
                    row += f"  {val:>{col_w}.4f}"
            lines.append(row)

    if by_var:
        groups = df[by_var].unique().sort().to_list()
        for g in groups:
            lines.append(f"\n  {by_var} = {g}")
            lines.append("-" * (22 + col_w * len(stat_labels)))
            _add_rows(df.filter(pl.col(by_var) == g), "  ")
    else:
        _add_rows(df)

    lines.append("=" * (22 + col_w * len(stat_labels)))
    return "\n".join(lines)
