"""Structural Equation Modeling (SEM) commands."""

from __future__ import annotations

from openstat.commands.base import command
from openstat.session import Session


def _fmt_sem(title: str, lines_body: list[str]) -> str:
    sep = "=" * 60
    return "\n".join([f"\n{title}", sep] + lines_body + [sep])


@command("sem", usage="sem model_syntax [--method=ml|gls|wls]")
def cmd_sem(session: Session, args: str) -> str:
    """Structural Equation Model estimation via semopy.

    Model syntax (lavaan/semopy style):
      Latent variable:   F1 =~ x1 + x2 + x3
      Regression:        y ~ x1 + x2
      Covariance:        x1 ~~ x2

    Examples:
      sem 'F1 =~ age + income + score'
      sem 'eta =~ x1 + x2 + x3 \\n y ~ eta + z'
      sem 'F1 =~ x1 + x2 \\n y ~ F1' --method=ml

    Requires: pip install semopy
    """
    try:
        import semopy
    except ImportError:
        return "SEM requires semopy. Install with: pip install semopy"

    df = session.require_data()

    import re
    args = args.strip()

    # Parse --method option
    method = "ml"
    m = re.search(r"--method[= ](\w+)", args)
    if m:
        method = m.group(1).lower()
        args = re.sub(r"--method[= ]\w+", "", args).strip()

    # Strip surrounding quotes
    model_str = args.strip("\"'")
    # Support \n in quoted strings as actual newlines
    model_str = model_str.replace("\\n", "\n")

    if not model_str:
        return (
            "Usage: sem '<model_syntax>' [--method=ml|gls|wls]\n"
            "Example: sem 'F1 =~ x1 + x2 + x3'\n"
            "Example: sem 'y ~ x1 + x2'"
        )

    # Map user-friendly method names to semopy obj parameter
    _method_map = {"ml": "MLW", "gls": "GLS", "wls": "WLS", "uls": "ULS", "fiml": "FIML"}
    obj = _method_map.get(method.lower(), "MLW")

    try:
        pandas_df = df.to_pandas()
        model = semopy.Model(model_str)
        res = model.fit(pandas_df, obj=obj)

        # Model fit indices
        stats = semopy.calc_stats(model)

        lines = [
            f"Model: {model_str.replace(chr(10), ' | ')}",
            f"Method: {method.upper()}   N = {len(pandas_df)}",
            "",
        ]

        # Parameter estimates
        try:
            params = model.inspect()
            lines.append("Parameter Estimates:")
            lines.append(f"  {'lval':<12} {'op':<5} {'rval':<12} {'Estimate':>10} {'Std.Err':>9} {'z':>7} {'p-value':>9}")
            lines.append("  " + "-" * 65)
            for _, row in params.iterrows():
                p_str = f"{row.get('p-value', float('nan')):.4f}" if row.get('p-value') is not None else "   ."
                z_str = f"{row.get('z-value', float('nan')):.3f}" if row.get('z-value') is not None else "   ."
                se_str = f"{row.get('Std. Err', float('nan')):.4f}" if row.get('Std. Err') is not None else "   ."
                lines.append(
                    f"  {str(row.get('lval','')):<12} {str(row.get('op','')):<5} "
                    f"{str(row.get('rval','')):<12} "
                    f"{row.get('Estimate', float('nan')):>10.4f} "
                    f"{se_str:>9} {z_str:>7} {p_str:>9}"
                )
        except Exception:
            lines.append("(Parameter table unavailable)")

        # Fit indices
        lines.append("")
        lines.append("Model Fit Indices:")
        fit_map = {
            "CFI": "CFI", "GFI": "GFI", "AGFI": "AGFI",
            "NFI": "NFI", "RMSEA": "RMSEA", "SRMR": "SRMR",
            "chi2": "Chi-sq", "DoF": "df", "pvalue": "p(chi-sq)",
            "AIC": "AIC", "BIC": "BIC",
        }
        for key, label in fit_map.items():
            try:
                val = stats[key].iloc[0] if hasattr(stats[key], 'iloc') else stats.get(key)
                if val is not None:
                    lines.append(f"  {label:<12} {float(val):.4f}")
            except Exception:
                pass

        # Interpretation hints
        lines.append("")
        lines.append("Fit guidelines: CFI/GFI > 0.95 (good), RMSEA < 0.05 (good), SRMR < 0.08 (good)")

        return _fmt_sem("Structural Equation Model (SEM)", lines)

    except Exception as exc:
        return f"SEM error: {exc}"


@command("cfa", usage="cfa factor =~ x1 + x2 + x3 [+ factor2 =~ y1 + y2]")
def cmd_cfa(session: Session, args: str) -> str:
    """Confirmatory Factor Analysis — shortcut for SEM with =~ syntax only.

    Examples:
      cfa F1 =~ age + income + score
      cfa 'F1 =~ x1 + x2 \\n F2 =~ y1 + y2'
    """
    try:
        import semopy
    except ImportError:
        return "CFA requires semopy. Install with: pip install semopy"

    df = session.require_data()
    model_str = args.strip().strip("\"'").replace("\\n", "\n")
    if not model_str:
        return "Usage: cfa 'F1 =~ x1 + x2 + x3'"

    try:
        pandas_df = df.to_pandas()
        model = semopy.Model(model_str)
        model.fit(pandas_df)

        params = model.inspect()  # type: ignore[attr-defined]
        stats = semopy.calc_stats(model)

        lines = [f"CFA Model: {model_str.replace(chr(10), ' | ')}", f"N = {len(pandas_df)}", ""]

        # Loadings only (=~ relationships)
        loadings = params[params["op"] == "=~"] if "op" in params.columns else params
        if not loadings.empty:
            lines.append("Factor Loadings:")
            lines.append(f"  {'Factor':<12} {'Indicator':<12} {'Loading':>9} {'Std.Err':>9} {'p-value':>9}")
            lines.append("  " + "-" * 55)
            for _, row in loadings.iterrows():
                p_str = f"{row.get('p-value', float('nan')):.4f}"
                se_str = f"{row.get('Std. Err', float('nan')):.4f}"
                lines.append(
                    f"  {str(row.get('lval','')):<12} {str(row.get('rval','')):<12} "
                    f"{row.get('Estimate', float('nan')):>9.4f} {se_str:>9} {p_str:>9}"
                )

        lines.append("")
        lines.append("Fit Indices:")
        for key, label in [("CFI", "CFI"), ("RMSEA", "RMSEA"), ("SRMR", "SRMR"), ("AIC", "AIC")]:
            try:
                val = stats[key].iloc[0]
                lines.append(f"  {label:<10} {float(val):.4f}")
            except Exception:
                pass

        return _fmt_sem("Confirmatory Factor Analysis (CFA)", lines)

    except Exception as exc:
        return f"CFA error: {exc}"
