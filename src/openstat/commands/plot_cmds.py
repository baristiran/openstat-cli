"""Plot commands: hist, scatter, line, box, bar, heatmap."""

from __future__ import annotations

from openstat.session import Session
from openstat.plots.plotter import (
    plot_histogram, plot_scatter, plot_line, plot_box, plot_bar, plot_heatmap,
    plot_residuals_vs_fitted, plot_qq, plot_scale_location, plot_coef,
    plot_interaction,
)
from openstat.commands.base import command, CommandArgs, friendly_error


@command("plot", usage="plot hist|scatter|line|box|bar|heatmap|coef|interaction ...")
def cmd_plot(session: Session, args: str) -> str:
    """Create plots: hist, scatter, line, box, bar, heatmap, coef, interaction."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return (
            "Usage:\n"
            "  plot hist <col>\n"
            "  plot scatter <y> <x>\n"
            "  plot line <y> <x>\n"
            "  plot box <col> [by <group_col>]\n"
            "  plot bar <col> [by <group_col>]\n"
            "  plot heatmap [col1 col2 ...]\n"
            "  plot coef                      (coefficient plot after model)\n"
            "  plot margins                   (marginal effects plot after margins)\n"
            "  plot interaction <y> <x> <mod> (interaction plot, ±1SD split)\n"
            "  plot diagnostics               (diagnostic plots after model)"
        )

    subcmd = ca.positional[0]
    try:
        if subcmd == "hist":
            if len(ca.positional) < 2:
                return "Usage: plot hist <col>"
            col = ca.positional[1]
            path = plot_histogram(df, col, session.output_dir)
            session.plot_paths.append(str(path))
            return f"Histogram saved: {path}"

        elif subcmd == "scatter":
            if len(ca.positional) < 3:
                return "Usage: plot scatter <y_col> <x_col>"
            y_col, x_col = ca.positional[1], ca.positional[2]
            path = plot_scatter(df, y_col, x_col, session.output_dir)
            session.plot_paths.append(str(path))
            return f"Scatter plot saved: {path}"

        elif subcmd == "line":
            if len(ca.positional) < 3:
                return "Usage: plot line <y_col> <x_col>"
            y_col, x_col = ca.positional[1], ca.positional[2]
            path = plot_line(df, y_col, x_col, session.output_dir)
            session.plot_paths.append(str(path))
            return f"Line plot saved: {path}"

        elif subcmd == "box":
            if len(ca.positional) < 2:
                return "Usage: plot box <col> [by <group_col>]"
            col = ca.positional[1]
            by_rest = ca.rest_after("by")
            group_col = by_rest.split()[0] if by_rest else None
            path = plot_box(df, col, session.output_dir, group_col=group_col)
            session.plot_paths.append(str(path))
            return f"Box plot saved: {path}"

        elif subcmd == "bar":
            if len(ca.positional) < 2:
                return "Usage: plot bar <col> [by <group_col>]"
            col = ca.positional[1]
            by_rest = ca.rest_after("by")
            group_col = by_rest.split()[0] if by_rest else None
            path = plot_bar(df, col, session.output_dir, group_col=group_col)
            session.plot_paths.append(str(path))
            return f"Bar chart saved: {path}"

        elif subcmd == "heatmap":
            cols = ca.positional[1:] if len(ca.positional) > 1 else None
            path = plot_heatmap(df, cols, session.output_dir)
            session.plot_paths.append(str(path))
            return f"Heatmap saved: {path}"

        elif subcmd == "acf":
            if len(ca.positional) < 2:
                return "Usage: plot acf <col>"
            col = ca.positional[1]
            from openstat.plots.ts_plots import plot_acf
            series = df[col].drop_nulls().to_numpy()
            path = plot_acf(series, col, session.output_dir)
            session.plot_paths.append(str(path))
            return f"ACF plot saved: {path}"

        elif subcmd == "pacf":
            if len(ca.positional) < 2:
                return "Usage: plot pacf <col>"
            col = ca.positional[1]
            from openstat.plots.ts_plots import plot_pacf
            series = df[col].drop_nulls().to_numpy()
            path = plot_pacf(series, col, session.output_dir)
            session.plot_paths.append(str(path))
            return f"PACF plot saved: {path}"

        elif subcmd == "coef":
            # session._last_model is the raw statsmodels result object
            raw = session._last_model
            if raw is None:
                return "No model fitted yet. Run ols/logit/etc. first, then plot coef."
            if not hasattr(raw, "params"):
                return "Current model does not support coefficient plots."
            import numpy as np
            # Get parameter names from model or FitResult
            fit_result = session._last_fit_result
            if fit_result is not None and hasattr(fit_result, "params") and isinstance(fit_result.params, dict):
                names = list(fit_result.params.keys())
                coef_vals = [fit_result.params[n] for n in names]
                se_vals = [fit_result.std_errors.get(n, 0.0) for n in names]
                params = dict(zip(names, coef_vals))
                ci_lower = {n: v - 1.96 * se for n, v, se in zip(names, coef_vals, se_vals)}
                ci_upper = {n: v + 1.96 * se for n, v, se in zip(names, coef_vals, se_vals)}
            else:
                # Fallback: use exog_names from the statsmodels model
                names = list(getattr(getattr(raw, "model", None), "exog_names", None) or [])
                coef_arr = np.atleast_1d(raw.params)
                if not names:
                    names = [f"x{i}" for i in range(len(coef_arr))]
                params = dict(zip(names, coef_arr.tolist()))
                try:
                    ci_arr = np.atleast_2d(raw.conf_int())
                    ci_lower = dict(zip(names, ci_arr[:, 0].tolist()))
                    ci_upper = dict(zip(names, ci_arr[:, 1].tolist()))
                except Exception:
                    se_arr = np.atleast_1d(raw.bse)
                    ci_lower = {n: v - 1.96 * se for n, v, se in zip(names, coef_arr, se_arr)}
                    ci_upper = {n: v + 1.96 * se for n, v, se in zip(names, coef_arr, se_arr)}
            model_cls = type(getattr(raw, "model", raw)).__name__
            title = f"Coefficient Plot ({model_cls})"
            path = plot_coef(params, ci_lower, ci_upper, session.output_dir, title=title)
            session.plot_paths.append(str(path))
            return f"Coefficient plot saved: {path}"

        elif subcmd == "margins":
            mg = session._last_margins
            if mg is None:
                return "No margins result. Run 'margins' after logit/probit first."
            if not hasattr(mg, "effects"):
                return "Stored margins result has no 'effects' attribute."
            params = dict(mg.effects)
            ci_lower = dict(mg.conf_int_low)
            ci_upper = dict(mg.conf_int_high)
            path = plot_coef(
                params, ci_lower, ci_upper, session.output_dir,
                title=f"Marginal Effects ({mg.method})",
                drop_const=False,
            )
            session.plot_paths.append(str(path))
            return f"Marginal effects plot saved: {path}"

        elif subcmd == "interaction":
            # plot interaction <y> <x> <moderator> [--levels=N]
            if len(ca.positional) < 4:
                return "Usage: plot interaction <y> <x> <moderator> [--levels=3]"
            y_col = ca.positional[1]
            x_col = ca.positional[2]
            mod_col = ca.positional[3]
            n_levels = int(ca.options.get("levels", 3))
            path = plot_interaction(df, y_col, x_col, mod_col, session.output_dir, n_levels=n_levels)
            session.plot_paths.append(str(path))
            return f"Interaction plot saved: {path}"

        elif subcmd == "diagnostics":
            if session._last_model is None or session._last_model_vars is None:
                return "No model fitted yet. Run ols first, then plot diagnostics."
            from openstat.stats.models import compute_residuals
            dep, indeps = session._last_model_vars
            diag = compute_residuals(session._last_model, df, dep, indeps)
            paths = []
            paths.append(plot_residuals_vs_fitted(diag["fitted"], diag["residuals"], session.output_dir))
            paths.append(plot_qq(diag["std_residuals"], session.output_dir))
            paths.append(plot_scale_location(diag["fitted"], diag["std_residuals"], session.output_dir))
            session.plot_paths.extend(str(p) for p in paths)
            return "Diagnostic plots saved:\n" + "\n".join(f"  {p}" for p in paths)

        else:
            return f"Unknown plot type: {subcmd}. Available: hist, scatter, line, box, bar, heatmap, coef, margins, interaction, diagnostics"
    except Exception as e:
        return friendly_error(e, "Plot error")
