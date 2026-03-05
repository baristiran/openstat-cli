"""Plot commands: hist, scatter, line, box, bar, heatmap."""

from __future__ import annotations

from openstat.session import Session
from openstat.plots.plotter import (
    plot_histogram, plot_scatter, plot_line, plot_box, plot_bar, plot_heatmap,
    plot_residuals_vs_fitted, plot_qq, plot_scale_location,
)
from openstat.commands.base import command, CommandArgs, friendly_error


@command("plot", usage="plot hist|scatter|line|box|bar|heatmap ...")
def cmd_plot(session: Session, args: str) -> str:
    """Create plots: hist, scatter, line, box, bar, heatmap."""
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
            "  plot diagnostics   (after fitting a model)"
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
            return f"Unknown plot type: {subcmd}. Available: hist, scatter, line, box, bar, heatmap, diagnostics"
    except Exception as e:
        return friendly_error(e, "Plot error")
