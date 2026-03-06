"""Advanced visualization: 3D, interactive, animated, missing, map."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("plot3d", usage="plot3d <x> <y> <z> [--color=<col>]")
def cmd_plot3d(session: Session, args: str) -> str:
    """3D scatter plot.

    Examples:
      plot3d x y z
      plot3d age income educ --color=gender
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    import polars as pl

    ca = CommandArgs(args)
    if len(ca.positional) < 3:
        return "Usage: plot3d <x> <y> <z> [--color=<col>]"
    x_col, y_col, z_col = ca.positional[:3]
    color_col = ca.options.get("color")

    try:
        df = session.require_data()
        for c in [x_col, y_col, z_col]:
            if c not in df.columns:
                return f"Column not found: {c}"

        cols = [x_col, y_col, z_col]
        if color_col and color_col in df.columns:
            cols.append(color_col)
        sub = df.select(cols).drop_nulls()

        x = sub[x_col].to_numpy().astype(float)
        y = sub[y_col].to_numpy().astype(float)
        z = sub[z_col].to_numpy().astype(float)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        if color_col and color_col in sub.columns:
            cats = sub[color_col].cast(pl.Utf8).to_list()
            unique_cats = sorted(set(cats))
            cmap = plt.colormaps.get_cmap("tab10")
            for i, cat in enumerate(unique_cats):
                mask = [c == cat for c in cats]
                ax.scatter(x[mask], y[mask], z[mask],
                           label=str(cat), alpha=0.7, color=cmap(i / max(len(unique_cats), 1)))
            ax.legend(title=color_col)
        else:
            ax.scatter(x, y, z, alpha=0.6, color="#4C72B0")

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Scatter: {x_col} × {y_col} × {z_col}")
        fig.tight_layout()

        session.output_dir.mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        path = session.output_dir / "scatter3d.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        session.plot_paths.append(str(path))
        return f"3D scatter plot saved: {path}"
    except Exception as e:
        return friendly_error(e, "plot3d")


@command("plotmissing", usage="plotmissing [cols...]")
def cmd_plotmissing(session: Session, args: str) -> str:
    """Missing data heatmap: visualise missingness patterns across columns.

    Shows rows (sampled if large) × columns with missing/present colour coding.

    Examples:
      plotmissing
      plotmissing income age education
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    ca = CommandArgs(args)
    try:
        df = session.require_data()
        cols = ca.positional if ca.positional else df.columns

        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            return f"Columns not found: {', '.join(missing_cols)}"

        # Sample up to 500 rows for readability
        MAX_ROWS = 500
        if df.height > MAX_ROWS:
            import random
            idx = sorted(random.sample(range(df.height), MAX_ROWS))
            sub = df[idx].select(cols)
        else:
            sub = df.select(cols)

        # Build missing matrix
        mat = np.zeros((sub.height, len(cols)))
        for j, c in enumerate(cols):
            mat[:, j] = sub[c].is_null().to_numpy().astype(float)

        miss_pct = [f"{100*df[c].null_count()/df.height:.0f}%" for c in cols]
        labels = [f"{c}\n({p})" for c, p in zip(cols, miss_pct)]

        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5 + 2), 6))
        ax.imshow(mat.T, aspect="auto", cmap="RdYlGn_r", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Rows")
        ax.set_title("Missingness Heatmap (red=missing, green=present)")
        fig.tight_layout()

        session.output_dir.mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        path = session.output_dir / "missing_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        session.plot_paths.append(str(path))

        # Summary table
        lines = [f"Missing data heatmap saved: {path}", ""]
        lines.append(f"  {'Column':<25} {'Missing':>8} {'%':>6}")
        lines.append("  " + "-" * 42)
        for c in cols:
            n = df[c].null_count()
            p = 100 * n / df.height
            lines.append(f"  {c:<25} {n:>8,} {p:>5.1f}%")
        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "plotmissing")


@command("plotinteractive", usage="plotinteractive scatter|bar|line|hist <args> [--out=plot.html]")
def cmd_plotinteractive(session: Session, args: str) -> str:
    """Interactive Plotly chart (saved as HTML — open in browser).

    Sub-commands: scatter, bar, line, hist, box, heatmap

    Options:
      --out=<path>   output HTML file (default: outputs/interactive_plot.html)
      --title=<txt>  chart title

    Examples:
      plotinteractive scatter y x
      plotinteractive hist income --out=income_dist.html
      plotinteractive bar category value
      plotinteractive line date price --title="Price Over Time"
    """
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        return "plotly required. Install: pip install plotly"

    import polars as pl
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: plotinteractive scatter|bar|line|hist|box|heatmap <args>"

    subcmd = ca.positional[0].lower()
    out_path = ca.options.get("out", str(session.output_dir / "interactive_plot.html"))
    title = ca.options.get("title", f"Interactive {subcmd.title()}")

    try:
        df = session.require_data()
        pd_df = df.to_pandas()

        if subcmd == "scatter":
            if len(ca.positional) < 3:
                return "Usage: plotinteractive scatter <y> <x>"
            y_col, x_col = ca.positional[1], ca.positional[2]
            color_col = ca.options.get("color")
            fig = px.scatter(pd_df, x=x_col, y=y_col, color=color_col, title=title, trendline="ols")

        elif subcmd == "hist":
            if len(ca.positional) < 2:
                return "Usage: plotinteractive hist <col>"
            col = ca.positional[1]
            fig = px.histogram(pd_df, x=col, title=title, nbins=30)

        elif subcmd == "bar":
            if len(ca.positional) < 3:
                return "Usage: plotinteractive bar <category> <value>"
            cat_col, val_col = ca.positional[1], ca.positional[2]
            fig = px.bar(pd_df, x=cat_col, y=val_col, title=title)

        elif subcmd == "line":
            if len(ca.positional) < 3:
                return "Usage: plotinteractive line <x> <y>"
            x_col, y_col = ca.positional[1], ca.positional[2]
            color_col = ca.options.get("color")
            fig = px.line(pd_df, x=x_col, y=y_col, color=color_col, title=title)

        elif subcmd == "box":
            if len(ca.positional) < 2:
                return "Usage: plotinteractive box <col> [by <group>]"
            col = ca.positional[1]
            by_raw = ca.rest_after("by")
            group = by_raw.strip().split()[0] if by_raw else None
            fig = px.box(pd_df, y=col, x=group, title=title)

        elif subcmd == "heatmap":
        
            NUMERIC = ["float32", "float64", "int32", "int64", "int8", "int16"]
            num_cols = [c for c in df.columns if str(df[c].dtype).lower() in NUMERIC]
            corr = df.select(num_cols).to_pandas().corr()
            fig = px.imshow(corr, title="Correlation Heatmap", color_continuous_scale="RdBu", zmin=-1, zmax=1)

        else:
            return f"Unknown sub-command: {subcmd}"

        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, out_path)
        return f"Interactive {subcmd} chart saved: {out_path}\nOpen in a web browser to view."
    except Exception as e:
        return friendly_error(e, "plotinteractive")


@command("plotanimated", usage="plotanimated <y> <x> <time_col> [--out=anim.gif]")
def cmd_plotanimated(session: Session, args: str) -> str:
    """Animated line/scatter plot over a time variable (saved as GIF).

    Shows how the relationship between x and y changes across time steps.

    Examples:
      plotanimated sales month year --out=sales_trend.gif
      plotanimated price date --out=price_animation.gif
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    import polars as pl

    ca = CommandArgs(args)
    if len(ca.positional) < 3:
        return "Usage: plotanimated <y> <x> <time_col> [--out=anim.gif]"

    y_col, x_col, t_col = ca.positional[0], ca.positional[1], ca.positional[2]
    out_path = ca.options.get("out", str(session.output_dir / "animated.gif"))

    try:
        df = session.require_data()
        for c in [y_col, x_col, t_col]:
            if c not in df.columns:
                return f"Column not found: {c}"

        sub = df.select([t_col, x_col, y_col]).drop_nulls().sort(t_col)
        time_vals = sub[t_col].cast(pl.Utf8).to_list()
        unique_times = sorted(set(time_vals))

        if len(unique_times) < 2:
            return f"Need at least 2 unique values in '{t_col}' for animation."
        if len(unique_times) > 60:
            unique_times = unique_times[:60]  # cap

        x_all = sub[x_col].to_numpy().astype(float)
        y_all = sub[y_col].to_numpy().astype(float)
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter([], [], alpha=0.7, color="#4C72B0", s=30)
        ax.set_xlim(x_min - abs(x_min) * 0.05, x_max + abs(x_max) * 0.05)
        ax.set_ylim(y_min - abs(y_min) * 0.05, y_max + abs(y_max) * 0.05)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        title_obj = ax.set_title("")

        def update(frame):
            t = unique_times[frame]
            mask = [tv == t for tv in time_vals]
            xd = x_all[mask]
            yd = y_all[mask]
            scatter.set_offsets(np.column_stack([xd, yd]))
            title_obj.set_text(f"{t_col}: {t}")
            return scatter, title_obj

        ani = animation.FuncAnimation(fig, update, frames=len(unique_times),
                                      interval=300, blit=False)

        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(out_path, writer="pillow", fps=3)
        except Exception:
            # Fallback: save as MP4 or just first frame
            out_path = out_path.replace(".gif", ".png")
            update(0)
            fig.savefig(out_path, dpi=120)
        plt.close(fig)
        session.plot_paths.append(out_path)
        return f"Animated plot saved: {out_path}  ({len(unique_times)} frames)"
    except Exception as e:
        return friendly_error(e, "plotanimated")
