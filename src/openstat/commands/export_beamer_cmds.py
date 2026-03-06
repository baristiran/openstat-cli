"""Export results as LaTeX Beamer presentation."""

from __future__ import annotations
import os
from datetime import date
from pathlib import Path

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("export beamer", usage="export beamer [path] [--title=...] [--author=...] [--theme=Madrid]")
def cmd_export_beamer(session: Session, args: str) -> str:
    """Export analysis results as a LaTeX Beamer presentation (.tex).

    Generates a slide deck with dataset summary, model results,
    and references to saved plots.

    Options:
      --title=<txt>     presentation title (default: 'OpenStat Analysis')
      --author=<txt>    author name
      --theme=<t>       Beamer theme (default: Madrid)
      --colortheme=<t>  Beamer colour theme (default: beaver)
      --out=<path>      output .tex path

    Examples:
      export beamer
      export beamer results/slides.tex --title="Income Analysis" --author="J. Smith"
      export beamer --theme=Berlin --colortheme=whale
    """
    ca = CommandArgs(args)
    out_path = (
        ca.options.get("out")
        or (ca.positional[0] if ca.positional else None)
        or "outputs/presentation.tex"
    )
    title = ca.options.get("title", "OpenStat Analysis")
    author = ca.options.get("author", "OpenStat")
    theme = ca.options.get("theme", "Madrid")
    color_theme = ca.options.get("colortheme", "beaver")

    try:
        lines = []

        def L(s=""):
            lines.append(s)

        L(r"\documentclass{beamer}")
        L(r"\usetheme{" + theme + "}")
        L(r"\usecolortheme{" + color_theme + "}")
        L(r"\usepackage{booktabs}")
        L(r"\usepackage{graphicx}")
        L(r"\usepackage{amsmath}")
        L()
        L(r"\title{" + title.replace("_", r"\_") + "}")
        L(r"\author{" + author.replace("_", r"\_") + "}")
        L(r"\date{" + date.today().isoformat() + "}")
        L()
        L(r"\begin{document}")
        L()
        L(r"\begin{frame}")
        L(r"  \titlepage")
        L(r"\end{frame}")
        L()

        # Dataset overview slide
        ds_name = (session.dataset_name or "Unknown").replace("_", r"\_")
        shape_str = session.shape_str
        L(r"\begin{frame}{Dataset Overview}")
        L(r"  \begin{itemize}")
        L(r"    \item \textbf{Dataset:} " + ds_name)
        L(r"    \item \textbf{Shape:} " + shape_str)
        L(r"    \item \textbf{Date:} " + date.today().isoformat())
        L(r"  \end{itemize}")
        L(r"\end{frame}")
        L()

        # Model result slides
        for mr in session.results:
            model_title = f"{mr.name} — {mr.formula}"[:60].replace("_", r"\_")
            L(r"\begin{frame}{" + model_title + "}")
            L(r"  \scriptsize")
            L(r"  \begin{verbatim}")
            # Truncate table to fit slide
            table_lines = mr.table.split("\n")[:25]
            for tl in table_lines:
                L("  " + tl[:80])
            L(r"  \end{verbatim}")
            # Model stats
            d = mr.details
            stats_parts = []
            if d.get("n"):
                stats_parts.append(f"N={d['n']}")
            if d.get("r2") is not None:
                stats_parts.append(f"R²={d['r2']:.3f}")
            if d.get("aic") is not None:
                stats_parts.append(f"AIC={d['aic']:.1f}")
            if stats_parts:
                L(r"  \medskip")
                L(r"  \normalsize " + " \\quad ".join(stats_parts))
            L(r"\end{frame}")
            L()

        # Plot slides
        for plot_path in session.plot_paths:
            if os.path.exists(plot_path):
                safe_path = plot_path.replace("\\", "/").replace("_", r"\_")
                base = os.path.basename(plot_path).replace("_", r"\_")
                raw_path = plot_path.replace("\\", "/")
                L(r"\begin{frame}{" + base + "}")
                L(r"  \centering")
                L(r"  \includegraphics[width=0.85\textwidth]{" + raw_path + "}")
                L(r"\end{frame}")
                L()

        # Commands history slide
        if session.history:
            L(r"\begin{frame}[fragile]{Command History}")
            L(r"  \scriptsize")
            L(r"  \begin{verbatim}")
            for h in session.history[-15:]:
                L("  " + h[:80])
            L(r"  \end{verbatim}")
            L(r"\end{frame}")
            L()

        L(r"\end{document}")
        L()

        tex_content = "\n".join(lines)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(tex_content, encoding="utf-8")

        abs_path = os.path.abspath(out_path)
        n_slides = tex_content.count(r"\begin{frame}")
        return (
            f"LaTeX Beamer presentation saved: {abs_path}\n"
            f"  Slides: {n_slides}  Theme: {theme}/{color_theme}\n"
            f"  Compile: pdflatex {out_path}"
        )
    except Exception as e:
        return friendly_error(e, "export beamer")
