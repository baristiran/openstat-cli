"""Extra export commands: Jupyter notebook, APA text, progress bars."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ── Jupyter Notebook export ──────────────────────────────────────────────────

@command("export ipynb", usage="export ipynb [path]")
def cmd_export_ipynb(session: Session, args: str) -> str:
    """Export session history as a Jupyter notebook (.ipynb).

    Creates a notebook where each command becomes a code cell with
    its output. Requires: pip install nbformat

    Examples:
      export ipynb
      export ipynb my_analysis.ipynb
    """
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    except ImportError:
        return "nbformat required. Install: pip install nbformat"

    ca = CommandArgs(args)
    out_path = ca.positional[0] if ca.positional else "outputs/analysis.ipynb"

    nb = new_notebook()
    cells = []

    # Title cell
    cells.append(new_markdown_cell(
        f"# OpenStat Analysis\n\n"
        f"**Dataset:** {session.dataset_name or 'Unknown'}  \n"
        f"**Date:** {date.today().isoformat()}  \n"
        f"**Shape:** {session.shape_str}"
    ))

    # Setup cell
    cells.append(new_code_cell(
        "# Auto-generated from OpenStat session\n"
        "# Run: pip install openstat\n"
        "from openstat.session import Session\n"
        "from openstat.commands import COMMANDS\n"
        "session = Session()\n"
        "\ndef run(cmd):\n"
        "    parts = cmd.split(None, 1)\n"
        "    name = parts[0]\n"
        "    args = parts[1] if len(parts) > 1 else ''\n"
        "    return COMMANDS[name](session, args)\n"
    ))

    # One cell per history command
    for cmd_line in session.history:
        if cmd_line.strip().startswith("#"):
            cells.append(new_markdown_cell(cmd_line.lstrip("# ")))
        else:
            safe_cmd = cmd_line.replace("'", "\\'")
            cells.append(new_code_cell(f"print(run('{safe_cmd}'))"))

    # Plots cell
    if session.plot_paths:
        plot_code = "from IPython.display import Image, display\n"
        for p in session.plot_paths:
            if os.path.exists(p):
                plot_code += f"display(Image('{p}'))\n"
        cells.append(new_code_cell(plot_code))

    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    n_cells = len(cells)
    return f"Jupyter notebook saved: {os.path.abspath(out_path)} ({n_cells} cells)"


# ── APA export ───────────────────────────────────────────────────────────────

@command("export apa", usage="export apa [path]")
def cmd_export_apa(session: Session, args: str) -> str:
    """Export model results in APA 7th edition format.

    Generates text suitable for inclusion in research papers.
    Supports OLS, logit, probit, and other regression models.

    Examples:
      ols income educ age
      export apa
      export apa results/apa_table.txt
    """
    import polars as pl

    ca = CommandArgs(args)
    out_path = ca.options.get("out")

    lines = [
        f"APA-Formatted Results",
        f"Generated: {date.today().isoformat()}",
        f"Dataset: {session.dataset_name or 'Unknown'} ({session.shape_str})",
        "",
    ]

    if not session.results:
        return "No model results to export. Run ols/logit/etc. first."

    for mr in session.results:
        model_type = mr.name
        formula = mr.formula
        details = mr.details

        n = details.get("n", "?")
        r2 = details.get("r2")
        adj_r2 = details.get("adj_r2")
        f_stat = details.get("f_stat")
        f_pval = details.get("f_pval")
        aic = details.get("aic")
        bic = details.get("bic")
        ll = details.get("log_likelihood")

        lines.append(f"{'='*60}")
        lines.append(f"Model: {model_type} — {formula}")
        lines.append("")

        # APA regression table header
        if model_type.upper() in ("OLS", "LINEAR"):
            if r2 is not None:
                lines.append(
                    f"A multiple linear regression was conducted to predict {formula.split('~')[0].strip()} "
                    f"from {formula.split('~')[1].strip() if '~' in formula else 'the predictors'}."
                )
                r2_str = f"R² = {r2:.3f}" if r2 is not None else ""
                adj_r2_str = f", adjusted R² = {adj_r2:.3f}" if adj_r2 is not None else ""
                f_str = f", F = {f_stat:.2f}" if f_stat is not None else ""
                p_str = f", p {'< .001' if (f_pval is not None and f_pval < 0.001) else f'= {f_pval:.3f}'}" if f_pval is not None else ""
                lines.append(f"The model was statistically significant: {r2_str}{adj_r2_str}{f_str}{p_str}.")
                lines.append(f"N = {n}.")
        else:
            if ll is not None:
                lines.append(f"N = {n}.")
                if aic is not None:
                    lines.append(f"AIC = {aic:.2f}, BIC = {bic:.2f}." if bic is not None else f"AIC = {aic:.2f}.")

        lines.append("")
        lines.append("Table. Regression Coefficients")
        lines.append(f"  {'Variable':<25} {'B':>10}  {'SE':>8}  {'p':>8}")
        lines.append("  " + "-" * 55)

        # Parse table for coefficients
        for line in mr.table.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("-"):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    coef = float(parts[-4]) if len(parts) >= 4 else float(parts[1])
                    se = float(parts[-3]) if len(parts) >= 3 else None
                    p = float(parts[-1]) if len(parts) >= 1 else None
                    varname = " ".join(parts[:-4]) if len(parts) > 4 else parts[0]
                    sig = "***" if p is not None and p < 0.001 else "**" if p is not None and p < 0.01 else "*" if p is not None and p < 0.05 else ""
                    lines.append(f"  {varname:<25} {coef:10.3f}  {se:8.3f}  {p:8.3f}{sig}" if se and p else f"  {varname:<25} {coef:10.3f}")
                except (ValueError, IndexError):
                    continue

        lines.append("  " + "-" * 55)
        lines.append("  Note. * p < .05. ** p < .01. *** p < .001.")
        lines.append("")

    text = "\n".join(lines)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
        return f"APA results saved: {os.path.abspath(out_path)}"

    if ca.positional:
        p = ca.positional[0]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(text, encoding="utf-8")
        return f"APA results saved: {os.path.abspath(p)}"

    return text


# ── Progress bar wrapper for long commands ────────────────────────────────────

@command("progress", usage="progress <command with args>")
def cmd_progress(session: Session, args: str) -> str:
    """Run a command with a live progress indicator.

    Useful for long-running commands like bootstrap, permtest, hyperopt.
    Uses rich progress bar.

    Examples:
      progress bootstrap ols income educ age --reps=2000
      progress hyperopt income educ age --model=rf --n_iter=50
    """
    from openstat.commands.base import run_command
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    import threading
    import time

    if not args.strip():
        return "Usage: progress <command> [args]"

    result_holder = {"result": None, "done": False}

    def _run():
        result_holder["result"] = run_command(session, args.strip())
        result_holder["done"] = True

    thread = threading.Thread(target=_run, daemon=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task(f"Running: {args[:50]}...", total=None)
        thread.start()
        while not result_holder["done"]:
            time.sleep(0.1)
        thread.join()

    return result_holder["result"] or ""
