"""outreg: export regression results to LaTeX or HTML tables. log: session logging."""

from __future__ import annotations

import re
from datetime import datetime

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


def _results_to_table(results, fmt: str = "latex", stars: bool = True) -> str:
    """Convert session results to LaTeX or HTML table."""
    # Normalize ModelResult → dict
    model_list = []
    for res in results:
        if isinstance(res, dict) and ("params" in res or "coefficients" in res):
            model_list.append(res)
        elif hasattr(res, "details"):
            d = res.details
            if "params" in d or "coefficients" in d:
                model_list.append(d)

    if not model_list:
        return ""

    all_params: list[str] = []
    for res in model_list:
        src = res.get("params") or res.get("coefficients") or {}
        for p in src:
            if p not in all_params:
                all_params.append(p)

    def _coef(res, param):
        if "params" in res:
            return res["params"].get(param, float("nan"))
        c = res.get("coefficients", {}).get(param, {})
        return c.get("mean", float("nan"))

    def _se(res, param):
        if "std_errors" in res:
            return res["std_errors"].get(param, float("nan"))
        c = res.get("coefficients", {}).get(param, {})
        return c.get("std", float("nan"))

    def _pval(res, param):
        if "p_values" in res:
            return res["p_values"].get(param, float("nan"))
        return float("nan")

    def _star(p):
        if p != p: return ""
        if p < 0.001: return "^{***}" if fmt == "latex" else "***"
        if p < 0.01: return "^{**}" if fmt == "latex" else "**"
        if p < 0.05: return "^{*}" if fmt == "latex" else "*"
        return ""

    n_models = len(model_list)

    if fmt == "latex":
        col_spec = "l" + "c" * n_models
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\hline\\hline",
        ]
        header = " & " + " & ".join(f"({i+1})" for i in range(n_models)) + " \\\\"
        lines.append(header)
        lines.append("\\hline")
        for param in all_params:
            coef_row = param.replace("_", "\\_") + " & "
            se_row = " & "
            vals = []
            ses = []
            for res in model_list:
                c = _coef(res, param)
                s = _se(res, param)
                p = _pval(res, param)
                st = _star(p) if stars else ""
                vals.append(f"{c:.4f}{st}" if c == c else "")
                ses.append(f"({s:.4f})" if s == s else "")
            coef_row += " & ".join(vals) + " \\\\"
            se_row += " & ".join(ses) + " \\\\"
            lines.append(coef_row)
            lines.append(se_row)
        lines.append("\\hline")
        n_row = "N & " + " & ".join(str(res.get("n_obs", "")) for res in model_list) + " \\\\"
        r2_row = "$R^2$ & " + " & ".join(
            f"{res.get('r_squared', float('nan')):.4f}" if isinstance(res.get('r_squared'), float) else ""
            for res in model_list
        ) + " \\\\"
        lines.extend([n_row, r2_row, "\\hline\\hline"])
        if stars:
            lines.append("\\multicolumn{" + str(n_models + 1) + "}{l}{\\footnotesize $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}")
        lines.extend(["\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)

    else:  # html
        lines = ["<table border='1' cellpadding='5' style='border-collapse:collapse'>"]
        lines.append("<tr><th>Variable</th>" + "".join(f"<th>({i+1})</th>" for i in range(n_models)) + "</tr>")
        for param in all_params:
            coef_cells = ""
            se_cells = ""
            for res in model_list:
                c = _coef(res, param)
                s = _se(res, param)
                p = _pval(res, param)
                st = _star(p) if stars else ""
                coef_cells += f"<td align='center'>{f'{c:.4f}{st}' if c==c else ''}</td>"
                se_cells += f"<td align='center'><small>{f'({s:.4f})' if s==s else ''}</small></td>"
            lines.append(f"<tr><td><b>{param}</b></td>{coef_cells}</tr>")
            lines.append(f"<tr><td></td>{se_cells}</tr>")
        n_row = "<tr><td>N</td>" + "".join(f"<td align='center'>{res.get('n_obs','')}</td>" for res in model_list) + "</tr>"
        def _r2_cell(res):
            r2 = res.get("r_squared", None)
            val = f"{r2:.4f}" if isinstance(r2, float) else ""
            return f"<td align='center'>{val}</td>"
        r2_row = "<tr><td>R\u00b2</td>" + "".join(_r2_cell(res) for res in model_list) + "</tr>"
        lines.extend([n_row, r2_row, "</table>"])
        if stars: lines.append("<p><small>* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</small></p>")
        return "\n".join(lines)


@command("outreg", usage="outreg [using path] [format(latex|html)] [--stars]")
def cmd_outreg(session: Session, args: str) -> str:
    """Export regression comparison table to LaTeX or HTML."""
    positional, opts = _stata_opts(args)
    fmt = opts.get("format", "latex")
    show_stars = "stars" in args or "--stars" in args
    path = opts.get("using")
    # Also accept: outreg using file.tex
    if not path and "using" in positional:
        idx = positional.index("using")
        if idx + 1 < len(positional):
            path = positional[idx + 1]

    if not session.results:
        return "No stored results. Run regression commands first."

    table = _results_to_table(session.results, fmt=fmt, stars=show_stars)
    if not table:
        return "No regression results with coefficients found."

    if path:
        try:
            import os
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(table)
            return f"outreg: table saved to {path} ({fmt} format)"
        except Exception as exc:
            return f"outreg write error: {exc}"

    return f"\noutreg ({fmt}):\n\n{table}"


@command("log", usage="log using <path> | log close | log status | log display | log clear")
def cmd_log(session: Session, args: str) -> str:
    """Session logging: real-time capture or history export.

    Examples:
      log using analysis.log   — start real-time logging
      log status               — check if logging is active
      log close                — stop real-time log and close file
      log display              — print command history to screen
      log clear                — clear command history
    """
    import os
    from pathlib import Path

    stripped = args.strip()
    positional, opts = _stata_opts(args)
    subcmd = positional[0].lower() if positional else "display"

    # ---- Real-time logging ----
    if subcmd == "using":
        log_path = positional[1] if len(positional) >= 2 else opts.get("using", "outputs/session.log")
        # Close existing real-time log if open
        if session._log_file is not None:
            try:
                session._log_file.write(f"\nLog replaced {datetime.now().isoformat()}\n")
                session._log_file.close()
            except Exception:
                pass
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            session._log_file = open(path, "w", encoding="utf-8")
            session._log_path = str(path)
            session._log_file.write("OpenStat session log\n")
            session._log_file.write(f"Started: {datetime.now().isoformat()}\n")
            session._log_file.write("=" * 60 + "\n\n")
            session._log_file.flush()
            return f"Log opened: {path}"
        except OSError as exc:
            return f"log error: {exc}"

    elif subcmd in ("close", "off"):
        if session._log_file is None:
            return "No active log. Use: log using <path>"
        closed_path = session._log_path
        try:
            session._log_file.write(f"\nLog closed {datetime.now().isoformat()}\n")
            session._log_file.close()
        except Exception:
            pass
        session._log_file = None
        session._log_path = None
        return f"Log closed: {closed_path}"

    elif subcmd == "status":
        if session._log_file is None:
            return "Logging: OFF"
        return f"Logging: ON  →  {session._log_path}"

    # ---- History display / export ----
    elif subcmd == "display":
        if not session.history:
            return "No commands in session history."
        lines = ["\nSession Log:", "=" * 50]
        for i, cmd_line in enumerate(session.history, 1):
            lines.append(f"  {i:>4}. {cmd_line}")
        return "\n".join(lines)

    elif subcmd == "clear":
        session.history.clear()
        return "Session history cleared."

    else:
        return (
            "Usage:\n"
            "  log using <path>   — start real-time logging to file\n"
            "  log status         — check if logging is active\n"
            "  log close          — stop logging\n"
            "  log display        — show command history\n"
            "  log clear          — clear command history"
        )
