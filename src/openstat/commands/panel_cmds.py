"""Panel data commands: xtset, xtreg, hausman."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.dsl.parser import parse_formula, ParseError


def _store_panel_model(session, result, raw_model, dep, indeps, fit_kwargs=None):
    """Store panel model result in session."""
    session._last_model = raw_model
    session._last_model_vars = (dep, indeps)
    session._last_fit_result = result
    session._last_fit_kwargs = fit_kwargs or {}
    md = result.to_markdown() if hasattr(result, "to_markdown") else str(result)
    details = {
        "n_obs": result.n_obs,
        "params": dict(result.params),
        "std_errors": dict(result.std_errors),
    }
    if result.r_squared is not None:
        details["r_squared"] = result.r_squared
    session.results.append(ModelResult(
        name=result.model_type, formula=result.formula,
        table=md, details=details,
    ))
    output = result.summary_table()
    if result.warnings:
        output += "\n" + "\n".join(result.warnings)
    return output


@command("xtset", usage="xtset <panel_var> <time_var>")
def cmd_xtset(session: Session, args: str) -> str:
    """Declare panel structure: entity and time variables."""
    df = session.require_data()
    parts = args.strip().split()
    if len(parts) < 2:
        return "Usage: xtset <panel_var> <time_var>"

    panel_var, time_var = parts[0], parts[1]
    for v in (panel_var, time_var):
        if v not in df.columns:
            return f"Column not found: {v}"

    n_entities = df[panel_var].n_unique()
    n_periods = df[time_var].n_unique()
    session._panel_var = panel_var
    session._time_var = time_var

    return (
        f"Panel variable: {panel_var} ({n_entities} entities)\n"
        f"Time variable: {time_var} ({n_periods} periods)\n"
        f"Observations: {df.height}"
    )


@command("xtreg", usage="xtreg y ~ x1 + x2, fe|re|be [--robust] [--cluster]")
def cmd_xtreg(session: Session, args: str) -> str:
    """Fit panel data regression: fixed effects, random effects, or between."""
    df = session.require_data()

    if session._panel_var is None or session._time_var is None:
        return "Panel structure not set. Use: xtset <panel_var> <time_var>"

    # Split on comma to get formula and estimator
    if "," in args:
        formula_part, options_part = args.rsplit(",", 1)
    else:
        return "Usage: xtreg y ~ x1 + x2, fe|re|be [--robust]"

    ca = CommandArgs(options_part)
    estimator = None
    for est in ("fe", "re", "be"):
        if est in [p.lower() for p in ca.positional]:
            estimator = est
            break
    if estimator is None:
        return "Specify estimator: fe (fixed effects), re (random effects), or be (between)"

    robust = ca.has_flag("--robust") or "--robust" in formula_part
    cluster = ca.get_option("cluster")
    formula_clean = formula_part.replace("--robust", "").strip()
    # Remove cluster option from formula
    formula_clean = re.sub(r'--cluster=\S+', '', formula_clean).strip()

    try:
        dep, indeps = parse_formula(formula_clean)
    except ParseError as e:
        return f"Formula error: {e}"

    try:
        from openstat.stats.panel import fit_panel_fe, fit_panel_re, fit_panel_be

        if estimator == "fe":
            result, raw = fit_panel_fe(
                df, dep, indeps, session._panel_var, session._time_var,
                robust=robust, cluster=cluster,
            )
        elif estimator == "re":
            result, raw = fit_panel_re(
                df, dep, indeps, session._panel_var, session._time_var,
                robust=robust,
            )
        else:  # be
            result, raw = fit_panel_be(
                df, dep, indeps, session._panel_var, session._time_var,
            )

        # Store raw model for hausman test
        session._panel_models[estimator] = raw

        return _store_panel_model(
            session, result, raw, dep, indeps,
            {"estimator": estimator, "robust": robust},
        )
    except ImportError as e:
        return str(e)
    except Exception as e:
        return friendly_error(e, "xtreg")


@command("hausman", usage="hausman")
def cmd_hausman(session: Session, args: str) -> str:
    """Hausman test: compare FE vs RE. Run both xtreg fe and xtreg re first."""
    fe_raw = session._panel_models.get("fe")
    re_raw = session._panel_models.get("re")

    if not fe_raw or not re_raw:
        return "Run both 'xtreg ..., fe' and 'xtreg ..., re' first."

    try:
        from openstat.stats.panel import hausman_test
        return hausman_test(fe_raw, re_raw)
    except ImportError as e:
        return str(e)
    except Exception as e:
        return friendly_error(e, "hausman")
