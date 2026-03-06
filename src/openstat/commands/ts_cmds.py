"""Time series commands: tsset, arima, var, dfuller, forecast, irf."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, rich_to_str, friendly_error
from openstat.dsl.parser import parse_formula, ParseError


@command("tsset", usage="tsset <time_var> [freq=M|Q|Y|D]")
def cmd_tsset(session: Session, args: str) -> str:
    """Declare the time variable for time series analysis."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: tsset <time_var> [freq=M|Q|Y|D]"

    time_var = ca.positional[0]
    if time_var not in df.columns:
        return f"Column not found: {time_var}"

    session._time_var = time_var
    session._ts_freq = ca.get_option("freq")
    n = df[time_var].n_unique()
    return f"Time variable: {time_var} ({n} unique values)" + (
        f"\nFrequency: {session._ts_freq}" if session._ts_freq else ""
    )


@command("dfuller", usage="dfuller <variable>")
def cmd_dfuller(session: Session, args: str) -> str:
    """Augmented Dickey-Fuller unit root test."""
    df = session.require_data()
    var_name = args.strip()
    if not var_name:
        return "Usage: dfuller <variable>"
    if var_name not in df.columns:
        return f"Column not found: {var_name}"

    from openstat.stats.timeseries import adf_test
    series = df[var_name].drop_nulls().to_numpy()
    return adf_test(series, var_name)


@command("arima", usage="arima y [~ x1], order(p,d,q)")
def cmd_arima(session: Session, args: str) -> str:
    """Fit ARIMA(p,d,q) or ARIMAX model."""
    df = session.require_data()

    # Parse order(p,d,q)
    m = re.search(r'order\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)', args)
    if not m:
        return "Usage: arima y [~ x1], order(p,d,q)"
    order = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Remove order(...) from args for formula parsing
    formula_str = args[:m.start()] + args[m.end():]
    formula_str = formula_str.strip().rstrip(",").strip()

    # Parse formula
    if "~" in formula_str:
        try:
            dep, exog_vars = parse_formula(formula_str)
        except ParseError as e:
            return f"Formula error: {e}"
    else:
        parts = formula_str.split()
        dep = parts[0] if parts else ""
        exog_vars = None

    if not dep or dep not in df.columns:
        return f"Dependent variable not found: {dep}"

    try:
        from openstat.stats.timeseries import fit_arima

        result, raw = fit_arima(df, dep, order, exog_vars, session._time_var)

        session._last_model = raw
        session._last_model_vars = (dep, exog_vars or [])
        session._last_fit_result = result
        session._last_fit_kwargs = {"order": order}

        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name=f"ARIMA{order}", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
                "aic": result.aic,
                "bic": result.bic,
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)
        return output
    except Exception as e:
        return friendly_error(e, "arima")


@command("var", usage="var y1 y2 [y3], lags(n)")
def cmd_var(session: Session, args: str) -> str:
    """Fit a Vector Autoregression (VAR) model."""
    df = session.require_data()

    # Parse lags(n)
    m = re.search(r'lags\((\d+)\)', args)
    if not m:
        return "Usage: var y1 y2, lags(n)"
    lags = int(m.group(1))

    # Parse variable list
    var_str = args[:m.start()].strip().rstrip(",").strip()
    variables = var_str.split()
    if len(variables) < 2:
        return "VAR requires at least 2 variables."

    missing = [v for v in variables if v not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        from openstat.stats.timeseries import fit_var

        summary, raw = fit_var(df, variables, lags, session._time_var)

        session._last_model = raw
        session._last_model_vars = (variables[0], variables[1:])
        session._last_fit_kwargs = {"lags": lags, "variables": variables}

        session.results.append(ModelResult(
            name=f"VAR({lags})", formula=f"VAR({', '.join(variables)})",
            table=summary, details={"lags": lags, "variables": variables},
        ))

        return summary
    except Exception as e:
        return friendly_error(e, "var")


@command("forecast", usage="forecast <steps>")
def cmd_forecast(session: Session, args: str) -> str:
    """Generate forecasts from the last fitted time series model."""
    if session._last_model is None:
        return "No model fitted. Run arima or var first."

    steps = int(args.strip()) if args.strip().isdigit() else 12

    try:
        from openstat.stats.timeseries import forecast_model
        fc = forecast_model(session._last_model, steps)

        def render(console: Console) -> None:
            table = Table(title=f"Forecast ({steps} steps)")
            table.add_column("Step", justify="right")
            if fc.ndim == 1:
                table.add_column("Forecast", justify="right")
                for i, val in enumerate(fc, 1):
                    table.add_row(str(i), f"{val:.4f}")
            else:
                vars_list = session._last_fit_kwargs.get("variables", [])
                for v in vars_list:
                    table.add_column(v, justify="right")
                for i, row in enumerate(fc, 1):
                    table.add_row(str(i), *[f"{v:.4f}" for v in row])
            console.print(table)

        return rich_to_str(render)
    except Exception as e:
        return friendly_error(e, "forecast")


@command("irf", usage="irf [steps=N]")
def cmd_irf(session: Session, args: str) -> str:
    """Impulse response functions for VAR model."""
    if session._last_model is None:
        return "No model fitted. Run var first."

    ca = CommandArgs(args)
    steps = int(ca.get_option("steps", "10"))

    try:
        from openstat.stats.timeseries import compute_irf
        return compute_irf(session._last_model, steps)
    except Exception as e:
        return friendly_error(e, "irf")
