"""Survival analysis commands: stset, stcox, sts."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, friendly_error


@command("stset", usage="stset <time_var>, failure(<event_var>)")
def cmd_stset(session: Session, args: str) -> str:
    """Declare survival time and failure event variables."""
    df = session.require_data()

    m = re.search(r'failure\((\w+)\)', args)
    if not m:
        return "Usage: stset <time_var>, failure(<event_var>)"

    event_var = m.group(1)
    # Time var is everything before the comma
    time_part = args[:args.index(',')] if ',' in args else args[:m.start()]
    time_var = time_part.strip()

    if time_var not in df.columns:
        return f"Column not found: {time_var}"
    if event_var not in df.columns:
        return f"Column not found: {event_var}"

    session._surv_time_var = time_var
    session._surv_event_var = event_var

    n = df.height
    events = df[event_var].sum()
    return (
        f"Survival time: {time_var}\n"
        f"Failure event: {event_var}\n"
        f"Observations: {n}, Events: {events}, Censored: {n - events}"
    )


@command("stcox", usage="stcox x1 x2 [--robust]")
def cmd_stcox(session: Session, args: str) -> str:
    """Fit a Cox Proportional Hazards model."""
    df = session.require_data()

    if session._surv_time_var is None or session._surv_event_var is None:
        return "Survival structure not set. Use: stset <time_var>, failure(<event_var>)"

    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    covariates = [p for p in ca.positional if not p.startswith("--")]

    if not covariates:
        return "Usage: stcox x1 x2 [--robust]"

    missing = [c for c in covariates if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        from openstat.stats.survival import fit_cox_ph

        result, raw = fit_cox_ph(
            df, session._surv_time_var, session._surv_event_var,
            covariates, robust=robust,
        )

        session._last_model = raw
        session._last_model_vars = (session._surv_time_var, covariates)
        session._last_fit_result = result
        session._last_fit_kwargs = {"survival": True}

        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name="Cox PH", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
                "log_likelihood": result.log_likelihood,
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)
        return output
    except ImportError as e:
        return str(e)
    except Exception as e:
        return friendly_error(e, "stcox")


@command("sts", usage="sts graph [by=group] | sts test <group_var>")
def cmd_sts(session: Session, args: str) -> str:
    """Kaplan-Meier survival curves and log-rank test."""
    df = session.require_data()

    if session._surv_time_var is None or session._surv_event_var is None:
        return "Survival structure not set. Use: stset <time_var>, failure(<event_var>)"

    ca = CommandArgs(args)
    subcmd = ca.positional[0].lower() if ca.positional else ""

    if subcmd == "graph":
        group_var = ca.get_option("by")
        try:
            from openstat.stats.survival import kaplan_meier
            summary, kmf = kaplan_meier(
                df, session._surv_time_var, session._surv_event_var, group_var,
            )

            # Plot
            try:
                from openstat.plots.surv_plots import plot_km
                path = plot_km(kmf, session.output_dir, group_var)
                session.plot_paths.append(str(path))
                summary += f"\nPlot saved: {path}"
            except Exception:
                pass

            return summary
        except ImportError as e:
            return str(e)
        except Exception as e:
            return friendly_error(e, "sts graph")

    elif subcmd == "test":
        group_var = ca.positional[1] if len(ca.positional) > 1 else None
        if not group_var:
            return "Usage: sts test <group_var>"
        if group_var not in df.columns:
            return f"Column not found: {group_var}"
        try:
            from openstat.stats.survival import log_rank_test
            return log_rank_test(
                df, session._surv_time_var, session._surv_event_var, group_var,
            )
        except ImportError as e:
            return str(e)
        except Exception as e:
            return friendly_error(e, "sts test")

    else:
        return "Usage: sts graph [by=group] | sts test <group_var>"
