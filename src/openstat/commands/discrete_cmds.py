"""Discrete / censored model commands: tobit, mlogit, ologit, oprobit."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.dsl.parser import parse_formula, ParseError
from openstat.stats.discrete import fit_tobit, fit_mlogit, fit_ordered
from openstat.commands.base import command, CommandArgs, friendly_error


def _store_model(session, result, raw_model, dep, indeps):
    """Store model in session state, return summary output."""
    session._last_model = raw_model
    session._last_model_vars = (dep, indeps)
    session._last_fit_result = result
    session._last_fit_kwargs = {}
    md = result.to_markdown()
    details: dict = {
        "n_obs": result.n_obs,
        "params": dict(result.params),
        "std_errors": dict(result.std_errors),
    }
    if result.aic is not None:
        details["aic"] = result.aic
    if result.bic is not None:
        details["bic"] = result.bic
    if result.pseudo_r2 is not None:
        details["pseudo_r2"] = result.pseudo_r2
    if result.log_likelihood is not None:
        details["log_likelihood"] = result.log_likelihood
    session.results.append(ModelResult(
        name=result.model_type, formula=result.formula,
        table=md, details=details,
    ))
    output = result.summary_table()
    if result.warnings:
        output += "\n" + "\n".join(result.warnings)
    return output


@command("tobit", usage="tobit y ~ x1 + x2 [, ll(0)] [, ul(100)] [--robust]")
def cmd_tobit(session: Session, args: str) -> str:
    """Fit a Tobit (censored) regression model."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")

    # Parse limits from args: ll(value) and ul(value)
    lower_limit = None
    upper_limit = None
    ll_match = re.search(r'(?:,\s*)?ll\(([^)]+)\)', args)
    ul_match = re.search(r'(?:,\s*)?ul\(([^)]+)\)', args)
    if ll_match:
        try:
            lower_limit = float(ll_match.group(1))
        except ValueError:
            return f"Invalid lower limit: {ll_match.group(1)}"
    if ul_match:
        try:
            upper_limit = float(ul_match.group(1))
        except ValueError:
            return f"Invalid upper limit: {ul_match.group(1)}"

    # Strip limit specs from formula
    formula_str = args
    for pattern in [r',?\s*ll\([^)]+\)', r',?\s*ul\([^)]+\)']:
        formula_str = re.sub(pattern, '', formula_str)
    formula_str = CommandArgs(formula_str).strip_flags_and_options()

    if not formula_str or "~" not in formula_str:
        return "Usage: tobit y ~ x1 + x2 [, ll(0)] [, ul(100)] [--robust]"

    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_tobit(
            df, dep, indeps,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            robust=robust,
            cluster_col=cluster_col,
        )
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Tobit error")


@command("mlogit", usage="mlogit y ~ x1 + x2 [--robust] [--cluster=col]")
def cmd_mlogit(session: Session, args: str) -> str:
    """Fit a Multinomial Logit model."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: mlogit y ~ x1 + x2 [--robust] [--cluster=col]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_mlogit(df, dep, indeps, robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "MNLogit error")


@command("ologit", usage="ologit y ~ x1 + x2 [--robust]")
def cmd_ologit(session: Session, args: str) -> str:
    """Fit an Ordered Logit model."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: ologit y ~ x1 + x2 [--robust]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_ordered(df, dep, indeps, link="logit", robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Ordered Logit error")


@command("oprobit", usage="oprobit y ~ x1 + x2 [--robust]")
def cmd_oprobit(session: Session, args: str) -> str:
    """Fit an Ordered Probit model."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()
    if not formula_str:
        return "Usage: oprobit y ~ x1 + x2 [--robust]"
    try:
        dep, indeps = parse_formula(formula_str)
        result, raw_model = fit_ordered(df, dep, indeps, link="probit", robust=robust, cluster_col=cluster_col)
        return _store_model(session, result, raw_model, dep, indeps)
    except ParseError as e:
        return f"Formula error: {e}"
    except Exception as e:
        return friendly_error(e, "Ordered Probit error")
