"""Survey weighting commands: svyset, svy: prefix."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.dsl.parser import parse_formula, ParseError
from openstat.types import NUMERIC_DTYPES


@command("svyset", usage="svyset <psu> [pw=<weight>], strata(<strata>)")
def cmd_svyset(session: Session, args: str) -> str:
    """Declare survey design: PSU, sampling weights, and strata."""
    df = session.require_data()

    # Parse weight: [pw=weight_var]
    m_pw = re.search(r'\[pw=(\w+)\]', args)
    weight_var = m_pw.group(1) if m_pw else None

    # Parse strata: strata(strata_var)
    m_strata = re.search(r'strata\((\w+)\)', args)
    strata_var = m_strata.group(1) if m_strata else None

    # PSU is the first positional argument
    clean = args
    if m_pw:
        clean = clean.replace(m_pw.group(0), "")
    if m_strata:
        clean = clean.replace(m_strata.group(0), "")
    clean = clean.replace(",", "").strip()
    psu_var = clean.split()[0] if clean.split() else None

    # Validate columns exist
    for var, label in [(psu_var, "PSU"), (weight_var, "weight"), (strata_var, "strata")]:
        if var and var not in df.columns:
            return f"{label} column not found: {var}"

    session._svy_psu_var = psu_var
    session._svy_weight_var = weight_var
    session._svy_strata_var = strata_var

    lines = ["Survey design set:"]
    if psu_var:
        lines.append(f"  PSU: {psu_var} ({df[psu_var].n_unique()} clusters)")
    if weight_var:
        lines.append(f"  Weight: {weight_var}")
    if strata_var:
        lines.append(f"  Strata: {strata_var} ({df[strata_var].n_unique()} strata)")
    return "\n".join(lines)


@command("svy:", usage="svy: summarize|ols|logit ...")
def cmd_svy(session: Session, args: str) -> str:
    """Run survey-weighted analysis. Requires svyset first."""
    df = session.require_data()

    if session._svy_weight_var is None:
        return "Survey design not set. Use: svyset <psu> [pw=<weight>], strata(<strata>)"

    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd == "summarize":
        return _svy_summarize(session, df, rest)
    elif subcmd == "ols":
        return _svy_ols(session, df, rest)
    elif subcmd == "logit":
        return _svy_logit(session, df, rest)
    else:
        return "Usage: svy: summarize [cols] | svy: ols y ~ x1 + x2 | svy: logit y ~ x1 + x2"


def _svy_summarize(session: Session, df, args: str) -> str:
    """Weighted summary statistics."""
    cols = args.split() if args.strip() else [c for c in df.columns if df[c].dtype in NUMERIC_DTYPES]
    if not cols:
        return "No numeric columns to summarize."

    from openstat.stats.survey import weighted_summary
    return weighted_summary(df, cols, session._svy_weight_var)


def _svy_ols(session: Session, df, args: str) -> str:
    """Weighted OLS regression."""
    try:
        dep, indeps = parse_formula(args)
    except ParseError as e:
        return f"Formula error: {e}"

    try:
        from openstat.stats.survey import fit_weighted_ols
        result, raw = fit_weighted_ols(
            df, dep, indeps, session._svy_weight_var,
            session._svy_strata_var, session._svy_psu_var,
        )

        session._last_model = raw
        session._last_model_vars = (dep, indeps)
        session._last_fit_result = result
        session._last_fit_kwargs = {"survey": True}

        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name="Svy: OLS", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
                "r_squared": result.r_squared,
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)
        return output
    except Exception as e:
        return friendly_error(e, "svy: ols")


def _svy_logit(session: Session, df, args: str) -> str:
    """Weighted logistic regression."""
    try:
        dep, indeps = parse_formula(args)
    except ParseError as e:
        return f"Formula error: {e}"

    try:
        from openstat.stats.survey import fit_weighted_logit
        result, raw = fit_weighted_logit(df, dep, indeps, session._svy_weight_var)

        session._last_model = raw
        session._last_model_vars = (dep, indeps)
        session._last_fit_result = result
        session._last_fit_kwargs = {"survey": True}

        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name="Svy: Logit", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)
        return output
    except Exception as e:
        return friendly_error(e, "svy: logit")
