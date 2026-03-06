"""Instrumental variable commands: ivregress."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.dsl.parser import parse_formula, ParseError


def _parse_iv_formula(raw: str) -> tuple[str, list[str], list[str], list[str]]:
    """Parse IV formula: y ~ x1 (x_endog = z1 z2).

    Returns (dep, exog_vars, endog_vars, instruments).
    """
    m = re.search(r'\((.+?)\)', raw)
    if not m:
        raise ParseError("IV formula requires parenthesized instruments: (endogenous = instruments)")

    inner = m.group(1)
    if '=' not in inner:
        raise ParseError("Instrument block must use '=': (x_endog = z1 z2)")

    endog_part, instr_part = inner.split('=', 1)
    endog_vars = endog_part.split()
    instruments = instr_part.split()

    if not endog_vars:
        raise ParseError("No endogenous variables specified")
    if not instruments:
        raise ParseError("No instruments specified")

    # Remove parenthetical from args, parse remaining as formula
    clean = raw[:m.start()] + raw[m.end():]
    clean = clean.strip()
    if '~' in clean:
        dep, exog_vars = parse_formula(clean)
    else:
        parts = clean.split()
        if not parts:
            raise ParseError("No dependent variable specified")
        dep = parts[0]
        exog_vars = parts[1:] if len(parts) > 1 else []

    return dep, exog_vars, endog_vars, instruments


@command("ivregress", usage="ivregress 2sls y ~ x1 (x_endog = z1 z2) [--robust]")
def cmd_ivregress(session: Session, args: str) -> str:
    """Fit instrumental variable regression via Two-Stage Least Squares."""
    df = session.require_data()
    ca = CommandArgs(args)

    # First positional should be method (2sls)
    if not ca.positional or ca.positional[0].lower() != "2sls":
        return "Usage: ivregress 2sls y ~ x1 (x_endog = z1 z2) [--robust]"

    robust = ca.has_flag("--robust")
    formula_str = ca.strip_flags_and_options()
    # Remove "2sls" prefix
    formula_str = re.sub(r'^\s*2sls\s+', '', formula_str, flags=re.IGNORECASE).strip()

    try:
        dep, exog, endog, instruments = _parse_iv_formula(formula_str)
    except ParseError as e:
        return f"Formula error: {e}"

    try:
        from openstat.stats.iv import fit_iv_2sls

        result, raw = fit_iv_2sls(df, dep, exog, endog, instruments, robust=robust)

        # Store in session
        session._last_model = raw
        session._last_model_vars = (dep, exog + endog)
        session._last_fit_result = result
        session._last_fit_kwargs = {"method": "2sls", "endog": endog, "instruments": instruments}

        all_vars = result.indep_vars
        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name="IV-2SLS", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
                "r_squared": result.r_squared,
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)

        # Auto-show first-stage diagnostics
        try:
            from openstat.stats.iv import first_stage_diagnostics
            output += "\n\n" + first_stage_diagnostics(raw)
        except Exception:
            pass

        return output
    except ImportError as e:
        return str(e)
    except Exception as e:
        return friendly_error(e, "ivregress")
