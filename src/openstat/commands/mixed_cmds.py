"""Mixed / hierarchical model commands: mixed, estat icc."""

from __future__ import annotations

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.dsl.parser import parse_formula, ParseError


def _parse_mixed_formula(raw: str) -> tuple[str, list[str], str, list[str]]:
    """Parse mixed model formula: y ~ x1 || group: x1.

    Returns (dep, fixed_effects, group_var, random_effects).
    """
    if '||' not in raw:
        raise ParseError("Mixed model requires '||' to specify grouping: y ~ x1 || group: [re_vars]")

    fixed_part, random_part = raw.split('||', 1)
    dep, fixed = parse_formula(fixed_part.strip())

    random_part = random_part.strip()
    if ':' in random_part:
        group_str, re_str = random_part.split(':', 1)
        group_var = group_str.strip()
        re_vars = re_str.strip().split() if re_str.strip() else []
    else:
        group_var = random_part.strip()
        re_vars = []

    if not group_var:
        raise ParseError("No grouping variable specified after '||'")

    return dep, fixed, group_var, re_vars


@command("mixed", usage="mixed y ~ x1 || group: [re_vars]")
def cmd_mixed(session: Session, args: str) -> str:
    """Fit a mixed/hierarchical linear model with random intercepts and/or slopes."""
    df = session.require_data()

    try:
        dep, fixed, group_var, re_vars = _parse_mixed_formula(args.strip())
    except ParseError as e:
        return f"Formula error: {e}"

    # Validate columns
    all_cols = [dep] + fixed + [group_var] + re_vars
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        from openstat.stats.mixed import fit_mixed

        result, raw = fit_mixed(df, dep, fixed, group_var, re_vars or None)

        session._last_model = raw
        session._last_model_vars = (dep, fixed)
        session._last_fit_result = result
        session._last_fit_kwargs = {"group_var": group_var, "re_vars": re_vars}

        md = result.to_markdown() if hasattr(result, "to_markdown") else ""
        session.results.append(ModelResult(
            name="Mixed LM", formula=result.formula,
            table=md, details={
                "n_obs": result.n_obs,
                "params": dict(result.params),
                "aic": result.aic,
                "bic": result.bic,
                "log_likelihood": result.log_likelihood,
            },
        ))

        output = result.summary_table()
        if result.warnings:
            output += "\n" + "\n".join(result.warnings)
        return output
    except Exception as e:
        return friendly_error(e, "mixed")
