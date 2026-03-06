"""Automatic model selection: automodel command."""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import polars as pl

from openstat.commands.base import command
from openstat.session import Session
from openstat.dsl.parser import parse_formula, ParseError


class _Candidate(NamedTuple):
    formula: str
    model_type: str
    aic: float
    bic: float
    r2: float | None
    n: int
    k: int  # number of predictors


def _fit_candidate(
    df: pl.DataFrame,
    dep: str,
    indeps: list[str],
    model_type: str,
) -> _Candidate | None:
    """Fit a single candidate model, return metrics or None on failure."""
    try:
        if model_type == "ols":
            from openstat.stats.models import fit_ols
            result, _ = fit_ols(df, dep, indeps)
            formula = f"{dep} ~ {' + '.join(indeps)}"
            return _Candidate(
                formula=formula, model_type="OLS",
                aic=result.aic or float("inf"), bic=result.bic or float("inf"),
                r2=result.r_squared, n=result.n_obs, k=len(indeps),
            )
        elif model_type == "logit":
            from openstat.stats.models import fit_logit
            result, _ = fit_logit(df, dep, indeps)
            formula = f"{dep} ~ {' + '.join(indeps)}"
            return _Candidate(
                formula=formula, model_type="Logit",
                aic=result.aic or float("inf"), bic=result.bic or float("inf"),
                r2=result.pseudo_r2, n=result.n_obs, k=len(indeps),
            )
        elif model_type == "poisson":
            from openstat.stats.models import fit_poisson
            result, _ = fit_poisson(df, dep, indeps)
            formula = f"{dep} ~ {' + '.join(indeps)}"
            return _Candidate(
                formula=formula, model_type="Poisson",
                aic=result.aic or float("inf"), bic=result.bic or float("inf"),
                r2=result.pseudo_r2, n=result.n_obs, k=len(indeps),
            )
    except Exception:
        return None
    return None


@command("automodel", usage="automodel <depvar> ~ <x1> <x2> ... [--ols|--logit|--poisson] [--criterion=aic|bic] [--maxvars=N]")
def cmd_automodel(session: Session, args: str) -> str:
    """Automatic model selection: fits all variable subsets and ranks by AIC/BIC.

    Uses exhaustive search for ≤ 8 predictors, forward stepwise for more.

    Examples:
      automodel score ~ age income education
      automodel employed ~ age income score region --logit --criterion=bic
      automodel score ~ age income education region --criterion=aic --maxvars=3
    """
    import re
    df = session.require_data()

    # Parse flags
    use_logit = "--logit" in args
    use_poisson = "--poisson" in args
    model_type = "logit" if use_logit else "poisson" if use_poisson else "ols"

    m_crit = re.search(r"--criterion[= ](\w+)", args)
    criterion = m_crit.group(1).lower() if m_crit else "aic"
    if criterion not in ("aic", "bic"):
        criterion = "aic"

    m_max = re.search(r"--maxvars[= ](\d+)", args)
    max_vars = int(m_max.group(1)) if m_max else None

    # Clean flags from formula
    formula_str = re.sub(r"--\w+(?:[= ]\w+)?", "", args).strip()
    if "~" not in formula_str:
        return (
            "Usage: automodel <depvar> ~ <x1> <x2> ... [--ols|--logit|--poisson]\n"
            "Example: automodel score ~ age income education"
        )

    # Normalize: allow space-separated predictors (convert to + separated)
    if "~" in formula_str:
        lhs, rhs = formula_str.split("~", 1)
        # If no + in rhs, convert spaces to +
        if "+" not in rhs:
            rhs = " + ".join(rhs.split())
        formula_str = f"{lhs.strip()} ~ {rhs.strip()}"

    try:
        dep, indeps = parse_formula(formula_str)
    except ParseError as e:
        return f"Formula error: {e}"

    if dep not in df.columns:
        return f"Dependent variable not found: {dep}"
    missing = [x for x in indeps if x not in df.columns]
    if missing:
        return f"Predictors not found: {', '.join(missing)}"

    if max_vars:
        indeps = indeps[:max_vars + 10]  # allow some buffer

    k = len(indeps)
    strategy = "exhaustive" if k <= 8 else "forward stepwise"

    # Build candidates
    candidates: list[_Candidate] = []

    if strategy == "exhaustive":
        total = 2 ** k - 1  # exclude empty model
        for r in range(1, k + 1):
            if max_vars and r > max_vars:
                break
            for subset in itertools.combinations(indeps, r):
                c = _fit_candidate(df, dep, list(subset), model_type)
                if c:
                    candidates.append(c)
    else:
        # Forward stepwise
        current = []
        remaining = list(indeps)
        while remaining and (max_vars is None or len(current) < max_vars):
            best: _Candidate | None = None
            for var in remaining:
                trial = current + [var]
                c = _fit_candidate(df, dep, trial, model_type)
                if c:
                    if best is None or getattr(c, criterion) < getattr(best, criterion):
                        best = c
            if best is None:
                break
            # Find which var was added
            best_vars = best.formula.split("~")[1].strip().split(" + ")
            added = [v for v in best_vars if v not in current]
            current.extend(added)
            remaining = [v for v in remaining if v not in current]
            candidates.append(best)

    if not candidates:
        return "No valid models found. Check your data and variable names."

    # Sort by criterion
    candidates.sort(key=lambda c: getattr(c, criterion))
    top_n = min(10, len(candidates))
    top = candidates[:top_n]

    # Store best model result in session
    best = candidates[0]
    try:
        dep2, indeps2 = parse_formula(best.formula)
        if model_type == "ols":
            from openstat.stats.models import fit_ols
            result, raw = fit_ols(df, dep2, indeps2)
        elif model_type == "logit":
            from openstat.stats.models import fit_logit
            result, raw = fit_logit(df, dep2, indeps2)
        else:
            from openstat.stats.models import fit_poisson
            result, raw = fit_poisson(df, dep2, indeps2)
        session._last_model = raw
        session._last_model_vars = (dep2, indeps2)
        session._last_fit_result = result
        session._last_fit_kwargs = {}
    except Exception:
        pass

    crit_label = criterion.upper()
    lines = [
        f"Dependent: {dep}   Candidates: {len(candidates)}   Strategy: {strategy}",
        f"Model type: {model_type.upper()}   Selection criterion: {crit_label}",
        "",
        f"Top {top_n} models by {crit_label}:",
        f"  {'#':<3} {'AIC':>9}  {'BIC':>9}  {'R²/PseudoR²':>11}  k  Formula",
        "  " + "-" * 76,
    ]
    for i, c in enumerate(top, 1):
        r2_str = f"{c.r2:.4f}" if c.r2 is not None else "      —"
        marker = " ← best" if i == 1 else ""
        lines.append(
            f"  {i:<3} {c.aic:>9.2f}  {c.bic:>9.2f}  "
            f"{r2_str:>11}  {c.k}  {c.formula}{marker}"
        )

    lines += [
        "",
        f"Best model: {best.formula}",
        f"  AIC = {best.aic:.2f}   BIC = {best.bic:.2f}",
        "",
        "Best model loaded. Use 'estimates', 'vif', 'residuals', 'plot coef' for diagnostics.",
    ]

    return "\n" + "=" * 60 + "\nAutomatic Model Selection\n" + "=" * 60 + "\n" + "\n".join(lines) + "\n" + "=" * 60
