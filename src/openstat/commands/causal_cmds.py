"""Causal inference commands: did, psmatch."""

from __future__ import annotations

import re

from openstat.session import Session, ModelResult
from openstat.dsl.parser import ParseError
from openstat.stats.causal import fit_did, fit_psm
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


@command("did", usage="did y ~ treatment_var time_var [--robust] [--cluster=col]")
def cmd_did(session: Session, args: str) -> str:
    """Difference-in-Differences estimation."""
    df = session.require_data()
    ca = CommandArgs(args)
    robust = ca.has_flag("--robust")
    cluster_col = ca.get_option("cluster")
    formula_str = ca.strip_flags_and_options()

    if not formula_str or "~" not in formula_str:
        return "Usage: did y ~ treatment_var time_var [--robust] [--cluster=col]"

    try:
        # Parse: y ~ treatment_var time_var
        left, right = formula_str.split("~", 1)
        dep = left.strip()
        if not dep:
            return "Usage: did y ~ treatment_var time_var"

        rhs_vars = right.strip().split()
        if len(rhs_vars) < 2:
            return "Usage: did y ~ treatment_var time_var (need both treatment and time variables)"

        treatment_col_name = rhs_vars[0]
        time_col_name = rhs_vars[1]

        result, raw_model = fit_did(
            df, dep, treatment_col_name, time_col_name,
            robust=robust, cluster_col=cluster_col,
        )
        return _store_model(session, result, raw_model, dep, [treatment_col_name, time_col_name])
    except Exception as e:
        return friendly_error(e, "DiD error")


@command("psmatch", usage="psmatch outcome ~ covars, treatment(tvar) [caliper(0.1)] [nn(3)]")
def cmd_psmatch(session: Session, args: str) -> str:
    """Propensity Score Matching."""
    df = session.require_data()

    # Parse treatment(var)
    treat_match = re.search(r'treatment\((\w+)\)', args)
    if not treat_match:
        return "Usage: psmatch outcome ~ x1 x2, treatment(tvar) [caliper(0.1)] [nn(3)]"
    treatment_col = treat_match.group(1)

    # Parse optional caliper(value)
    caliper = None
    caliper_match = re.search(r'caliper\(([^)]+)\)', args)
    if caliper_match:
        try:
            caliper = float(caliper_match.group(1))
        except ValueError:
            return f"Invalid caliper value: {caliper_match.group(1)}"

    # Parse optional nn(value)
    n_neighbors = 1
    nn_match = re.search(r'nn\((\d+)\)', args)
    if nn_match:
        n_neighbors = int(nn_match.group(1))

    # Strip options to get formula part
    formula_part = args
    for pattern in [r',?\s*treatment\([^)]+\)', r',?\s*caliper\([^)]+\)', r',?\s*nn\(\d+\)']:
        formula_part = re.sub(pattern, '', formula_part)
    formula_part = formula_part.strip().rstrip(',').strip()

    if not formula_part or "~" not in formula_part:
        return "Usage: psmatch outcome ~ x1 x2, treatment(tvar) [caliper(0.1)] [nn(3)]"

    try:
        left, right = formula_part.split("~", 1)
        outcome = left.strip()
        covariates = right.strip().split()

        if not outcome or not covariates:
            return "Usage: psmatch outcome ~ x1 x2, treatment(tvar)"

        result_str = fit_psm(
            df, outcome, covariates, treatment_col,
            n_neighbors=n_neighbors, caliper=caliper,
        )

        # Store a simple record
        session.results.append(ModelResult(
            name="PSM", formula=f"{outcome} ~ {' + '.join(covariates)}",
            table=result_str, details={"treatment": treatment_col},
        ))
        return result_str
    except Exception as e:
        return friendly_error(e, "PSM error")


@command("iptw", usage="iptw <outcome> ~ <covars>, treatment(<tvar>) [--ate|--att] [--trim=0.01]")
def cmd_iptw(session: Session, args: str) -> str:
    """Inverse Probability Treatment Weighting (IPTW) for causal inference.

    Estimates propensity scores via logistic regression, then uses IPT weights
    to estimate the Average Treatment Effect (ATE) or Average Treatment Effect
    on the Treated (ATT) via weighted OLS.

    Examples:
      iptw score ~ age + income, treatment(employed)
      iptw score ~ age + income, treatment(employed) --att
      iptw score ~ age + income, treatment(employed) --ate --trim=0.05
    """
    import re
    import numpy as np
    import polars as pl
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression

    df = session.require_data()

    # Parse treatment(var)
    treat_m = re.search(r'treatment\((\w+)\)', args)
    if not treat_m:
        return "Usage: iptw outcome ~ x1 + x2, treatment(tvar) [--ate|--att] [--trim=0.01]"
    treatment_col = treat_m.group(1)

    # Parse estimand
    att = "--att" in args
    estimand = "ATT" if att else "ATE"

    # Parse trim
    trim = 0.01
    trim_m = re.search(r'--trim[= ]([\d.]+)', args)
    if trim_m:
        trim = float(trim_m.group(1))

    # Clean formula
    formula_part = re.sub(r',?\s*treatment\([^)]+\)', '', args)
    formula_part = re.sub(r'--\w+(?:[= ][\d.]+)?', '', formula_part).strip()

    if "~" not in formula_part:
        return "Usage: iptw outcome ~ x1 + x2, treatment(tvar)"

    lhs, rhs = formula_part.split("~", 1)
    outcome = lhs.strip()
    covars = [c.strip() for c in rhs.replace("+", " ").split() if c.strip()]

    needed = [outcome, treatment_col] + covars
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    sub = df.select(needed).drop_nulls()
    y = sub[outcome].to_numpy().astype(float)
    treat = sub[treatment_col].to_numpy().astype(float)
    X = sub.select(covars).to_numpy().astype(float)

    # Step 1: Propensity score model
    try:
        ps_model = LogisticRegression(max_iter=1000, C=1e6)
        ps_model.fit(X, treat)
        ps = ps_model.predict_proba(X)[:, 1]
    except Exception as exc:
        return f"Propensity score estimation failed: {exc}"

    # Trim extreme propensity scores
    ps_clipped = np.clip(ps, trim, 1 - trim)

    # Step 2: Compute IPTW weights
    if estimand == "ATE":
        weights = treat / ps_clipped + (1 - treat) / (1 - ps_clipped)
    else:  # ATT
        weights = treat + (1 - treat) * ps_clipped / (1 - ps_clipped)

    # Step 3: Weighted OLS of outcome on treatment
    X_ols = sm.add_constant(treat)
    try:
        wls_res = sm.WLS(y, X_ols, weights=weights).fit()
    except Exception as exc:
        return f"Weighted regression failed: {exc}"

    ate_est = wls_res.params[1]
    ate_se = wls_res.bse[1]
    ate_t = wls_res.tvalues[1]
    ate_p = wls_res.pvalues[1]
    ci_low, ci_high = wls_res.conf_int()[1]

    def _sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    # Balance assessment (standardized mean differences before/after)
    n_treat = int(treat.sum())
    n_ctrl = int((1 - treat).sum())

    lines = [
        f"Outcome: {outcome}   Treatment: {treatment_col}   Estimand: {estimand}",
        f"N = {len(y)}  (treated={n_treat}, control={n_ctrl})",
        f"Propensity score trim: [{trim:.3f}, {1-trim:.3f}]",
        "",
        f"IPTW {estimand} Estimate:",
        f"  {'Coef':>10}  {'SE':>8}  {'t':>7}  {'p-value':>9}  {'95% CI':>20}",
        "  " + "-" * 58,
        f"  {ate_est:>10.4f}  {ate_se:>8.4f}  {ate_t:>7.3f}  "
        f"{ate_p:>9.4f}{_sig(ate_p)}  [{ci_low:.4f}, {ci_high:.4f}]",
        "",
        "Weight Summary:",
        f"  Min={weights.min():.3f}  Mean={weights.mean():.3f}  "
        f"Max={weights.max():.3f}  SD={weights.std():.3f}",
        "",
        "Propensity Score Summary:",
        f"  Treated  — mean={ps[treat==1].mean():.3f}  "
        f"min={ps[treat==1].min():.3f}  max={ps[treat==1].max():.3f}",
        f"  Control  — mean={ps[treat==0].mean():.3f}  "
        f"min={ps[treat==0].min():.3f}  max={ps[treat==0].max():.3f}",
    ]

    # Standardized mean differences (covariate balance)
    lines += ["", "Covariate Balance (Standardized Mean Differences):"]
    lines.append(f"  {'Variable':<20} {'Before SMD':>12} {'After SMD':>11}  {'Balanced?':>10}")
    lines.append("  " + "-" * 56)
    for j, cname in enumerate(covars):
        xj = X[:, j]
        mu_t = xj[treat == 1].mean()
        mu_c = xj[treat == 0].mean()
        sd_pool = np.sqrt((xj[treat == 1].var() + xj[treat == 0].var()) / 2 + 1e-10)
        smd_before = abs(mu_t - mu_c) / sd_pool
        # After weighting
        mu_t_w = np.average(xj[treat == 1], weights=weights[treat == 1])
        mu_c_w = np.average(xj[treat == 0], weights=weights[treat == 0])
        smd_after = abs(mu_t_w - mu_c_w) / sd_pool
        balanced = "✓" if smd_after < 0.1 else "✗"
        lines.append(
            f"  {cname:<20} {smd_before:>12.4f} {smd_after:>11.4f}  {balanced:>10}"
        )

    return "\n" + "=" * 60 + "\nIPTW Causal Estimate\n" + "=" * 60 + "\n" + "\n".join(lines) + "\n" + "=" * 60
