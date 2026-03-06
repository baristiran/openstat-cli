"""Bayesian regression commands."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("bayes", usage="bayes: ols depvar indepvars [, samples(4000) priorscale(10) ci(0.95)]")
def cmd_bayes(session: Session, args: str) -> str:
    """Bayesian OLS with conjugate Normal-Inverse-Gamma prior (no MCMC required)."""
    df = session.require_data()

    # strip "ols" or ": ols" prefix
    clean = re.sub(r'^\s*:?\s*ols\s+', '', args, flags=re.IGNORECASE)
    positional, opts = _stata_opts(clean)

    dep = positional[0] if positional else ""
    indeps = [c for c in positional[1:] if c in df.columns]

    if not dep or not indeps:
        return "Usage: bayes: ols depvar indepvar1 indepvar2 ... [, samples(4000)]"

    n_samples = int(opts.get("samples", 4000))
    prior_scale = float(opts.get("priorscale", 10.0))
    ci = float(opts.get("ci", 0.95))

    try:
        from openstat.stats.bayesian import bayes_ols
        result = bayes_ols(
            df, dep, indeps,
            n_samples=n_samples,
            prior_scale=prior_scale,
            credible_interval=ci,
        )
    except Exception as exc:
        return f"bayes error: {exc}"

    ci_pct = int(ci * 100)
    lines = [f"\n{result['model']}", "=" * 70]
    lines.append(f"  Dependent: {dep}   N = {result['n_obs']}   "
                 f"Draws = {n_samples}   R² ≈ {result['r_squared']:.4f}")
    lines.append(f"  σ̂ = {result['sigma_mean']:.4f} (±{result['sigma_std']:.4f})")
    lines.append("")
    lines.append(
        f"  {'Variable':<20}  {'Post. Mean':>12}  {'Post. SD':>10}  "
        f"{'CI Lo ({ci_pct}%)':>12}  {'CI Hi':>12}  {'P(β>0)':>8}"
    )
    lines.append("  " + "-" * 66)
    for name, stats in result["coefficients"].items():
        lo_key = f"ci_{ci_pct}_lo"
        hi_key = f"ci_{ci_pct}_hi"
        lines.append(
            f"  {name:<20}  {stats['mean']:>12.4f}  {stats['std']:>10.4f}  "
            f"  {stats[lo_key]:>12.4f}  {stats[hi_key]:>12.4f}  {stats['prob_positive']:>8.4f}"
        )
    lines.append("=" * 70)
    lines.append(f"  Prior: Normal(0, {prior_scale}²) on coefficients  |  IG(0.001, 0.001) on σ²")
    session._last_model = result
    return "\n".join(lines)
