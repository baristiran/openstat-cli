"""Advanced regression commands: nls, betareg, zip, zinb, hurdle, sureg."""

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


def _parse_eq(args: str, df_cols: list[str]) -> tuple[str, list[str], dict]:
    positional, opts = _stata_opts(args)
    dep = positional[0] if positional else ""
    indeps = [c for c in positional[1:] if c in df_cols]
    return dep, indeps, opts


def _coef_table(params: dict, se: dict = None, pvals: dict = None) -> str:
    lines = [f"  {'Variable':<22}  {'Coef.':>10}"]
    if se:
        lines[0] += f"  {'Std.Err.':>10}"
    if pvals:
        lines[0] += f"  {'p-value':>8}"
    lines.append("  " + "-" * 55)
    for k, v in params.items():
        row = f"  {k:<22}  {v:>10.4f}"
        if se and k in se:
            row += f"  {se[k]:>10.4f}"
        if pvals and k in pvals:
            row += f"  {pvals[k]:>8.4f}"
        lines.append(row)
    return "\n".join(lines)


@command("nls", usage="nls depvar indepvars [, p0(a,b,c) fn(power|exp|log)]")
def cmd_nls(session: Session, args: str) -> str:
    """Nonlinear least squares with built-in functional forms."""
    df = session.require_data()
    dep, indeps, opts = _parse_eq(args, df.columns)
    if not dep or not indeps:
        return "Usage: nls depvar indepvar1 ... [, fn(power) p0(1,1)]"

    fn_name = opts.get("fn", "power")
    p0_raw = opts.get("p0", "1,1")
    try:
        p0 = [float(x) for x in p0_raw.split(",")]
    except ValueError:
        p0 = [1.0, 1.0]

    built_in = {
        "power": lambda X, a, b: a * X[:, 0] ** b,
        "exp": lambda X, a, b: a * np.exp(b * X[:, 0]),
        "log": lambda X, a, b: a + b * np.log(np.maximum(X[:, 0], 1e-15)),
    }

    import numpy as np
    fn = built_in.get(fn_name)
    if fn is None:
        return f"Unknown function '{fn_name}'. Use: power, exp, log"

    # Adjust p0 to match function arity
    import inspect
    n_params = len(inspect.signature(fn).parameters) - 1
    p0 = (p0 + [1.0] * n_params)[:n_params]

    try:
        from openstat.stats.advanced_regression import fit_nls
        result = fit_nls(df, dep, indeps, fn, p0)
        session._last_model = result
        lines = [f"\nNLS ({fn_name}): {dep}", "=" * 50]
        lines.append(f"  {'N obs':<22}  {result['n_obs']:>10}")
        lines.append(f"  {'R²':<22}  {result['r_squared']:>10.4f}")
        lines.append(f"  {'Converged':<22}  {result['converged']!s:>10}")
        lines.append("\nParameters:")
        lines.append(_coef_table(result["params"], result["std_errors"]))
        lines.append("=" * 50)
        return "\n".join(lines)
    except Exception as exc:
        return f"nls error: {exc}"


@command("betareg", usage="betareg depvar indepvars")
def cmd_betareg(session: Session, args: str) -> str:
    """Beta regression for (0,1) bounded outcomes."""
    df = session.require_data()
    dep, indeps, opts = _parse_eq(args, df.columns)
    if not dep or not indeps:
        return "Usage: betareg depvar indepvar1 ..."
    try:
        from openstat.stats.advanced_regression import fit_betareg
        result = fit_betareg(df, dep, indeps)
        session._last_model = result
        lines = [f"\n{result['method']}: {dep}", "=" * 55]
        lines.append(f"  N = {result['n_obs']}   AIC = {result['aic']:.4f}   Pseudo-R² = {result['pseudo_r2']:.4f}")
        lines.append("\nCoefficients:")
        lines.append(_coef_table(result["params"], result["std_errors"], result["p_values"]))
        lines.append("=" * 55)
        return "\n".join(lines)
    except Exception as exc:
        return f"betareg error: {exc}"


@command("zip", usage="zip depvar indepvars")
def cmd_zip(session: Session, args: str) -> str:
    """Zero-Inflated Poisson regression."""
    df = session.require_data()
    dep, indeps, opts = _parse_eq(args, df.columns)
    if not dep or not indeps:
        return "Usage: zip depvar indepvar1 ..."
    try:
        from openstat.stats.advanced_regression import fit_zip
        result = fit_zip(df, dep, indeps)
        session._last_model = result
        lines = [f"\n{result['method']}: {dep}", "=" * 55]
        lines.append(f"  N = {result['n_obs']}   AIC = {result['aic']:.4f}   LL = {result['log_likelihood']:.4f}")
        lines.append("\nCoefficients:")
        lines.append(_coef_table(result["params"], result["std_errors"], result["p_values"]))
        lines.append("=" * 55)
        return "\n".join(lines)
    except Exception as exc:
        return f"zip error: {exc}"


@command("zinb", usage="zinb depvar indepvars")
def cmd_zinb(session: Session, args: str) -> str:
    """Zero-Inflated Negative Binomial regression."""
    df = session.require_data()
    dep, indeps, opts = _parse_eq(args, df.columns)
    if not dep or not indeps:
        return "Usage: zinb depvar indepvar1 ..."
    try:
        from openstat.stats.advanced_regression import fit_zinb
        result = fit_zinb(df, dep, indeps)
        session._last_model = result
        lines = [f"\n{result['method']}: {dep}", "=" * 55]
        lines.append(f"  N = {result['n_obs']}   AIC = {result['aic']:.4f}   LL = {result['log_likelihood']:.4f}")
        lines.append("\nCoefficients:")
        lines.append(_coef_table(result["params"], result["std_errors"], result["p_values"]))
        lines.append("=" * 55)
        return "\n".join(lines)
    except Exception as exc:
        return f"zinb error: {exc}"


@command("hurdle", usage="hurdle depvar indepvars")
def cmd_hurdle(session: Session, args: str) -> str:
    """Hurdle model: Logit for zeros + Poisson for positives."""
    df = session.require_data()
    dep, indeps, opts = _parse_eq(args, df.columns)
    if not dep or not indeps:
        return "Usage: hurdle depvar indepvar1 ..."
    try:
        from openstat.stats.advanced_regression import fit_hurdle
        result = fit_hurdle(df, dep, indeps)
        session._last_model = result
        lines = [f"\nHurdle Model: {dep}", "=" * 55]
        lines.append(f"  N = {result['n_obs']}   Zeros = {result['n_zeros']}   Positive = {result['n_positive']}")
        lines.append("\nPart 1 — Logit (P(y > 0)):")
        lines.append(_coef_table(result["logit_params"], pvals=result["logit_pvalues"]))
        lines.append("\nPart 2 — Poisson (E[y | y > 0]):")
        lines.append(_coef_table(result["count_params"], pvals=result["count_pvalues"]))
        lines.append("=" * 55)
        return "\n".join(lines)
    except Exception as exc:
        return f"hurdle error: {exc}"


@command("sureg", usage="sureg (dep1 x1 x2) (dep2 x3 x4)")
def cmd_sureg(session: Session, args: str) -> str:
    """Seemingly Unrelated Regression."""
    df = session.require_data()
    # Parse parenthesized equations: (dep x1 x2) (dep2 x3 x4)
    eq_matches = re.findall(r'\(([^)]+)\)', args)
    if not eq_matches:
        return "Usage: sureg (dep1 x1 x2) (dep2 x3 x4)"

    equations = []
    for eq_str in eq_matches:
        parts = eq_str.split()
        if len(parts) < 2:
            continue
        dep = parts[0]
        indeps = [c for c in parts[1:] if c in df.columns]
        if dep in df.columns and indeps:
            equations.append((dep, indeps))

    if not equations:
        return "No valid equations found."
    try:
        from openstat.stats.advanced_regression import fit_sur
        result = fit_sur(df, equations)
        session._last_model = result
        lines = [f"\nSUR: {result['n_equations']} equations", "=" * 60]
        for eq in result["equations"]:
            lines.append(f"\nEquation {eq['equation']}: {eq['dep']}   R² = {eq['r_squared']:.4f}   N = {eq['n_obs']}")
            lines.append(_coef_table(eq["params"], eq["std_errors"]))
        lines.append("=" * 60)
        return "\n".join(lines)
    except Exception as exc:
        return f"sureg error: {exc}"
