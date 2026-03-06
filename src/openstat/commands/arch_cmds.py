"""ARCH/GARCH volatility model commands."""

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


def _arch_table(result: dict) -> str:
    lines = [f"\n{result['model']}: {result['var']}", "=" * 55]
    lines.append(f"  {'N observations':<25}  {result['n_obs']}")
    lines.append(f"  {'Log-likelihood':<25}  {result['log_likelihood']:.4f}")
    lines.append(f"  {'AIC':<25}  {result['aic']:.4f}")
    lines.append(f"  {'BIC':<25}  {result['bic']:.4f}")
    lines.append("\nParameters:")
    for name, val in result["params"].items():
        lines.append(f"  {name:<25}  {val:>12.6f}")
    if "cond_volatility_last5" in result:
        lines.append("\nConditional volatility (last 5 obs):")
        for v in result["cond_volatility_last5"]:
            lines.append(f"  {v:.4f}")
    lines.append("=" * 55)
    return "\n".join(lines)


@command("arch", usage="arch var [, p(1) dist(normal|t)]")
def cmd_arch(session: Session, args: str) -> str:
    """ARCH(p) model for volatility clustering."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: arch var [, p(1) dist(normal)]"

    var = positional[0]
    p = int(opts.get("p", 1))
    dist = opts.get("dist", "normal")

    try:
        from openstat.stats.arch_garch import fit_arch
        result = fit_arch(df, var, p=p, dist=dist)
        session._last_model = result
        return _arch_table(result)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"arch error: {exc}"


@command("garch", usage="garch var [, p(1) q(1) model(GARCH|EGARCH|GJR-GARCH) dist(normal)]")
def cmd_garch(session: Session, args: str) -> str:
    """GARCH(p,q) or variant volatility model."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: garch var [, p(1) q(1) model(GARCH) dist(normal)]"

    var = positional[0]
    p = int(opts.get("p", 1))
    q = int(opts.get("q", 1))
    dist = opts.get("dist", "normal")
    model = opts.get("model", "GARCH")

    try:
        from openstat.stats.arch_garch import fit_garch
        result = fit_garch(df, var, p=p, q=q, dist=dist, model=model)
        session._last_model = result
        return _arch_table(result)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"garch error: {exc}"
