"""Power analysis commands: power, sampsi."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session
from openstat.stats.power import (
    power_onemean,
    power_twomeans,
    power_oneproportion,
    power_twoproportions,
    power_ols,
    sampsi as _sampsi,
)


# ── Stata-style argument parser ────────────────────────────────────────────

def _stata_parse(raw: str) -> tuple[list[str], dict[str, str]]:
    """Parse Stata-style args: positional tokens and key(value) options.

    Handles:
      - positional tokens (bare words, numbers)
      - key(value) options  e.g. n(50), alpha(0.05)
      - key=value options   e.g. n=50
      - --flag / bare flag  e.g. --onesided, onesided
      - commas are ignored (Stata separator)
    """
    opts: dict[str, str] = {}
    positional: list[str] = []
    flags: set[str] = set()

    # key(value)
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)

    # Remove key(value) tokens from raw for further parsing
    rest = re.sub(r'\w+\([^)]*\)', '', raw)

    for tok in rest.split():
        tok = tok.strip(',')
        if not tok:
            continue
        if '=' in tok:
            k, v = tok.split('=', 1)
            opts[k.lower().lstrip('-')] = v
        elif tok.startswith('--'):
            flags.add(tok.lstrip('-').lower())
        elif re.match(r'^-?(\d+\.?\d*|\.\d+)$', tok):
            positional.append(tok)
        elif re.match(r'^\w+$', tok):
            # Could be a sub-command, flag, or positional
            positional.append(tok)

    return positional, opts, flags


def _fmt_row(label: str, value) -> str:
    return f"  {label:<30} {value}"


def _power_table(result: dict) -> str:
    lines = [f"\n{result['test']}", "-" * 50]
    skip = {"test"}
    for k, v in result.items():
        if k in skip:
            continue
        if isinstance(v, float):
            lines.append(_fmt_row(k, f"{v:.4f}"))
        else:
            lines.append(_fmt_row(k, v))
    lines.append("-" * 50)
    return "\n".join(lines)


# ── Command ────────────────────────────────────────────────────────────────

@command("power", usage="power onemean|twomeans|oneprop|twoprop|ols [options]")
def cmd_power(session: Session, args: str) -> str:
    """Power analysis for common statistical tests."""
    positional, opts, flags = _stata_parse(args)

    if not positional:
        return (
            "Usage: power <subcommand> [options]\n"
            "Subcommands: onemean, twomeans, oneprop, twoprop, ols\n\n"
            "Examples:\n"
            "  power onemean, n(50) delta(0.5) sd(1)\n"
            "  power twomeans, n(80) delta(0.5) sd(1)\n"
            "  power oneprop, n(100) p0(0.5) pa(0.65)\n"
            "  power twoprop, p1(0.3) p2(0.5) power(0.80)\n"
            "  power ols, n(100) f2(0.15) k(3)"
        )

    sub = positional[0].lower()
    alpha = float(opts.get("alpha", 0.05))
    n = int(opts["n"]) if "n" in opts else None
    pwr = float(opts["power"]) if "power" in opts else None
    two_sided = "onesided" not in flags

    try:
        if sub in ("onemean", "one_mean"):
            delta = float(opts.get("delta", 0.5))
            sd = float(opts.get("sd", 1.0))
            es = float(opts["es"]) if "es" in opts else None
            result = power_onemean(
                effect_size=es, alpha=alpha, n=n, power=pwr,
                sd=sd, delta=delta, two_sided=two_sided,
            )

        elif sub in ("twomeans", "two_means"):
            delta = float(opts.get("delta", 0.5))
            sd = float(opts.get("sd", 1.0))
            ratio = float(opts.get("ratio", 1.0))
            n1 = int(opts["n1"]) if "n1" in opts else n
            es = float(opts["es"]) if "es" in opts else None
            result = power_twomeans(
                effect_size=es, alpha=alpha, n=n1, power=pwr,
                ratio=ratio, sd=sd, delta=delta, two_sided=two_sided,
            )

        elif sub in ("oneprop", "one_prop", "onepropo"):
            p0 = float(opts.get("p0", 0.5))
            pa = float(opts.get("pa", 0.6))
            result = power_oneproportion(
                p0=p0, pa=pa, alpha=alpha, n=n, power=pwr, two_sided=two_sided,
            )

        elif sub in ("twoprop", "two_prop", "twopropo"):
            p1 = float(opts.get("p1", 0.3))
            p2 = float(opts.get("p2", 0.5))
            result = power_twoproportions(
                p1=p1, p2=p2, alpha=alpha, n=n, power=pwr, two_sided=two_sided,
            )

        elif sub == "ols":
            f2 = float(opts["f2"]) if "f2" in opts else None
            k = int(opts.get("k", 1))
            result = power_ols(f2=f2, alpha=alpha, n=n, power=pwr, k=k)

        else:
            return f"Unknown power subcommand: {sub}"

    except (ValueError, TypeError) as exc:
        return f"Power analysis error: {exc}"

    return _power_table(result)


@command("sampsi", usage="sampsi mu1 mu2 [, sd(1) alpha(0.05) power(0.80)]")
def cmd_sampsi(session: Session, args: str) -> str:
    """Compute required sample size (Stata-style sampsi)."""
    positional, opts, flags = _stata_parse(args)

    # Filter only numeric positionals
    nums = [p for p in positional if re.match(r'^-?(\d+\.?\d*|\.\d+)$', p)]
    if len(nums) < 2:
        return "Usage: sampsi mu1 mu2 [, sd(1) alpha(0.05) power(0.80)]"

    try:
        mu1 = float(nums[0])
        mu2 = float(nums[1])
        sd = float(opts.get("sd", 1.0))
        alpha = float(opts.get("alpha", 0.05))
        pwr = float(opts.get("power", 0.80))
        result = _sampsi(mu1=mu1, mu2=mu2, sd=sd, alpha=alpha, power=pwr)
    except (ValueError, TypeError) as exc:
        return f"sampsi error: {exc}"

    return _power_table(result)
