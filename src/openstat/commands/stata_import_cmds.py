"""Stata .do file importer: convert Stata syntax to OpenStat .ost script."""

from __future__ import annotations

import re
from pathlib import Path

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session

# ── Stata → OpenStat translation rules ──────────────────────────────────────
# Each rule is (pattern, replacement_or_callable)
# Applied in order; first match wins for each line.

_RULES: list[tuple[re.Pattern, object]] = []


def _rule(pattern: str, repl):
    _RULES.append((re.compile(pattern, re.IGNORECASE), repl))


# Comments
_rule(r"^\s*\*.*$", lambda m: "# " + m.group(0).lstrip("* \t"))
_rule(r"^(\s*)//(.*)$", lambda m: m.group(1) + "# " + m.group(2))

# use → load
_rule(r"^\s*use\s+[\"']?(.+?)[\"']?\s*(?:,\s*clear)?\s*$",
      lambda m: f"load {m.group(1).strip()}")

# save → save
_rule(r"^\s*save\s+[\"']?(.+?)[\"']?\s*(?:,\s*replace)?\s*$",
      lambda m: f"save {m.group(1).strip()}")

# summarize → summarize
_rule(r"^\s*su(?:mmarize)?\s*(.*)", lambda m: f"summarize {m.group(1).strip()}")

# describe → describe
_rule(r"^\s*desc(?:ribe)?\s*(.*)", lambda m: f"describe")

# regress → ols
_rule(r"^\s*reg(?:ress)?\s+(\S+)\s+(.*)",
      lambda m: f"ols {m.group(1)} {m.group(2).strip()}")

# logit → logit (same)
_rule(r"^\s*logit\s+(\S+)\s+(.*)",
      lambda m: f"logit {m.group(1)} {m.group(2).strip()}")

# probit → probit (same)
_rule(r"^\s*probit\s+(\S+)\s+(.*)",
      lambda m: f"probit {m.group(1)} {m.group(2).strip()}")

# poisson → poisson (same)
_rule(r"^\s*poisson\s+(\S+)\s+(.*)",
      lambda m: f"poisson {m.group(1)} {m.group(2).strip()}")

# tab → tabulate
_rule(r"^\s*tab(?:ulate)?\s+(.*)", lambda m: f"tabulate {m.group(1).strip()}")

# cor → correlate
_rule(r"^\s*cor(?:relate)?\s+(.*)", lambda m: f"correlate {m.group(1).strip()}")

# drop if → filter
_rule(r"^\s*drop\s+if\s+(.*)",
      lambda m: f"# [manual] drop if {m.group(1)}  →  filter {_stata_cond(m.group(1))}")

# keep if → filter
_rule(r"^\s*keep\s+if\s+(.*)",
      lambda m: f"# [manual] keep if {m.group(1)}  →  filter {_stata_cond(m.group(1))}")

# drop varlist → drop
_rule(r"^\s*drop\s+(?!if)(.*)", lambda m: f"drop {m.group(1).strip()}")

# keep varlist → keep
_rule(r"^\s*keep\s+(?!if)(.*)", lambda m: f"keep {m.group(1).strip()}")

# gen/generate → generate
_rule(r"^\s*gen(?:erate)?\s+(.+)=(.+)",
      lambda m: f"generate {m.group(1).strip()} = {m.group(2).strip()}")

# replace → replace
_rule(r"^\s*replace\s+(.+)=(.+)",
      lambda m: f"replace {m.group(1).strip()} = {m.group(2).strip()}")

# rename → rename
_rule(r"^\s*rename\s+(\S+)\s+(\S+)", lambda m: f"rename {m.group(1)} {m.group(2)}")

# sort → sort
_rule(r"^\s*sort\s+(.*)", lambda m: f"sort {m.group(1).strip()}")

# set seed → set seed
_rule(r"^\s*set\s+seed\s+(\d+)", lambda m: f"set seed {m.group(1)}")

# xtset → xtset (panel)
_rule(r"^\s*xtset\s+(.*)", lambda m: f"xtset {m.group(1).strip()}")

# tsset → tsset
_rule(r"^\s*tsset\s+(.*)", lambda m: f"tsset {m.group(1).strip()}")

# stset → stset
_rule(r"^\s*stset\s+(.*)", lambda m: f"stset {m.group(1).strip()}")

# display → print (closest)
_rule(r"^\s*dis(?:play)?\s+(.*)", lambda m: f"# display {m.group(1).strip()}")

# local macro → define
_rule(r"^\s*local\s+(\w+)\s+(.*)", lambda m: f"define {m.group(1)} = {m.group(2).strip()}")

# forvalues
_rule(r"^\s*forvalues\s+(.*)", lambda m: f"forvalues {m.group(1).strip()} {{")

# foreach
_rule(r"^\s*foreach\s+(.*)", lambda m: f"foreach {m.group(1).strip()} {{")

# if / else blocks - pass through
_rule(r"^\s*\}\s*$", lambda m: "}")

# quietly → strip (run silently)
_rule(r"^\s*quietly\s+(.*)", lambda m: m.group(1).strip())

# capture → strip (ignore errors)
_rule(r"^\s*capture\s+(.*)", lambda m: m.group(1).strip())


def _stata_cond(cond: str) -> str:
    """Very rough Stata condition → Polars filter hint."""
    # == → ==, != → !=, & → &, | → |
    cond = cond.replace(" & ", " & ").replace(" | ", " | ")
    return cond


def _translate_line(line: str) -> str:
    """Translate a single Stata .do line to OpenStat syntax."""
    stripped = line.rstrip()
    if not stripped.strip():
        return ""

    for pat, repl in _RULES:
        m = pat.match(stripped)
        if m:
            if callable(repl):
                return repl(m)
            return repl
    # No match — keep as comment with note
    return f"# [untranslated] {stripped}"


def convert_do_file(do_text: str) -> str:
    """Convert Stata .do file content to OpenStat .ost script."""
    out_lines = ["# OpenStat script — converted from Stata .do file", ""]
    for line in do_text.splitlines():
        translated = _translate_line(line)
        out_lines.append(translated)
    return "\n".join(out_lines)


@command("import do", usage="import do <file.do> [--out=<file.ost>]")
def cmd_import_do(session: Session, args: str) -> str:
    """Convert a Stata .do file to an OpenStat .ost script.

    Translates common Stata commands (use, regress, gen, etc.) to their
    OpenStat equivalents. Untranslated lines are kept as comments.

    Options:
      --out=<path>   output .ost file path (default: same name, .ost extension)
      --run          also execute the converted script immediately

    Examples:
      import do analysis.do
      import do stata_script.do --out=my_analysis.ost --run
    """
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: import do <file.do> [--out=<path>] [--run]"

    do_path = Path(ca.positional[0])
    if not do_path.exists():
        return f"File not found: {do_path}"

    out_path_str = ca.options.get("out", str(do_path.with_suffix(".ost")))
    out_path = Path(out_path_str)
    run_after = "--run" in args

    try:
        do_text = do_path.read_text(encoding="utf-8", errors="replace")
        ost_text = convert_do_file(do_text)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(ost_text, encoding="utf-8")

        n_lines = len(do_text.splitlines())
        n_translated = sum(
            1 for line in ost_text.splitlines()
            if line.strip() and not line.startswith("# [untranslated]")
            and not line.startswith("# OpenStat")
        )
        n_untranslated = ost_text.count("# [untranslated]")

        result = (
            f"Converted: {do_path} → {out_path}\n"
            f"  Lines: {n_lines}  |  Translated: {n_translated}  |  "
            f"Untranslated (kept as comments): {n_untranslated}"
        )

        if run_after:
            from openstat.script_runner import run_script_advanced
            try:
                run_script_advanced(str(out_path), session)
                result += f"\nScript executed: {out_path}"
            except Exception as exc:
                result += f"\nScript error: {exc}"

        return result

    except Exception as e:
        return friendly_error(e, "import do")
