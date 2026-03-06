"""Enhanced help command with examples and category browsing."""

from __future__ import annotations

from openstat.commands.base import command, get_registry, get_usage
from openstat.session import Session

# ── Command categories and examples ─────────────────────────────────────────

_CATEGORIES: dict[str, list[str]] = {
    "Data Management": [
        "load", "save", "describe", "summarize", "drop", "keep", "rename",
        "generate", "replace", "sort", "filter", "sample", "undo", "redo",
        "append", "merge", "reshape", "encode", "decode", "validate",
        "fuzzyjoin", "regex", "sql", "sqlload",
    ],
    "Descriptive Statistics": [
        "tabulate", "correlate", "crosstab", "anova",
    ],
    "Regression Models": [
        "ols", "logit", "probit", "poisson", "ivregress",
        "quantreg", "truncreg", "intreg", "heckman",
    ],
    "Post-Estimation": [
        "margins", "predict", "test", "posthoc", "mediate", "modmediate",
        "estat", "esttab", "outreg2",
    ],
    "Panel / Time Series": [
        "xtset", "xtreg", "xttest", "hausman",
        "tsset", "arima", "ardl", "adf", "kpss", "forecast",
        "vecm", "var", "granger", "threshold", "arch", "garch",
    ],
    "Survival Analysis": [
        "stset", "stcox", "streg", "stsum",
    ],
    "Causal Inference": [
        "pscore", "teffects", "did", "rddesign", "iptw",
    ],
    "Machine Learning": [
        "mlfit", "randomforest", "gradientboost", "neuralnet", "svm", "knn",
        "automodel", "cluster", "discriminant",
    ],
    "Factor / SEM": [
        "factor", "pca", "sem", "cfa",
    ],
    "Epidemiology / Biostatistics": [
        "epi", "irt",
    ],
    "Meta-analysis": [
        "meta",
    ],
    "Network Analysis": [
        "network",
    ],
    "Non-parametric": [
        "kruskal", "mannwhitney", "wilcoxon", "friedman",
    ],
    "Resampling & Inference": [
        "bootstrap", "jackknife", "permtest", "power",
    ],
    "Model Evaluation": [
        "roc", "calibration", "confusion", "influence",
    ],
    "Plots": [
        "plot hist", "plot scatter", "plot line", "plot box", "plot bar",
        "plot heatmap", "plot coef", "plot margins", "plot interaction",
        "plot diagnostics", "plot violin", "plot pairplot",
    ],
    "Export & Reporting": [
        "export docx", "export pptx", "export pdf", "export md",
        "log using", "report",
    ],
    "Session & Reproducibility": [
        "session info", "session save", "session replay",
        "set seed", "set backend", "version",
    ],
    "Scripting": [
        "run", "watch", "import do", "define", "alias",
    ],
    "Settings & Integration": [
        "theme", "locale", "dashboard", "ask", "r",
        "config", "plugin",
    ],
}

_EXAMPLES: dict[str, list[str]] = {
    "load":        ["load data.csv", "load survey.dta", "load results.parquet"],
    "save":        ["save cleaned.csv", "save output.parquet"],
    "ols":         ["ols income educ age", "ols y x1 x2 x3 --robust"],
    "logit":       ["logit employed age educ female", "logit y x1 x2 --margins"],
    "margins":     ["margins educ", "margins age --at(educ=12)"],
    "mediate":     ["mediate income educ age --boot=2000"],
    "modmediate":  ["modmediate y mediator x moderator"],
    "plot":        [
        "plot hist income", "plot scatter y x", "plot coef",
        "plot interaction y x moderator", "plot margins",
    ],
    "export":      ["export docx", "export pdf report.pdf", "export md summary.md"],
    "validate":    ["validate age min=0 max=120 notnull", "validate email regex=^[^@]+@[^@]+"],
    "fuzzyjoin":   ["fuzzyjoin companies.csv on(name) --threshold=85"],
    "regex":       [
        'regex extract email "([^@]+)@" into(username)',
        'regex replace phone "[^0-9]" "" into(phone_clean)',
    ],
    "alias":       ["alias reg ols", "alias list", "alias rm reg"],
    "theme":       ["theme dark", "theme solarized", "theme list"],
    "ask":         ['ask "What variables have the most missing data?"'],
    "watch":       ["watch analysis.ost", "watch pipeline.ost --interval=5"],
    "import do":   ["import do stata_script.do", "import do analysis.do --run"],
    "dashboard":   ["dashboard"],
    "session":     ["session info", "session save analysis.ost", "session replay analysis.ost"],
    "sem":         ["sem 'y1 =~ x1 + x2\ny2 =~ x3 + x4'"],
    "meta":        ["meta es se --random --forest"],
    "network":     ["network build from src to dst", "network centrality", "network plot"],
    "automodel":   ["automodel y x1 x2 x3 x4 x5 --criterion=aic"],
    "bootstrap":   ["bootstrap ols income educ age --reps=500"],
    "power":       ["power ttest --delta=0.5 --alpha=0.05"],
    "r":           ['r "summary(data)"', 'r "cor(data)"'],
}


@command("help", usage="help [<command>] [--list] [--category=<name>]")
def cmd_help(session: Session, args: str) -> str:
    """Show help for a command, list all commands, or browse by category.

    Usage:
      help                        — show command categories
      help <command>              — detailed help with examples
      help --list                 — alphabetical list of all commands
      help --category=Plots       — list commands in a category
      help --search=<keyword>     — search command names and descriptions

    Examples:
      help ols
      help plot
      help --list
      help --category="Machine Learning"
      help --search=regression
    """
    from openstat.commands.base import _REGISTRY

    tokens = args.strip().split()

    # --list flag
    if "--list" in tokens:
        cmds = sorted(_REGISTRY.keys())
        lines = [f"All commands ({len(cmds)}):", "=" * 50]
        for i, name in enumerate(cmds):
            usage = get_usage(name)
            short_desc = _REGISTRY[name].__doc__ or ""
            short_desc = short_desc.strip().split("\n")[0][:60] if short_desc else ""
            lines.append(f"  {name:<22} {short_desc}")
        return "\n".join(lines)

    # --search flag
    search_term = None
    for t in tokens:
        if t.startswith("--search="):
            search_term = t[9:].lower()
    if search_term:
        matches = []
        for name, handler in sorted(_REGISTRY.items()):
            doc = (handler.__doc__ or "").lower()
            if search_term in name.lower() or search_term in doc:
                short = (handler.__doc__ or "").strip().split("\n")[0][:60]
                matches.append(f"  {name:<22} {short}")
        if matches:
            return f"Search results for '{search_term}':\n" + "\n".join(matches)
        return f"No commands matching '{search_term}'."

    # --category flag
    for t in tokens:
        if t.startswith("--category="):
            cat = t[11:].strip('"\'')
            # Case-insensitive match
            matched_cat = next(
                (k for k in _CATEGORIES if k.lower() == cat.lower()), None
            )
            if matched_cat is None:
                cats = "\n".join(f"  {k}" for k in _CATEGORIES)
                return f"Category '{cat}' not found. Available:\n{cats}"
            cmds = _CATEGORIES[matched_cat]
            lines = [f"Category: {matched_cat}", "-" * 40]
            for name in cmds:
                handler = _REGISTRY.get(name)
                if handler:
                    short = (handler.__doc__ or "").strip().split("\n")[0][:60]
                    lines.append(f"  {name:<22} {short}")
                else:
                    lines.append(f"  {name:<22} (not loaded)")
            return "\n".join(lines)

    # Specific command help
    cmd_name = " ".join(tokens).strip() if tokens else ""
    if cmd_name and cmd_name in _REGISTRY:
        handler = _REGISTRY[cmd_name]
        usage = get_usage(cmd_name)
        doc = (handler.__doc__ or "").strip()
        examples = _EXAMPLES.get(cmd_name, [])

        lines = [
            f"Command: {cmd_name}",
            "=" * 50,
            f"Usage: {usage}",
            "",
        ]
        if doc:
            lines += [doc, ""]
        if examples:
            lines += ["Examples:"]
            for ex in examples:
                lines.append(f"  {ex}")
        return "\n".join(lines)

    # No args — show category overview
    lines = [
        "OpenStat Help — type 'help <command>' for details",
        "=" * 55,
        "Options: help --list | help --search=<kw> | help --category=<name>",
        "",
        "Command Categories:",
        "-" * 55,
    ]
    for cat, cmds in _CATEGORIES.items():
        available = [c for c in cmds if c in _REGISTRY]
        if available:
            lines.append(f"  {cat:<30} ({len(available)} commands)")
    lines += [
        "",
        "Quick start:",
        "  load data.csv",
        "  describe",
        "  ols outcome predictor1 predictor2",
        "  plot coef",
        "  export pdf",
    ]
    return "\n".join(lines)
