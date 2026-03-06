"""Multiple imputation commands: mi impute, mi estimate, mi describe."""

from __future__ import annotations

import re
import io

import polars as pl
from rich.console import Console
from rich.table import Table

from openstat.session import Session, ModelResult
from openstat.commands.base import command, CommandArgs, rich_to_str, friendly_error
from openstat.dsl.parser import parse_formula, ParseError


def _parse_impute_specs(args: str) -> tuple[list[tuple[str, str]], dict]:
    """Parse imputation specification.

    Formats:
        chained (regress) x1 (logit) x2, add(5)
        pmm x1 x2, add(10) knn(5)

    Returns (specs, options).
    """
    # Split on comma for options
    if ',' in args:
        spec_part, opt_part = args.split(',', 1)
    else:
        spec_part, opt_part = args, ""

    # Parse options
    options: dict = {}
    m_add = re.search(r'add\((\d+)\)', opt_part)
    if m_add:
        options["m"] = int(m_add.group(1))
    m_knn = re.search(r'knn\((\d+)\)', opt_part)
    if m_knn:
        options["knn"] = int(m_knn.group(1))

    spec_part = spec_part.strip()
    specs: list[tuple[str, str]] = []

    if spec_part.startswith("chained"):
        # Parse (method) var pairs
        spec_str = spec_part[len("chained"):].strip()
        pattern = r'\((\w+)\)\s+(\w+)'
        for match in re.finditer(pattern, spec_str):
            method = match.group(1)
            col = match.group(2)
            specs.append((method, col))
    elif spec_part.startswith("pmm"):
        # PMM for all listed variables
        cols = spec_part[len("pmm"):].strip().split()
        for col in cols:
            specs.append(("pmm", col))
    else:
        # Default: regress for all variables
        cols = spec_part.split()
        for col in cols:
            specs.append(("regress", col))

    return specs, options


@command("mi", usage="mi impute|estimate|describe ...")
def cmd_mi(session: Session, args: str) -> str:
    """Multiple imputation: impute missing data, estimate models, describe imputations."""
    df = session.require_data()

    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd == "impute":
        return _mi_impute(session, df, rest)
    elif subcmd.startswith("estimate"):
        # Handle "estimate:" or "estimate :"
        model_cmd = rest
        if model_cmd.startswith(":"):
            model_cmd = model_cmd[1:].strip()
        elif ":" in args:
            model_cmd = args.split(":", 1)[1].strip()
        return _mi_estimate(session, model_cmd)
    elif subcmd == "describe":
        return _mi_describe(session)
    else:
        return "Usage: mi impute|estimate|describe\n  mi impute chained (regress) x1 (logit) x2, add(5)\n  mi estimate: ols y ~ x1 + x2\n  mi describe"

def _mi_impute(session: Session, df: pl.DataFrame, args: str) -> str:
    """Run MICE imputation."""
    specs, options = _parse_impute_specs(args)
    if not specs:
        return "No variables specified for imputation."

    m = options.get("m", 5)
    knn = options.get("knn", 5)

    # Validate columns
    for method, col in specs:
        if col not in df.columns:
            return f"Column not found: {col}"

    # Check for missing values (both null and NaN)
    has_missing = False
    for _, col in specs:
        if df[col].null_count() > 0:
            has_missing = True
            break
        if df[col].dtype.is_float() and df[col].is_nan().sum() > 0:
            has_missing = True
            break
    if not has_missing:
        return "No missing values found in specified columns."

    try:
        from openstat.stats.imputation import mice_impute

        datasets = mice_impute(df, specs, m=m)
        session._imputed_datasets = datasets
        session._mi_m = m

        lines = [f"Created {m} imputed datasets."]
        for method, col in specs:
            n_missing = df[col].null_count()
            lines.append(f"  {col}: {n_missing} missing values imputed ({method})")
        lines.append(f"\nUse 'mi estimate: <model>' to run analysis across imputed datasets.")
        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "mi impute")


def _mi_estimate(session: Session, model_cmd: str) -> str:
    """Run model on each imputed dataset and combine with Rubin's rules."""
    if not session._imputed_datasets:
        return "No imputed datasets. Run 'mi impute' first."

    model_cmd = model_cmd.strip()
    if not model_cmd:
        return "Usage: mi estimate: ols y ~ x1 + x2"

    # Parse model type and formula
    parts = model_cmd.split(None, 1)
    model_type = parts[0].lower()
    formula_str = parts[1] if len(parts) > 1 else ""

    try:
        dep, indeps = parse_formula(formula_str)
    except ParseError as e:
        return f"Formula error: {e}"

    estimates: list[dict[str, float]] = []
    std_errors: list[dict[str, float]] = []

    for i, imp_df in enumerate(session._imputed_datasets):
        try:
            if model_type == "ols":
                from openstat.stats.models import fit_ols
                result, _ = fit_ols(imp_df, dep, indeps)
            elif model_type == "logit":
                from openstat.stats.models import fit_logit
                result, _ = fit_logit(imp_df, dep, indeps)
            else:
                return f"Unsupported model for MI: {model_type}. Use ols or logit."

            estimates.append(dict(result.params))
            std_errors.append(dict(result.std_errors))
        except Exception as e:
            return f"Error fitting imputation {i + 1}: {e}"

    # Combine with Rubin's rules
    from openstat.stats.imputation import rubins_rules
    n_obs = session._imputed_datasets[0].height
    mi_result = rubins_rules(estimates, std_errors, n_obs)
    mi_result.model_type = f"MI ({mi_result.m} imputations): {model_type.upper()}"
    mi_result.formula = f"{dep} ~ {' + '.join(indeps)}"

    # Display as table
    def render(console: Console) -> None:
        table = Table(title=mi_result.model_type)
        table.add_column("Variable", style="cyan")
        table.add_column("Coef", justify="right")
        table.add_column("MI Std.Err", justify="right")
        table.add_column("t", justify="right")
        table.add_column("P>|t|", justify="right")
        table.add_column("[95% CI Low]", justify="right")
        table.add_column("[95% CI High]", justify="right")
        table.add_column("FMI", justify="right")

        for var in mi_result.params:
            sig = ""
            pv = mi_result.p_values[var]
            if pv < 0.001:
                sig = " ***"
            elif pv < 0.01:
                sig = " **"
            elif pv < 0.05:
                sig = " *"

            table.add_row(
                var,
                f"{mi_result.params[var]:.4f}",
                f"{mi_result.std_errors[var]:.4f}",
                f"{mi_result.t_values[var]:.3f}",
                f"{pv:.4f}{sig}",
                f"{mi_result.conf_int_low[var]:.4f}",
                f"{mi_result.conf_int_high[var]:.4f}",
                f"{mi_result.fmi[var]:.3f}",
            )
        console.print(table)

    output = rich_to_str(render)
    output += f"\nN = {mi_result.n_obs}  |  Imputations = {mi_result.m}"
    return output


def _mi_describe(session: Session) -> str:
    """Describe imputed datasets."""
    if not session._imputed_datasets:
        return "No imputed datasets. Run 'mi impute' first."

    lines = [
        f"Multiple Imputation Summary:",
        f"  Number of imputations (m): {session._mi_m}",
        f"  Rows per dataset: {session._imputed_datasets[0].height}",
        f"  Columns: {session._imputed_datasets[0].width}",
    ]
    return "\n".join(lines)
