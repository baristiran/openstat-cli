"""Extended data manipulation commands:
cast (multi-col), lag, lead, cumulative, bin, antijoin, semijoin,
sample stratified, anonymize.
"""

from __future__ import annotations

import re

import polars as pl

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, pl.DataType] = {
    "int":      pl.Int64,
    "float":    pl.Float64,
    "str":      pl.Utf8,
    "bool":     pl.Boolean,
    "date":     pl.Date,
    "datetime": pl.Datetime,
}

_VALID_TYPES = ", ".join(_TYPE_MAP)


def _parse_into(args: str) -> tuple[str, str | None]:
    """Strip trailing into(<newcol>) from an arg string.

    Returns (remaining_args, new_col_or_None).
    """
    m = re.search(r'\binto\(\s*(\w+)\s*\)', args, re.IGNORECASE)
    if m:
        new_col = m.group(1)
        remaining = (args[: m.start()] + args[m.end():]).strip()
        return remaining, new_col
    return args, None


# ---------------------------------------------------------------------------
# cast — multi-column version
# ---------------------------------------------------------------------------

@command("cast", usage="cast <col> <type> [col2 type2 ...]")
def cmd_cast(session: Session, args: str) -> str:
    """Cast one or more columns to a new type.

    Types: int, float, str, bool, date, datetime
    Example: cast age int income float gender str
    """
    df = session.require_data()
    tokens = args.split()
    if len(tokens) < 2:
        return (
            "Usage: cast <col> <type> [col2 type2 ...]\n"
            f"Types: {_VALID_TYPES}"
        )

    # Pair up tokens: (col, type), (col, type), ...
    if len(tokens) % 2 != 0:
        return (
            "Provide pairs of <col> <type>.\n"
            f"Types: {_VALID_TYPES}"
        )

    pairs: list[tuple[str, str]] = [
        (tokens[i], tokens[i + 1].lower()) for i in range(0, len(tokens), 2)
    ]

    # Validate all before touching the data
    for col, tname in pairs:
        if col not in df.columns:
            return f"Column not found: '{col}'. Use 'describe' to list columns."
        if tname not in _TYPE_MAP:
            return f"Unknown type '{tname}'. Valid types: {_VALID_TYPES}"

    session.snapshot()
    lines: list[str] = []
    try:
        for col, tname in pairs:
            old_dtype = str(df[col].dtype)
            pl_type = _TYPE_MAP[tname]
            session.df = session.df.with_columns(pl.col(col).cast(pl_type))
            lines.append(f"  '{col}': {old_dtype} -> {tname}")
        lines.insert(0, f"Cast {len(pairs)} column(s):")
        lines.append("Use 'undo' to revert.")
        return "\n".join(lines)
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "cast")


# ---------------------------------------------------------------------------
# lag
# ---------------------------------------------------------------------------

@command("lag", usage="lag <col> [n=1] [into(<newcol>)]")
def cmd_lag(session: Session, args: str) -> str:
    """Create a lagged column (shift values forward by n rows, default n=1).

    Example: lag price 2 into(price_lag2)
    """
    df = session.require_data()

    # Strip into() first
    rest, new_col = _parse_into(args)
    tokens = rest.split()

    if not tokens:
        return "Usage: lag <col> [n=1] [into(<newcol>)]"

    col = tokens[0]
    if col not in df.columns:
        return friendly_error(KeyError(col), "lag")

    n = 1
    if len(tokens) >= 2:
        try:
            n = int(tokens[1])
        except ValueError:
            return f"n must be an integer, got '{tokens[1]}'"

    if new_col is None:
        new_col = f"{col}_lag{n}"

    session.snapshot()
    try:
        session.df = df.with_columns(
            pl.col(col).shift(n).alias(new_col)
        )
        return (
            f"Created '{new_col}' as lag({col}, {n}). "
            "Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "lag")


# ---------------------------------------------------------------------------
# lead
# ---------------------------------------------------------------------------

@command("lead", usage="lead <col> [n=1] [into(<newcol>)]")
def cmd_lead(session: Session, args: str) -> str:
    """Create a lead column (shift values backward by n rows, default n=1).

    Example: lead gdp 3 into(gdp_lead3)
    """
    df = session.require_data()

    rest, new_col = _parse_into(args)
    tokens = rest.split()

    if not tokens:
        return "Usage: lead <col> [n=1] [into(<newcol>)]"

    col = tokens[0]
    if col not in df.columns:
        return friendly_error(KeyError(col), "lead")

    n = 1
    if len(tokens) >= 2:
        try:
            n = int(tokens[1])
        except ValueError:
            return f"n must be an integer, got '{tokens[1]}'"

    if new_col is None:
        new_col = f"{col}_lead{n}"

    session.snapshot()
    try:
        session.df = df.with_columns(
            pl.col(col).shift(-n).alias(new_col)
        )
        return (
            f"Created '{new_col}' as lead({col}, {n}). "
            "Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "lead")


# ---------------------------------------------------------------------------
# cumulative
# ---------------------------------------------------------------------------

_CUM_FUNCS: dict[str, str] = {
    "sum":   "cum_sum",
    "prod":  "cum_prod",
    "max":   "cum_max",
    "min":   "cum_min",
    "count": "cum_count",
}


@command("cumulative", usage="cumulative <col> <func> [into(<newcol>)]")
def cmd_cumulative(session: Session, args: str) -> str:
    """Compute a cumulative statistic along a column.

    func: sum, prod, max, min, count
    Example: cumulative sales sum into(sales_cumsum)
    """
    df = session.require_data()

    rest, new_col = _parse_into(args)
    tokens = rest.split()

    if len(tokens) < 2:
        valid = ", ".join(_CUM_FUNCS)
        return f"Usage: cumulative <col> <func> [into(<newcol>)]  (func: {valid})"

    col, func_name = tokens[0], tokens[1].lower()
    if col not in df.columns:
        return friendly_error(KeyError(col), "cumulative")

    if func_name not in _CUM_FUNCS:
        valid = ", ".join(_CUM_FUNCS)
        return f"Unknown function '{func_name}'. Valid: {valid}"

    if new_col is None:
        new_col = f"{col}_cum{func_name}"

    pl_method = _CUM_FUNCS[func_name]

    session.snapshot()
    try:
        expr = getattr(pl.col(col), pl_method)()
        session.df = df.with_columns(expr.alias(new_col))
        return (
            f"Created '{new_col}' as cumulative {func_name}({col}). "
            "Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "cumulative")


# ---------------------------------------------------------------------------
# bin
# ---------------------------------------------------------------------------

@command("bin", usage="bin <col> <n_bins> [into(<newcol>)] [--labels=a,b,c] [--equal-freq]")
def cmd_bin(session: Session, args: str) -> str:
    """Discretize a continuous column into n_bins bins.

    --equal-freq  use quantile-based (equal-frequency) bins; default is equal-width.
    --labels=a,b  comma-separated bin labels (must match n_bins).
    Example: bin income 5 into(income_cat)
    Example: bin age 3 --labels=young,middle,senior --equal-freq
    """
    df = session.require_data()
    ca = CommandArgs(args)

    # Strip into() from the raw string before positional parsing
    raw_no_into, new_col = _parse_into(ca.strip_flags_and_options())
    pos_tokens = raw_no_into.split()

    if len(pos_tokens) < 2:
        return "Usage: bin <col> <n_bins> [into(<newcol>)] [--labels=a,b,c] [--equal-freq]"

    col = pos_tokens[0]
    try:
        n_bins = int(pos_tokens[1])
    except ValueError:
        return f"n_bins must be an integer, got '{pos_tokens[1]}'"

    if n_bins < 2:
        return "n_bins must be >= 2"

    if col not in df.columns:
        return friendly_error(KeyError(col), "bin")

    equal_freq = ca.has_flag("--equal-freq")
    labels_raw = ca.options.get("labels")
    labels: list[str] | None = None
    if labels_raw:
        labels = [lb.strip() for lb in labels_raw.split(",")]
        if len(labels) != n_bins:
            return (
                f"Number of labels ({len(labels)}) must match n_bins ({n_bins})."
            )

    if new_col is None:
        new_col = f"{col}_bin"

    series = df[col].cast(pl.Float64)
    non_null = series.drop_nulls()

    if non_null.len() == 0:
        return f"Column '{col}' has no non-null values."

    try:
        if equal_freq:
            # Quantile breakpoints
            quantiles = [i / n_bins for i in range(n_bins + 1)]
            breaks = [float(non_null.quantile(q, interpolation="linear")) for q in quantiles]
            breaks[0] -= 1e-10   # include minimum
        else:
            lo = float(non_null.min())
            hi = float(non_null.max())
            step = (hi - lo) / n_bins
            breaks = [lo + i * step for i in range(n_bins + 1)]
            breaks[0] -= 1e-10

        def _bin_value(v: float | None) -> str | None:
            if v is None:
                return None
            for i in range(n_bins):
                if breaks[i] < v <= breaks[i + 1]:
                    if labels:
                        return labels[i]
                    lo_s = f"{breaks[i]:.4g}"
                    hi_s = f"{breaks[i + 1]:.4g}"
                    return f"({lo_s}, {hi_s}]"
            # Edge: value equals the minimum
            if labels:
                return labels[0]
            lo_s = f"{breaks[0]:.4g}"
            hi_s = f"{breaks[1]:.4g}"
            return f"({lo_s}, {hi_s}]"

        bin_col = pl.Series(
            name=new_col,
            values=[_bin_value(v) for v in series.to_list()],
            dtype=pl.Utf8,
        )
    except Exception as exc:
        return friendly_error(exc, "bin")

    session.snapshot()
    session.df = df.with_columns(bin_col)
    method = "equal-frequency" if equal_freq else "equal-width"
    return (
        f"Binned '{col}' into {n_bins} {method} bins -> '{new_col}'. "
        "Use 'undo' to revert."
    )


# ---------------------------------------------------------------------------
# antijoin
# ---------------------------------------------------------------------------

@command("antijoin", usage="antijoin <file> on(<col>) [how=left|right]")
def cmd_antijoin(session: Session, args: str) -> str:
    """Keep rows from current dataset that are NOT found in another file.

    Example: antijoin excluded_ids.csv on(id)
    """
    from openstat.io.loader import load_file

    df = session.require_data()
    ca = CommandArgs(args)

    # Parse on(<col>)
    on_m = re.search(r'\bon\(\s*(\w+)\s*\)', args, re.IGNORECASE)
    if not on_m:
        return "Usage: antijoin <file> on(<col>)"
    key_col = on_m.group(1)

    # File is everything before the on(...) token
    file_part = args[: on_m.start()].strip()
    # Remove any --options or flags from the file part
    file_path = re.sub(r'--\S+', '', file_part).strip()

    if not file_path:
        return "Usage: antijoin <file> on(<col>)"

    if key_col not in df.columns:
        return f"Key column '{key_col}' not found in current dataset."

    try:
        other = load_file(file_path)
    except Exception as exc:
        return f"Cannot load file: {exc}"

    if key_col not in other.columns:
        return f"Key column '{key_col}' not found in '{file_path}'."

    session.snapshot()
    try:
        # Anti-join: left join then filter for nulls from right side
        right_keys = other.select(pl.col(key_col).alias("__right_key__")).unique()
        merged = df.join(
            right_keys,
            left_on=key_col,
            right_on="__right_key__",
            how="left",
        )
        mask = merged["__right_key__"].is_null()
        session.df = df.filter(mask)
        kept = session.df.height
        total = df.height
        return (
            f"Anti-join on '{key_col}': kept {kept:,} of {total:,} rows "
            f"(removed {total - kept:,} matching rows). Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "antijoin")


# ---------------------------------------------------------------------------
# semijoin
# ---------------------------------------------------------------------------

@command("semijoin", usage="semijoin <file> on(<col>)")
def cmd_semijoin(session: Session, args: str) -> str:
    """Keep rows from current dataset that ARE found in another file.

    Example: semijoin valid_ids.csv on(id)
    """
    from openstat.io.loader import load_file

    df = session.require_data()

    on_m = re.search(r'\bon\(\s*(\w+)\s*\)', args, re.IGNORECASE)
    if not on_m:
        return "Usage: semijoin <file> on(<col>)"
    key_col = on_m.group(1)

    file_part = re.sub(r'--\S+', '', args[: on_m.start()]).strip()
    if not file_part:
        return "Usage: semijoin <file> on(<col>)"

    if key_col not in df.columns:
        return f"Key column '{key_col}' not found in current dataset."

    try:
        other = load_file(file_part)
    except Exception as exc:
        return f"Cannot load file: {exc}"

    if key_col not in other.columns:
        return f"Key column '{key_col}' not found in '{file_part}'."

    session.snapshot()
    try:
        right_keys = other.select(pl.col(key_col).alias("__right_key__")).unique()
        merged = df.join(
            right_keys,
            left_on=key_col,
            right_on="__right_key__",
            how="left",
        )
        mask = merged["__right_key__"].is_not_null()
        session.df = df.filter(mask)
        kept = session.df.height
        total = df.height
        return (
            f"Semi-join on '{key_col}': kept {kept:,} of {total:,} rows "
            f"(removed {total - kept:,} non-matching rows). Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "semijoin")


# ---------------------------------------------------------------------------
# sample stratified
# ---------------------------------------------------------------------------

@command("sample stratified", usage="sample stratified <n_or_frac> by(<stratum_col>) [--seed=N]")
def cmd_sample_stratified(session: Session, args: str) -> str:
    """Stratified random sample: draw proportionally from each stratum.

    n_or_frac: integer (absolute per stratum) or float < 1 (fraction of each stratum).
    Example: sample stratified 100 by(region)
    Example: sample stratified 0.2 by(gender) --seed=42
    """
    df = session.require_data()
    ca = CommandArgs(args)

    # Parse by(<col>)
    by_m = re.search(r'\bby\(\s*(\w+)\s*\)', args, re.IGNORECASE)
    if not by_m:
        return "Usage: sample stratified <n_or_frac> by(<stratum_col>) [--seed=N]"
    stratum_col = by_m.group(1)

    if stratum_col not in df.columns:
        return f"Stratum column '{stratum_col}' not found."

    # First positional token is n_or_frac
    pos_tokens = ca.positional
    if not pos_tokens:
        return "Provide a sample size or fraction as the first argument."

    try:
        n_or_frac_raw = pos_tokens[0]
        if "." in n_or_frac_raw:
            n_or_frac: float | int = float(n_or_frac_raw)
            use_frac = True
        else:
            n_or_frac = int(n_or_frac_raw)
            use_frac = False
    except ValueError:
        return f"n_or_frac must be a number, got '{pos_tokens[0]}'"

    if use_frac and not (0 < n_or_frac < 1):
        return "Fraction must be between 0 and 1 (exclusive)."
    if not use_frac and n_or_frac <= 0:
        return "Sample size must be a positive integer."

    seed_str = ca.options.get("seed")
    seed: int | None = None
    if seed_str is not None:
        try:
            seed = int(seed_str)
        except ValueError:
            return f"--seed must be an integer, got '{seed_str}'"

    session.snapshot()
    try:
        strata = df[stratum_col].unique().to_list()
        parts: list[pl.DataFrame] = []
        for stratum_val in strata:
            stratum_df = df.filter(pl.col(stratum_col) == stratum_val)
            stratum_n = stratum_df.height
            if stratum_n == 0:
                continue
            if use_frac:
                take = max(1, int(stratum_n * float(n_or_frac)))
            else:
                take = min(int(n_or_frac), stratum_n)
            parts.append(stratum_df.sample(n=take, shuffle=True, seed=seed))

        if not parts:
            session.undo()
            return "No strata found — dataset may be empty."

        session.df = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=seed)
        total_sampled = session.df.height
        mode_desc = f"{n_or_frac} per stratum" if not use_frac else f"{int(float(n_or_frac)*100)}% per stratum"
        return (
            f"Stratified sample ({mode_desc}, {len(strata)} strata): "
            f"{total_sampled:,} rows drawn from {df.height:,}. "
            "Use 'undo' to revert."
        )
    except Exception as exc:
        session.undo()
        return friendly_error(exc, "sample stratified")


# ---------------------------------------------------------------------------
# anonymize
# ---------------------------------------------------------------------------

@command("anonymize", usage="anonymize <col> [col2 ...] [--method=mask|hash|noise|drop]")
def cmd_anonymize(session: Session, args: str) -> str:
    """Anonymize columns using a chosen method.

    Methods:
      mask   Replace string values with asterisks, keeping first and last char.
      hash   Replace values with their SHA-256 hex digest.
      noise  Add Gaussian noise to numeric columns (--noise_std=0.1 controls scale).
      drop   Drop the column(s) entirely.

    Example: anonymize name email --method=mask
    Example: anonymize ssn --method=hash
    Example: anonymize income --method=noise --noise_std=0.05
    """
    import hashlib

    df = session.require_data()
    ca = CommandArgs(args)

    cols = ca.positional
    if not cols:
        return "Usage: anonymize <col> [col2 ...] [--method=mask|hash|noise|drop]"

    method = ca.options.get("method", "mask").lower()
    valid_methods = {"mask", "hash", "noise", "drop"}
    if method not in valid_methods:
        return f"Unknown method '{method}'. Valid: {', '.join(sorted(valid_methods))}"

    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    noise_std = 0.1
    if "noise_std" in ca.options:
        try:
            noise_std = float(ca.options["noise_std"])
        except ValueError:
            return f"--noise_std must be a float, got '{ca.options['noise_std']}'"

    session.snapshot()
    try:
        if method == "drop":
            session.df = df.drop(cols)
            return (
                f"Dropped {len(cols)} column(s): {', '.join(cols)}. "
                "Use 'undo' to revert."
            )

        work_df = df
        report_lines: list[str] = []

        for col in cols:
            dtype = df[col].dtype
            is_numeric = dtype in (
                pl.Float32, pl.Float64,
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            )

            if method == "mask":
                def _mask(v: object) -> str | None:
                    if v is None:
                        return None
                    s = str(v)
                    if len(s) <= 2:
                        return "*" * len(s)
                    return s[0] + "*" * (len(s) - 2) + s[-1]

                masked = pl.Series(
                    name=col,
                    values=[_mask(v) for v in work_df[col].to_list()],
                    dtype=pl.Utf8,
                )
                work_df = work_df.with_columns(masked)
                report_lines.append(f"  '{col}': masked (first+last char kept)")

            elif method == "hash":
                def _hash(v: object) -> str | None:
                    if v is None:
                        return None
                    raw = str(v).encode("utf-8")
                    return hashlib.sha256(raw).hexdigest()

                hashed = pl.Series(
                    name=col,
                    values=[_hash(v) for v in work_df[col].to_list()],
                    dtype=pl.Utf8,
                )
                work_df = work_df.with_columns(hashed)
                report_lines.append(f"  '{col}': SHA-256 hashed")

            elif method == "noise":
                if not is_numeric:
                    report_lines.append(
                        f"  '{col}': skipped (noise requires numeric column, got {dtype})"
                    )
                    continue
                import random as _random
                vals = work_df[col].cast(pl.Float64).to_list()
                noisy = [
                    v + _random.gauss(0, noise_std) if v is not None else None
                    for v in vals
                ]
                work_df = work_df.with_columns(
                    pl.Series(name=col, values=noisy, dtype=pl.Float64)
                )
                report_lines.append(
                    f"  '{col}': Gaussian noise added (std={noise_std})"
                )

        session.df = work_df
        header = f"Anonymized {len(cols)} column(s) using method='{method}':"
        report_lines.append("Use 'undo' to revert.")
        return "\n".join([header] + report_lines)

    except Exception as exc:
        session.undo()
        return friendly_error(exc, "anonymize")
