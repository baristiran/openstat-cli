"""Data validation commands: validate, fuzzyjoin."""

from __future__ import annotations

import re as _re

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ── validate ─────────────────────────────────────────────────────────────────

@command("validate", usage="validate <col> <rule> [<rule> ...]")
def cmd_validate(session: Session, args: str) -> str:
    """Validate a column against one or more rules.

    Rules:
      min=<N>          — all values ≥ N
      max=<N>          — all values ≤ N
      notnull          — no missing values
      unique           — all values distinct
      positive         — all values > 0
      nonneg           — all values ≥ 0
      regex=<pattern>  — all values match regex (string columns)
      oneof=a,b,c      — all values in the allowed set
      between=lo,hi    — all values in [lo, hi]

    Examples:
      validate age min=0 max=120 notnull
      validate gender oneof=male,female,other
      validate email regex=^[^@]+@[^@]+\\.[^@]+$
      validate score between=0,100
    """
    import polars as pl

    ca = CommandArgs(args)
    if len(ca.positional) < 2:
        return "Usage: validate <col> <rule> [<rule> ...]"

    col_name = ca.positional[0]
    rules = ca.positional[1:]

    try:
        df = session.require_data()
        if col_name not in df.columns:
            return f"Column not found: {col_name}"

        col = df[col_name]
        n_total = df.height
        failures: list[str] = []
        passes: list[str] = []

        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

        for rule in rules:
            rule = rule.strip()

            if rule == "notnull":
                n_miss = col.null_count()
                if n_miss > 0:
                    failures.append(f"notnull: {n_miss} missing values")
                else:
                    passes.append("notnull ✓")

            elif rule == "unique":
                n_dup = n_total - col.drop_nulls().n_unique()
                if n_dup > 0:
                    failures.append(f"unique: {n_dup} duplicate values")
                else:
                    passes.append("unique ✓")

            elif rule == "positive":
                if col.dtype not in NUMERIC:
                    failures.append("positive: column is not numeric")
                else:
                    n_bad = col.drop_nulls().filter(col.drop_nulls() <= 0).len()
                    if n_bad:
                        failures.append(f"positive: {n_bad} values ≤ 0")
                    else:
                        passes.append("positive ✓")

            elif rule == "nonneg":
                if col.dtype not in NUMERIC:
                    failures.append("nonneg: column is not numeric")
                else:
                    n_bad = col.drop_nulls().filter(col.drop_nulls() < 0).len()
                    if n_bad:
                        failures.append(f"nonneg: {n_bad} negative values")
                    else:
                        passes.append("nonneg ✓")

            elif rule.startswith("min="):
                val = float(rule[4:])
                if col.dtype not in NUMERIC:
                    failures.append(f"min={val}: column is not numeric")
                else:
                    n_bad = col.drop_nulls().filter(col.drop_nulls() < val).len()
                    if n_bad:
                        failures.append(f"min={val}: {n_bad} values below minimum")
                    else:
                        passes.append(f"min={val} ✓")

            elif rule.startswith("max="):
                val = float(rule[4:])
                if col.dtype not in NUMERIC:
                    failures.append(f"max={val}: column is not numeric")
                else:
                    n_bad = col.drop_nulls().filter(col.drop_nulls() > val).len()
                    if n_bad:
                        failures.append(f"max={val}: {n_bad} values above maximum")
                    else:
                        passes.append(f"max={val} ✓")

            elif rule.startswith("between="):
                parts = rule[8:].split(",")
                if len(parts) != 2:
                    failures.append(f"between: invalid format (use between=lo,hi)")
                    continue
                lo, hi = float(parts[0]), float(parts[1])
                if col.dtype not in NUMERIC:
                    failures.append(f"between={lo},{hi}: column is not numeric")
                else:
                    ser = col.drop_nulls()
                    n_bad = ser.filter((ser < lo) | (ser > hi)).len()
                    if n_bad:
                        failures.append(f"between={lo},{hi}: {n_bad} values out of range")
                    else:
                        passes.append(f"between={lo},{hi} ✓")

            elif rule.startswith("oneof="):
                allowed = set(rule[6:].split(","))
                vals = col.drop_nulls().cast(pl.Utf8).to_list()
                bad = [v for v in vals if v not in allowed]
                if bad:
                    sample = bad[:5]
                    failures.append(f"oneof: {len(bad)} invalid values, e.g. {sample}")
                else:
                    passes.append(f"oneof={rule[6:]} ✓")

            elif rule.startswith("regex="):
                pattern = rule[6:]
                try:
                    compiled = _re.compile(pattern)
                except _re.error as exc:
                    failures.append(f"regex: invalid pattern — {exc}")
                    continue
                vals = col.drop_nulls().cast(pl.Utf8).to_list()
                bad = [v for v in vals if not compiled.search(v)]
                if bad:
                    sample = bad[:3]
                    failures.append(f"regex: {len(bad)} non-matching values, e.g. {sample}")
                else:
                    passes.append(f"regex ✓")

            else:
                failures.append(f"Unknown rule: {rule}")

        lines = [f"Validation: {col_name}  (N={n_total})", "=" * 50]
        for msg in passes:
            lines.append(f"  PASS  {msg}")
        for msg in failures:
            lines.append(f"  FAIL  {msg}")
        lines.append("=" * 50)
        if failures:
            lines.append(f"Result: {len(failures)} check(s) FAILED, {len(passes)} passed")
        else:
            lines.append(f"Result: All {len(passes)} check(s) PASSED")
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "validate")


# ── fuzzyjoin ────────────────────────────────────────────────────────────────

@command("fuzzyjoin", usage="fuzzyjoin <other_file> on(<col>) [--threshold=80] [--method=ratio]")
def cmd_fuzzyjoin(session: Session, args: str) -> str:
    """Fuzzy (approximate) string join with another dataset.

    Matches rows by similarity of a string column, useful for messy data.

    Options:
      on(<col>)           — column to match on (must exist in both datasets)
      --threshold=80      — minimum similarity score (0-100, default 80)
      --method=ratio      — scoring: ratio, partial_ratio, token_sort_ratio

    Examples:
      fuzzyjoin companies.csv on(name) --threshold=85
      fuzzyjoin lookup.parquet on(city) --method=token_sort_ratio
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return (
            "rapidfuzz is required for fuzzyjoin.\n"
            "Install: pip install rapidfuzz"
        )

    import polars as pl
    from openstat.io.loader import load_file

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: fuzzyjoin <other_file> on(<col>) [--threshold=80]"

    other_path = ca.positional[0]
    on_raw = ca.rest_after("on")
    if not on_raw:
        return "Specify join column: on(<col>)"
    on_col = on_raw.strip().strip("()")

    threshold = float(ca.options.get("threshold", 80))
    method_name = ca.options.get("method", "ratio")
    scorer = {
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
    }.get(method_name, fuzz.ratio)

    try:
        df_left = session.require_data()
        df_right = load_file(other_path)

        if on_col not in df_left.columns:
            return f"Column '{on_col}' not in current dataset."
        if on_col not in df_right.columns:
            return f"Column '{on_col}' not in {other_path}."

        left_vals = df_left[on_col].cast(pl.Utf8).to_list()
        right_vals = df_right[on_col].cast(pl.Utf8).to_list()

        # For each left value find best match in right
        best_match = []
        best_score = []
        for lv in left_vals:
            if lv is None:
                best_match.append(None)
                best_score.append(0.0)
                continue
            result = process.extractOne(lv, right_vals, scorer=scorer)
            if result and result[1] >= threshold:
                best_match.append(result[0])
                best_score.append(float(result[1]))
            else:
                best_match.append(None)
                best_score.append(float(result[1]) if result else 0.0)

        df_left = df_left.with_columns([
            pl.Series("_fuzzy_match", best_match),
            pl.Series("_fuzzy_score", best_score),
        ])

        # Join right dataset on the matched value
        df_right_renamed = df_right.rename(
            {c: f"_r_{c}" if c != on_col else c for c in df_right.columns}
        ).rename({on_col: "_fuzzy_match"})

        result = df_left.join(df_right_renamed, on="_fuzzy_match", how="left")
        session.snapshot()
        session.df = result

        n_matched = sum(1 for s in best_score if s >= threshold)
        return (
            f"Fuzzy join complete. {n_matched}/{df_left.height} rows matched "
            f"(threshold={threshold:.0f}). New shape: {session.shape_str}"
        )

    except Exception as e:
        return friendly_error(e, "fuzzyjoin")
