"""Group-by aggregation and rolling window commands."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("groupby", usage="groupby <groupcol> [groupcol2 ...] agg <col>:<func> [...]")
def cmd_groupby(session: Session, args: str) -> str:
    """Group-by aggregation: compute statistics by group.

    Aggregation functions: mean, sum, min, max, std, var, median,
                           count, n, nunique, first, last

    Examples:
      groupby gender agg income:mean age:mean
      groupby country year agg sales:sum profit:mean
      groupby category agg price:min price:max price:mean count:n
      groupby region agg value:median value:std

    The result replaces the current dataset. Use 'undo' to restore.
    """
    import polars as pl

    ca = CommandArgs(args)
    # Split at 'agg' keyword
    agg_raw = ca.rest_after("agg")
    if not agg_raw:
        return "Usage: groupby <col> [col2] agg <col>:<func> [col:func ...]"

    # Everything before 'agg' is group columns
    before_agg = args.split("agg", 1)[0].strip()
    group_cols = before_agg.split()
    if not group_cols:
        return "Specify at least one group column."

    # Parse agg specs: col:func or col:func=alias
    agg_specs = agg_raw.strip().split()
    FUNC_MAP = {
        "mean": pl.Expr.mean,
        "sum": pl.Expr.sum,
        "min": pl.Expr.min,
        "max": pl.Expr.max,
        "std": pl.Expr.std,
        "var": pl.Expr.var,
        "median": pl.Expr.median,
        "count": pl.Expr.count,
        "n": pl.Expr.count,
        "nunique": pl.Expr.n_unique,
        "first": pl.Expr.first,
        "last": pl.Expr.last,
    }

    try:
        df = session.require_data()
        for gc in group_cols:
            if gc not in df.columns:
                return f"Group column not found: {gc}"

        exprs = []
        for spec in agg_specs:
            if ":" not in spec:
                return f"Invalid agg spec '{spec}'. Use col:func format."
            col, func = spec.split(":", 1)
            alias = None
            if "=" in func:
                func, alias = func.split("=", 1)
            func = func.lower()
            if col not in df.columns and func not in ("n", "count"):
                return f"Column not found: {col}"
            if func not in FUNC_MAP:
                avail = ", ".join(FUNC_MAP)
                return f"Unknown function '{func}'. Available: {avail}"

            expr_col = pl.col(col) if col in df.columns else pl.first()
            expr = FUNC_MAP[func](expr_col)
            out_name = alias or f"{col}_{func}"
            exprs.append(expr.alias(out_name))

        session.snapshot()
        session.df = df.group_by(group_cols).agg(exprs).sort(group_cols)
        return f"Group-by complete. Result: {session.shape_str}"

    except Exception as e:
        return friendly_error(e, "groupby")


@command("rolling", usage="rolling <col> <window> <func> [into(<newcol>)]")
def cmd_rolling(session: Session, args: str) -> str:
    """Rolling window statistics on a column.

    Functions: mean, sum, min, max, std, var, median

    Options:
      --center          — centered window (default: trailing)
      --min_periods=N   — minimum observations required

    Examples:
      rolling price 7 mean into(price_7d_avg)
      rolling sales 30 sum into(sales_30d)
      rolling returns 20 std into(vol_20d) --center
      rolling value 5 median
    """
    import polars as pl

    ca = CommandArgs(args)
    if len(ca.positional) < 3:
        return "Usage: rolling <col> <window> <func> [into(<newcol>)]"

    col = ca.positional[0]
    try:
        window = int(ca.positional[1])
    except ValueError:
        return f"Window must be an integer, got: {ca.positional[1]}"
    func = ca.positional[2].lower()
    center = "--center" in args
    min_periods = int(ca.options.get("min_periods", 1))

    into_raw = ca.rest_after("into")
    newcol = into_raw.strip().strip("()") if into_raw else f"{col}_roll{window}_{func}"

    FUNC_MAP = {
        "mean": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "std": "std",
        "var": "var",
        "median": "median",
    }

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"
        if func not in FUNC_MAP:
            return f"Unknown function '{func}'. Available: {', '.join(FUNC_MAP)}"

        roll_kwargs = dict(window_size=window, min_periods=min_periods, center=center)
        roll = pl.col(col).rolling_mean(**roll_kwargs) if func == "mean" else \
               pl.col(col).rolling_sum(**roll_kwargs) if func == "sum" else \
               pl.col(col).rolling_min(**roll_kwargs) if func == "min" else \
               pl.col(col).rolling_max(**roll_kwargs) if func == "max" else \
               pl.col(col).rolling_std(**roll_kwargs) if func == "std" else \
               pl.col(col).rolling_var(**roll_kwargs) if func == "var" else \
               pl.col(col).rolling_median(**roll_kwargs)

        session.snapshot()
        session.df = df.with_columns(roll.alias(newcol))
        n_valid = session.df[newcol].drop_nulls().len()
        return f"Rolling {func}({window}) → '{newcol}': {n_valid}/{df.height} non-null values."

    except Exception as e:
        return friendly_error(e, "rolling")
