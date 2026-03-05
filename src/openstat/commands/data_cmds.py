"""Data manipulation commands: load, describe, head, filter, select, derive, dropna, sort, rename, count, tail, merge, pivot, melt, sample, replace, duplicates, unique, encode, recode."""

from __future__ import annotations

import re

import polars as pl
from rich.console import Console
from rich.table import Table

from openstat.session import Session
from openstat.config import get_config
from openstat.io.loader import load_file, save_file
from openstat.dsl.parser import parse_expression, ParseError
from openstat.commands.base import command, CommandArgs, rich_to_str, friendly_error


@command("load", usage="load <path.csv|path.parquet|path.dta>")
def cmd_load(session: Session, args: str) -> str:
    """Load a dataset from CSV, Parquet, or Stata (.dta) file."""
    path = args.strip()
    if not path:
        return "Usage: load <path>"
    session.df = load_file(path)
    session.dataset_path = path
    session.dataset_name = path.split("/")[-1]
    session._undo_stack.clear()
    return f"Loaded {session.shape_str} from {path}"


@command("describe", usage="describe")
def cmd_describe(session: Session, args: str) -> str:
    """Show dataset structure: columns, types, missing values."""
    df = session.require_data()

    def render(console: Console) -> None:
        table = Table(title=f"Dataset: {session.dataset_name or '(unnamed)'}")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Non-null", justify="right")
        table.add_column("Missing", justify="right")
        table.add_column("Unique", justify="right")

        for col_name in df.columns:
            col = df[col_name]
            nulls = col.null_count()
            non_null = df.height - nulls
            unique = col.n_unique()
            table.add_row(col_name, str(col.dtype), str(non_null), str(nulls), str(unique))
        console.print(table)
        console.print(f"Shape: {df.height:,} rows x {df.width} columns")

    return rich_to_str(render)


@command("head", usage="head [N]")
def cmd_head(session: Session, args: str) -> str:
    """Show first N rows (default from config)."""
    df = session.require_data()
    n = get_config().head_default
    if args.strip():
        try:
            n = int(args.strip())
        except ValueError:
            return "Usage: head [N]"

    def render(console: Console) -> None:
        table = Table(title=f"First {min(n, df.height)} rows")
        for col_name in df.columns:
            table.add_column(col_name)
        for row in df.head(n).iter_rows():
            table.add_row(*[str(v) for v in row])
        console.print(table)

    return rich_to_str(render)


@command("tail", usage="tail [N]")
def cmd_tail(session: Session, args: str) -> str:
    """Show last N rows (default 10)."""
    df = session.require_data()
    n = 10
    if args.strip():
        try:
            n = int(args.strip())
        except ValueError:
            return "Usage: tail [N]"

    def render(console: Console) -> None:
        table = Table(title=f"Last {min(n, df.height)} rows")
        for col_name in df.columns:
            table.add_column(col_name)
        for row in df.tail(n).iter_rows():
            table.add_row(*[str(v) for v in row])
        console.print(table)

    return rich_to_str(render)


@command("count", usage="count")
def cmd_count(session: Session, args: str) -> str:
    """Show the number of rows and columns."""
    df = session.require_data()
    return f"{df.height:,} rows x {df.width} columns"


@command("filter", usage="filter <expression>")
def cmd_filter(session: Session, args: str) -> str:
    """Filter rows using an expression. Use 'undo' to revert."""
    df = session.require_data()
    if not args.strip():
        return "Usage: filter <expression>  (e.g. filter age > 30)"
    try:
        expr = parse_expression(args)
        before = df.height
        session.snapshot()
        session.df = df.filter(expr)
        after = session.df.height
        return f"Filtered: {before:,} -> {after:,} rows ({before - after:,} dropped). Use 'undo' to revert."
    except ParseError as e:
        return f"Parse error: {e}"
    except Exception as e:
        return friendly_error(e, "Filter error")


@command("select", usage="select <col1> <col2> ...")
def cmd_select(session: Session, args: str) -> str:
    """Select specific columns. Use 'undo' to revert."""
    df = session.require_data()
    cols = args.split()
    if not cols:
        return "Usage: select <col1> <col2> ..."
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"
    session.snapshot()
    session.df = df.select(cols)
    return f"Selected {len(cols)} columns. Shape: {session.shape_str}"


@command("derive", usage="derive <newcol> = <expression>")
def cmd_derive(session: Session, args: str) -> str:
    """Create a new column from an expression. Use 'undo' to revert."""
    df = session.require_data()
    if "=" not in args:
        return "Usage: derive <newcol> = <expression>"
    name, expr_str = args.split("=", 1)
    name = name.strip()
    if not name:
        return "Usage: derive <newcol> = <expression>"
    try:
        expr = parse_expression(expr_str.strip())
        session.snapshot()
        session.df = df.with_columns(expr.alias(name))
        return f"Created column '{name}'. Shape: {session.shape_str}"
    except ParseError as e:
        return f"Parse error: {e}"
    except Exception as e:
        return friendly_error(e, "Derive error")


@command("dropna", usage="dropna [col1 col2 ...]")
def cmd_dropna(session: Session, args: str) -> str:
    """Drop rows with missing values. Use 'undo' to revert."""
    df = session.require_data()
    cols = args.split() if args.strip() else None
    before = df.height
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}"
    session.snapshot()
    if cols:
        session.df = df.drop_nulls(subset=cols)
    else:
        session.df = df.drop_nulls()
    after = session.df.height
    return f"Dropped nulls: {before:,} -> {after:,} rows ({before - after:,} removed)"


@command("sort", usage="sort <col> [--desc]")
def cmd_sort(session: Session, args: str) -> str:
    """Sort dataset by one or more columns. Use --desc for descending."""
    df = session.require_data()
    ca = CommandArgs(args)
    descending = ca.has_flag("--desc")
    cols = ca.positional
    if not cols:
        return "Usage: sort <col1> [col2 ...] [--desc]"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"
    session.snapshot()
    session.df = df.sort(cols, descending=descending)
    direction = "descending" if descending else "ascending"
    return f"Sorted by {', '.join(cols)} ({direction}). {session.shape_str}"


@command("rename", usage="rename <old> <new>")
def cmd_rename(session: Session, args: str) -> str:
    """Rename a column."""
    df = session.require_data()
    parts = args.split()
    if len(parts) != 2:
        return "Usage: rename <old_name> <new_name>"
    old, new = parts
    if old not in df.columns:
        return f"Column not found: {old}"
    if new in df.columns:
        return f"Column already exists: {new}"
    session.snapshot()
    session.df = df.rename({old: new})
    return f"Renamed '{old}' -> '{new}'"


@command("undo", usage="undo")
def cmd_undo(session: Session, args: str) -> str:
    """Undo the last data-modifying command (filter, select, derive, dropna, sort, rename)."""
    if session.undo():
        return f"Undone. Data restored: {session.shape_str} (undo stack: {session.undo_depth} remaining)"
    return "Nothing to undo."


@command("save", usage="save <path.csv|path.parquet>")
def cmd_save(session: Session, args: str) -> str:
    """Save current dataset to CSV or Parquet."""
    df = session.require_data()
    path = args.strip()
    if not path:
        return "Usage: save <path.csv|path.parquet>"
    try:
        out = save_file(df, path)
        return f"Saved {session.shape_str} to {out}"
    except Exception as e:
        return f"Save error: {e}"


@command("merge", usage="merge <path> on <key> [how=left|right|inner|outer]")
def cmd_merge(session: Session, args: str) -> str:
    """Merge (join) current dataset with another file on a key column."""
    df = session.require_data()
    ca = CommandArgs(args)
    how = ca.get_option("how", "inner")
    on_rest = ca.rest_after("on")
    if not on_rest:
        return "Usage: merge <path> on <key_col> [how=left|right|inner|outer]"

    # file_path is everything before "on" keyword (with options stripped)
    clean = ca.strip_flags_and_options()
    before_on = re.split(r"\bon\b", clean, maxsplit=1)
    file_path = before_on[0].strip()
    key_col = on_rest.split()[0] if on_rest.strip() else ""
    if not file_path or not key_col:
        return "Usage: merge <path> on <key_col> [how=left|right|inner|outer]"

    # Polars uses "full" instead of "outer"
    if how == "outer":
        how = "full"
    valid_how = {"left", "right", "inner", "full", "cross"}
    if how not in valid_how:
        return f"Invalid join type: {how}. Use: {', '.join(sorted(valid_how))}"

    try:
        other = load_file(file_path)
    except Exception as e:
        return f"Cannot load merge file: {e}"

    if key_col not in df.columns:
        return f"Key column '{key_col}' not found in current dataset"
    if key_col not in other.columns:
        return f"Key column '{key_col}' not found in merge file"

    session.snapshot()
    before = df.height
    session.df = df.join(other, on=key_col, how=how, suffix="_right")
    after = session.df.height

    return (
        f"Merged ({how}): {before:,} + {other.height:,} -> {after:,} rows, "
        f"{session.df.width} columns. Use 'undo' to revert."
    )


@command("pivot", usage="pivot <value_col> by <col_col> [over <row_col>] [agg=mean|sum|count|first]")
def cmd_pivot(session: Session, args: str) -> str:
    """Pivot (reshape to wide format). Optional aggregation function."""
    df = session.require_data()
    ca = CommandArgs(args)
    agg_func = ca.get_option("agg", "first")

    AGG_MAP = {
        "first": "first",
        "mean": "mean",
        "sum": "sum",
        "count": "len",
        "min": "min",
        "max": "max",
    }
    if agg_func not in AGG_MAP:
        return f"Unknown aggregation: {agg_func}. Available: {', '.join(AGG_MAP)}"

    # Work with cleaned args (options/flags removed), split on word-boundary "by"
    clean = ca.strip_flags_and_options()
    by_parts = re.split(r"\bby\b", clean, maxsplit=1)
    if len(by_parts) < 2:
        return "Usage: pivot <value_col> by <column_col> [over <index_col>] [agg=mean|sum|count|first]"

    value_col = by_parts[0].strip()
    rest = by_parts[1].strip()

    # Check for "over" keyword with word boundaries
    over_parts = re.split(r"\bover\b", rest, maxsplit=1)
    col_col = over_parts[0].strip()
    index_col = over_parts[1].strip() if len(over_parts) > 1 else None

    if not value_col or not col_col:
        return "Usage: pivot <value_col> by <column_col> [over <index_col>] [agg=mean|sum|count|first]"

    for c in [value_col, col_col] + ([index_col] if index_col else []):
        if c not in df.columns:
            return f"Column not found: {c}"

    session.snapshot()
    try:
        pivot_kwargs: dict = dict(
            on=col_col, values=value_col,
            aggregate_function=AGG_MAP[agg_func],
        )
        if index_col:
            pivot_kwargs["index"] = index_col
        else:
            others = [c for c in df.columns if c not in (value_col, col_col)]
            if not others:
                return "Need at least one column to serve as row index"
            pivot_kwargs["index"] = others
        session.df = df.pivot(**pivot_kwargs)
        return f"Pivoted to wide format: {session.shape_str}. Use 'undo' to revert."
    except Exception as e:
        session.undo()
        return friendly_error(e, "Pivot error")


@command("melt", usage="melt <id_cols>, <value_cols> [var_name=X] [value_name=Y]")
def cmd_melt(session: Session, args: str) -> str:
    """Melt (reshape to long format). Separate id and value cols with a comma."""
    df = session.require_data()
    ca = CommandArgs(args)
    var_name = ca.get_option("var_name", "variable")
    value_name = ca.get_option("value_name", "value")

    clean = ca.strip_flags_and_options()
    if "," not in clean:
        return "Usage: melt <id_col1> <id_col2>, <val_col1> <val_col2> [var_name=X] [value_name=Y]"

    parts = clean.split(",", 1)
    id_cols = parts[0].split()
    val_cols = parts[1].split()

    if not id_cols or not val_cols:
        return "Usage: melt <id_col1> <id_col2>, <val_col1> <val_col2> [var_name=X] [value_name=Y]"

    for c in id_cols + val_cols:
        if c not in df.columns:
            return f"Column not found: {c}"

    session.snapshot()
    try:
        session.df = df.unpivot(
            on=val_cols, index=id_cols,
            variable_name=var_name, value_name=value_name,
        )
        return f"Melted to long format: {session.shape_str}. Use 'undo' to revert."
    except Exception as e:
        session.undo()
        return friendly_error(e, "Melt error")


@command("sample", usage="sample <N|N%>")
def cmd_sample(session: Session, args: str) -> str:
    """Take a random sample: N rows or N% of data."""
    df = session.require_data()
    arg = args.strip()
    if not arg:
        return "Usage: sample <N> or sample <N%>"

    try:
        if arg.endswith("%"):
            pct = float(arg[:-1])
            if not (0 < pct <= 100):
                return "Percentage must be between 0 and 100"
            n = max(1, int(df.height * pct / 100))
        else:
            n = int(arg)
            if n <= 0:
                return "Sample size must be positive"
    except ValueError:
        return "Usage: sample <N> or sample <N%>"

    n = min(n, df.height)
    session.snapshot()
    session.df = df.sample(n=n, shuffle=True)
    return f"Sampled {n:,} rows from {df.height:,}. {session.shape_str}. Use 'undo' to revert."


@command("replace", usage="replace <col> <old_value> <new_value>")
def cmd_replace(session: Session, args: str) -> str:
    """Replace values in a column."""
    df = session.require_data()
    parts = args.split(None, 2)
    if len(parts) < 3:
        return "Usage: replace <col> <old_value> <new_value>"

    col, old_val, new_val = parts[0], parts[1], parts[2]
    if col not in df.columns:
        return f"Column not found: {col}"

    # Strip quotes from values
    old_val = old_val.strip("\"'")
    new_val = new_val.strip("\"'")

    dtype = df[col].dtype
    session.snapshot()

    try:
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            old_v = float(old_val)
            new_v = float(new_val)
            count = df.filter(pl.col(col) == old_v).height
            session.df = df.with_columns(
                pl.when(pl.col(col) == old_v).then(pl.lit(new_v)).otherwise(pl.col(col)).alias(col)
            )
        else:
            count = df.filter(pl.col(col) == old_val).height
            session.df = df.with_columns(
                pl.when(pl.col(col) == old_val).then(pl.lit(new_val)).otherwise(pl.col(col)).alias(col)
            )
        return f"Replaced {count:,} occurrence(s) in '{col}'. Use 'undo' to revert."
    except Exception as e:
        session.undo()
        return friendly_error(e, "Replace error")


@command("duplicates", usage="duplicates [drop] [col1 col2 ...]")
def cmd_duplicates(session: Session, args: str) -> str:
    """Find or drop duplicate rows. Use 'drop' to remove them."""
    df = session.require_data()
    parts = args.split()
    drop = "drop" in parts
    cols = [p for p in parts if p != "drop"]

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {', '.join(missing)}"
        subset = cols
    else:
        subset = None

    if drop:
        before = df.height
        session.snapshot()
        session.df = df.unique(subset=subset, keep="first")
        after = session.df.height
        removed = before - after
        return f"Dropped {removed:,} duplicate(s): {before:,} -> {after:,} rows. Use 'undo' to revert."
    else:
        n_dups = df.height - df.unique(subset=subset, keep="first").height
        if n_dups == 0:
            suffix = f" (on {', '.join(cols)})" if cols else ""
            return f"No duplicates found{suffix}."
        return f"Found {n_dups:,} duplicate row(s). Use 'duplicates drop' to remove them."


@command("unique", usage="unique <col>")
def cmd_unique(session: Session, args: str) -> str:
    """Show unique values of a column."""
    df = session.require_data()
    col = args.strip()
    if not col:
        return "Usage: unique <col>"
    if col not in df.columns:
        return f"Column not found: {col}"

    values = df[col].unique().sort().to_list()
    n = len(values)
    if n > 50:
        shown = [str(v) for v in values[:50]]
        return f"{col}: {n} unique values (showing first 50):\n{', '.join(shown)} ..."
    return f"{col}: {n} unique values:\n{', '.join(str(v) for v in values)}"


@command("encode", usage="encode <col> [as <new_col>]")
def cmd_encode(session: Session, args: str) -> str:
    """Encode a string column as numeric codes (label encoding)."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: encode <col> [as <new_col>]"

    col = ca.positional[0]
    if col not in df.columns:
        return f"Column not found: {col}"

    as_rest = ca.rest_after("as")
    new_col = as_rest.split()[0] if as_rest else col + "_code"

    # Build mapping: sorted unique values → 0, 1, 2, ...
    unique_vals = df[col].unique().sort().to_list()
    mapping = {v: i for i, v in enumerate(unique_vals)}

    session.snapshot()
    session.df = df.with_columns(
        pl.col(col).replace_strict(mapping).cast(pl.Int64).alias(new_col)
    )

    lines = [f"Encoded '{col}' -> '{new_col}' ({len(unique_vals)} levels):"]
    for v, code in list(mapping.items())[:20]:
        lines.append(f"  {code} = {v}")
    if len(mapping) > 20:
        lines.append(f"  ... ({len(mapping) - 20} more)")
    lines.append("Use 'undo' to revert.")
    return "\n".join(lines)


@command("recode", usage='recode <col> "old1"=new1 "old2"=new2 ...')
def cmd_recode(session: Session, args: str) -> str:
    """Recode values in a column using mappings."""
    df = session.require_data()
    if "=" not in args:
        return 'Usage: recode <col> "old1"=new1 "old2"=new2 ...'

    parts = args.split(None, 1)
    if len(parts) < 2:
        return 'Usage: recode <col> "old1"=new1 "old2"=new2 ...'

    col = parts[0]
    if col not in df.columns:
        return f"Column not found: {col}"

    # Parse mapping pairs
    mapping_str = parts[1]
    mapping: dict[str, str] = {}
    import shlex
    try:
        tokens = shlex.split(mapping_str)
    except ValueError:
        tokens = mapping_str.split()

    for token in tokens:
        if "=" not in token:
            return f"Invalid mapping: {token}. Use old=new format."
        old, new = token.split("=", 1)
        old = old.strip("\"'")
        new = new.strip("\"'")
        mapping[old] = new

    if not mapping:
        return "No mappings provided."

    session.snapshot()
    dtype = df[col].dtype
    is_numeric = dtype in (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )

    expr = pl.col(col)
    for old_val, new_val in mapping.items():
        if is_numeric:
            try:
                old_v = float(old_val)
                new_v = float(new_val)
                expr = pl.when(pl.col(col) == old_v).then(pl.lit(new_v)).otherwise(expr)
            except ValueError:
                session.undo()
                return f"Cannot convert '{old_val}' or '{new_val}' to number for numeric column '{col}'"
        else:
            expr = pl.when(pl.col(col) == old_val).then(pl.lit(new_val)).otherwise(expr)

    session.df = df.with_columns(expr.alias(col))
    return f"Recoded {len(mapping)} value(s) in '{col}'. Use 'undo' to revert."


@command("fillna", usage="fillna <col> <strategy>")
def cmd_fillna(session: Session, args: str) -> str:
    """Fill missing values: mean, median, mode, forward, backward, or value=N."""
    df = session.require_data()
    parts = args.split()
    if len(parts) < 2:
        return (
            "Usage: fillna <col> <strategy>\n"
            "Strategies: mean, median, mode, forward, backward, value=N"
        )

    col = parts[0]
    strategy = parts[1]
    if col not in df.columns:
        return f"Column not found: {col}"

    null_count = df[col].null_count()
    if null_count == 0:
        return f"No missing values in '{col}'."

    session.snapshot()

    try:
        if strategy == "mean":
            fill_val = df[col].mean()
            session.df = df.with_columns(pl.col(col).fill_null(fill_val))
        elif strategy == "median":
            fill_val = df[col].median()
            session.df = df.with_columns(pl.col(col).fill_null(fill_val))
        elif strategy == "mode":
            mode_val = df[col].drop_nulls().mode().to_list()
            if not mode_val:
                session.undo()
                return "Cannot compute mode: no non-null values."
            session.df = df.with_columns(pl.col(col).fill_null(mode_val[0]))
        elif strategy == "forward":
            session.df = df.with_columns(pl.col(col).forward_fill())
        elif strategy == "backward":
            session.df = df.with_columns(pl.col(col).backward_fill())
        elif strategy.startswith("value="):
            val_str = strategy.split("=", 1)[1]
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            session.df = df.with_columns(pl.col(col).fill_null(val))
        else:
            session.undo()
            return f"Unknown strategy: {strategy}. Use: mean, median, mode, forward, backward, value=N"

        remaining = session.df[col].null_count()
        filled = null_count - remaining
        return f"Filled {filled:,} null(s) in '{col}' using {strategy}. Use 'undo' to revert."
    except Exception as e:
        session.undo()
        return friendly_error(e, "Fillna error")


@command("cast", usage="cast <col> <type>")
def cmd_cast(session: Session, args: str) -> str:
    """Convert column type: int, float, str, bool."""
    df = session.require_data()
    parts = args.split()
    if len(parts) < 2:
        return "Usage: cast <col> <type>  (types: int, float, str, bool)"

    col = parts[0]
    target_type = parts[1].lower()
    if col not in df.columns:
        return f"Column not found: {col}"

    TYPE_MAP = {
        "int": pl.Int64,
        "float": pl.Float64,
        "str": pl.Utf8,
        "string": pl.Utf8,
        "bool": pl.Boolean,
    }
    pl_type = TYPE_MAP.get(target_type)
    if pl_type is None:
        return f"Unknown type: {target_type}. Available: {', '.join(TYPE_MAP)}"

    session.snapshot()
    try:
        session.df = df.with_columns(pl.col(col).cast(pl_type))
        return f"Cast '{col}' to {target_type}. Use 'undo' to revert."
    except Exception as e:
        session.undo()
        return friendly_error(e, "Cast error")


@command("lag", usage="lag <col> [N] [as <name>]")
def cmd_lag(session: Session, args: str) -> str:
    """Create a lagged variable (shift values down by N rows, default 1)."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: lag <col> [N] [as <name>]"

    col = ca.positional[0]
    if col not in df.columns:
        return f"Column not found: {col}"

    n = 1
    for p in ca.positional[1:]:
        if p.lower() == "as":
            break
        try:
            n = int(p)
        except ValueError:
            pass

    as_rest = ca.rest_after("as")
    new_name = as_rest.split()[0] if as_rest else f"{col}_lag{n}"

    session.snapshot()
    session.df = df.with_columns(pl.col(col).shift(n).alias(new_name))
    return f"Created '{new_name}' (lag {n}). Use 'undo' to revert."


@command("lead", usage="lead <col> [N] [as <name>]")
def cmd_lead(session: Session, args: str) -> str:
    """Create a lead variable (shift values up by N rows, default 1)."""
    df = session.require_data()
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: lead <col> [N] [as <name>]"

    col = ca.positional[0]
    if col not in df.columns:
        return f"Column not found: {col}"

    n = 1
    for p in ca.positional[1:]:
        if p.lower() == "as":
            break
        try:
            n = int(p)
        except ValueError:
            pass

    as_rest = ca.rest_after("as")
    new_name = as_rest.split()[0] if as_rest else f"{col}_lead{n}"

    session.snapshot()
    session.df = df.with_columns(pl.col(col).shift(-n).alias(new_name))
    return f"Created '{new_name}' (lead {n}). Use 'undo' to revert."
