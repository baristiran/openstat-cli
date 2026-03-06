"""reshape, collapse, encode, decode commands."""

from __future__ import annotations

import re

import polars as pl

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


# ── reshape ────────────────────────────────────────────────────────────────

@command("reshape", usage="reshape wide|long varlist, i(id) j(timevar) [stub(prefix)]")
def cmd_reshape(session: Session, args: str) -> str:
    """Reshape data between wide and long format (Stata-style).

    Wide→Long: reshape long prefix, i(id) j(time)
    Long→Wide: reshape wide varlist, i(id) j(timevar)
    """
    df = session.require_data()
    positional, opts = _stata_opts(args)

    if len(positional) < 2:
        return (
            "Usage:\n"
            "  reshape long stubname, i(idvar) j(timevar)\n"
            "  reshape wide varlist, i(idvar) j(timevar)"
        )

    direction = positional[0].lower()
    id_var = opts.get("i")
    j_var = opts.get("j")
    if not id_var or not j_var:
        return "Specify: i(idvar) j(timevar)"

    session.snapshot()

    if direction == "long":
        # wide → long: column prefix → stub
        stub = positional[1] if len(positional) > 1 else ""
        value_vars = [c for c in df.columns if c.startswith(stub) and c != id_var]
        if not value_vars:
            return f"No columns found with prefix '{stub}'"
        try:
            long_df = df.unpivot(
                on=value_vars,
                index=[id_var],
                variable_name=j_var,
                value_name=stub or "value",
            )
            # Extract numeric suffix from variable name
            long_df = long_df.with_columns(
                pl.col(j_var).str.replace(stub, "", literal=True).alias(j_var)
            )
            session.df = long_df
            return f"Reshaped wide→long: {df.shape} → {long_df.shape}. {j_var} = {long_df[j_var].unique().to_list()}"
        except Exception as exc:
            return f"reshape long error: {exc}"

    elif direction == "wide":
        # long → wide
        var_list = [c for c in positional[1:] if c in df.columns]
        if not var_list:
            return "No valid value variables found."
        try:
            wide_df = df.pivot(
                on=j_var,
                index=id_var,
                values=var_list[0] if len(var_list) == 1 else var_list,
                aggregate_function="first",
            )
            session.df = wide_df
            return f"Reshaped long→wide: {df.shape} → {wide_df.shape}"
        except Exception as exc:
            return f"reshape wide error: {exc}"

    else:
        return f"Unknown reshape direction: {direction}. Use 'wide' or 'long'."


# ── collapse ───────────────────────────────────────────────────────────────

_COLLAPSE_FUNS = {
    "mean":   lambda c: pl.col(c).mean(),
    "sum":    lambda c: pl.col(c).sum(),
    "count":  lambda c: pl.col(c).count(),
    "min":    lambda c: pl.col(c).min(),
    "max":    lambda c: pl.col(c).max(),
    "median": lambda c: pl.col(c).median(),
    "std":    lambda c: pl.col(c).std(),
    "var":    lambda c: pl.col(c).var(),
    "first":  lambda c: pl.col(c).first(),
    "last":   lambda c: pl.col(c).last(),
}


@command("collapse", usage="collapse (stat) varlist [, by(groupvars)]")
def cmd_collapse(session: Session, args: str) -> str:
    """Collapse dataset to group-level aggregates (replaces df).

    Examples:
      collapse (mean) income age, by(region)
      collapse (sum) sales, by(year region)
    """
    df = session.require_data()
    session.snapshot()

    # Parse (stat) from args
    stat_m = re.search(r'\((\w+)\)', args)
    stat = stat_m.group(1).lower() if stat_m else "mean"
    if stat not in _COLLAPSE_FUNS:
        return f"Unknown statistic: {stat}. Use: {', '.join(_COLLAPSE_FUNS)}"

    # _stata_opts extracts key(value) pairs (including by(...)) from full args
    positional_raw, opts = _stata_opts(args)
    # positional_raw may include "(mean)" token — filter it out
    value_vars = [c for c in positional_raw if c in df.columns]
    by_raw = opts.get("by", "")
    by_vars = [c.strip() for c in by_raw.split() if c.strip() in df.columns]

    if not value_vars:
        return "No valid value variables found."

    agg_fn = _COLLAPSE_FUNS[stat]
    agg_exprs = [agg_fn(c).alias(c) for c in value_vars]

    try:
        if by_vars:
            result = df.group_by(by_vars).agg(agg_exprs).sort(by_vars)
        else:
            result = df.select(agg_exprs)

        session.df = result
        return (
            f"Collapsed to {result.shape[0]} rows × {result.shape[1]} cols "
            f"using {stat}({', '.join(value_vars)})"
            + (f" by {', '.join(by_vars)}" if by_vars else "")
        )
    except Exception as exc:
        return f"collapse error: {exc}"


# ── encode ─────────────────────────────────────────────────────────────────

@command("encode", usage="encode varname [, gen(newvar)]")
def cmd_encode(session: Session, args: str) -> str:
    """Encode a string/categorical column to integer codes (0-based)."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if not positional:
        return "Usage: encode varname [, gen(newvar)]"

    var = positional[0]
    if var not in df.columns:
        return f"Column '{var}' not found."

    new_var = opts.get("gen", var + "_encoded")
    session.snapshot()

    try:
        # Map unique sorted values to integers
        unique_vals = sorted(df[var].drop_nulls().unique().to_list(), key=str)
        val_map = {v: i for i, v in enumerate(unique_vals)}
        encoded = df[var].map_elements(
            lambda x: val_map.get(x), return_dtype=pl.Int64
        )
        session.df = df.with_columns(encoded.alias(new_var))
        mapping_str = "\n".join(f"  {i} = {v}" for i, v in enumerate(unique_vals[:20]))
        if len(unique_vals) > 20:
            mapping_str += f"\n  ... ({len(unique_vals)} total)"
        return f"Encoded '{var}' → '{new_var}' ({len(unique_vals)} categories)\n{mapping_str}"
    except Exception as exc:
        return f"encode error: {exc}"


# ── decode ─────────────────────────────────────────────────────────────────

@command("decode", usage="decode encodedvar origvar [, gen(newvar)]")
def cmd_decode(session: Session, args: str) -> str:
    """Decode integer codes back to string labels.

    Requires a reference (original) string column with the same row order.
    decode encoded_col orig_col [, gen(newvar)]
    """
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: decode encoded_col orig_col [, gen(newvar)]"

    enc_var = positional[0]
    orig_var = positional[1]
    new_var = opts.get("gen", enc_var + "_decoded")

    for v in (enc_var, orig_var):
        if v not in df.columns:
            return f"Column '{v}' not found."

    session.snapshot()
    try:
        # Build map from int codes → original labels using orig_var
        code_col = df[enc_var].cast(pl.Int64)
        label_col = df[orig_var].cast(pl.Utf8)
        pairs = list(zip(code_col.to_list(), label_col.to_list()))
        code_map = {c: l for c, l in pairs if c is not None and l is not None}

        decoded = df[enc_var].map_elements(
            lambda x: code_map.get(x, None), return_dtype=pl.Utf8
        )
        session.df = df.with_columns(decoded.alias(new_var))
        return f"Decoded '{enc_var}' → '{new_var}' ({len(code_map)} unique codes)"
    except Exception as exc:
        return f"decode error: {exc}"
