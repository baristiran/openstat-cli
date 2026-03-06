"""Backend management commands: set backend, sql."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from openstat.session import Session
from openstat.commands.base import command, CommandArgs, rich_to_str, friendly_error


@command("set", usage="set seed <N> | set backend polars|duckdb")
def cmd_set(session: Session, args: str) -> str:
    """Change settings: random seed, backend."""
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: set seed <N> | set backend polars|duckdb"

    subcmd = ca.positional[0].lower()

    if subcmd == "seed":
        if len(ca.positional) < 2:
            seed = getattr(session, "_repro_seed", None)
            return f"Current seed: {seed}" if seed is not None else "No seed set."
        try:
            seed = int(ca.positional[1])
        except ValueError:
            return f"Invalid seed: {ca.positional[1]}. Must be an integer."
        import numpy as np
        import random as _random
        np.random.seed(seed)
        _random.seed(seed)
        session._repro_seed = seed  # type: ignore[attr-defined]
        return f"Seed set to {seed}. Reproducible random operations enabled."

    elif subcmd == "backend":
        backend_name = ca.positional[1].lower() if len(ca.positional) > 1 else ""
        if backend_name == "polars":
            session._backend = "polars"
            session._backend_obj = None
            return "Backend set to: polars"
        elif backend_name == "duckdb":
            try:
                from openstat.backends.duckdb_backend import DuckDBBackend
                session._backend_obj = DuckDBBackend()
                session._backend = "duckdb"
                # If data already loaded, register it
                if session.df is not None:
                    session._backend_obj._conn.register("data", session.df.to_pandas())
                    session._backend_obj._table_loaded = True
                return "Backend set to: duckdb"
            except ImportError as e:
                return str(e)
        else:
            return f"Unknown backend: {backend_name}. Use 'polars' or 'duckdb'."
    else:
        return f"Unknown setting: {subcmd}. Available: seed, backend"


@command("sql", usage='sql "SELECT * FROM data WHERE ..."')
def cmd_sql(session: Session, args: str) -> str:
    """Execute SQL query on the loaded dataset (DuckDB backend recommended)."""
    query = args.strip().strip('"\'')
    if not query:
        return 'Usage: sql "SELECT * FROM data WHERE ..."'

    try:
        if session._backend == "duckdb" and session._backend_obj is not None:
            result_df = session._backend_obj.sql(query)
        elif session.df is not None:
            # Use Polars SQL context as fallback
            import polars as pl
            ctx = pl.SQLContext({"data": session.df})
            result_df = ctx.execute(query).collect()
        else:
            return "No data loaded."

        session.snapshot()
        session.df = result_df
        return f"Query returned {session.shape_str}"
    except Exception as e:
        return friendly_error(e, "sql")
