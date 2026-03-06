"""TUI dashboard command: dashboard (requires textual)."""

from __future__ import annotations

from openstat.commands.base import command
from openstat.session import Session


@command("dashboard", usage="dashboard")
def cmd_dashboard(session: Session, args: str) -> str:
    """Launch an interactive TUI dashboard (requires: pip install textual).

    Shows dataset overview, variable list, model results, and recent history
    in a full-screen terminal UI. Press Q or Ctrl+C to exit.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import (
            DataTable, Footer, Header, Label, RichLog, TabbedContent, TabPane,
        )
        from textual.binding import Binding
    except ImportError:
        return (
            "textual is required for the dashboard.\n"
            "Install: pip install textual"
        )

    import polars as pl

    # ── Snapshot data so the TUI doesn't need the live session ──────────────
    dataset_name = session.dataset_name or "(no dataset)"
    shape_str = session.shape_str
    df = session.df
    results = list(session.results)
    history = list(session.history[-50:])  # last 50 commands

    # Build column summary
    col_rows: list[tuple[str, ...]] = []
    if df is not None:
        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        for c in df.columns:
            dtype = str(df[c].dtype)
            n_miss = str(df[c].null_count())
            if df[c].dtype in NUMERIC:
                col_data = df[c].drop_nulls()
                if col_data.len() > 0:
                    mean_str = f"{col_data.mean():.3f}"
                    sd_str = f"{col_data.std():.3f}" if col_data.len() > 1 else "—"
                else:
                    mean_str = sd_str = "—"
            else:
                mean_str = sd_str = "—"
            col_rows.append((c, dtype, n_miss, mean_str, sd_str))

    class OpenStatDashboard(App):
        TITLE = f"OpenStat Dashboard — {dataset_name}"
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("ctrl+c", "quit", "Quit"),
        ]
        CSS = """
        Screen { background: #1a1a2e; }
        Header { background: #16213e; color: #e94560; }
        Footer { background: #16213e; color: #a8b2d8; }
        TabbedContent { background: #1a1a2e; }
        TabPane { padding: 1 2; }
        DataTable { height: auto; }
        Label { color: #cdd6f4; margin: 0 0 1 0; }
        RichLog { height: 30; border: solid #313244; background: #181825; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            with TabbedContent():
                with TabPane("Overview", id="overview"):
                    yield Label(f"[bold]Dataset:[/bold] {dataset_name}  |  [bold]Shape:[/bold] {shape_str}")
                    yield Label(
                        f"[bold]Models fitted:[/bold] {len(results)}  |  "
                        f"[bold]Commands run:[/bold] {len(session.history)}"
                    )

                with TabPane("Variables", id="variables"):
                    tbl = DataTable(zebra_stripes=True)
                    tbl.add_columns("Variable", "Type", "Missing", "Mean", "SD")
                    for row in col_rows:
                        tbl.add_row(*row)
                    yield tbl

                with TabPane("Models", id="models"):
                    if results:
                        log = RichLog(markup=True, highlight=True)
                        for mr in results:
                            log.write(f"[bold cyan]{mr.name}: {mr.formula}[/bold cyan]")
                            log.write(mr.table)
                            log.write("")
                        yield log
                    else:
                        yield Label("No models fitted yet.")

                with TabPane("History", id="history"):
                    log = RichLog(markup=False, highlight=False)
                    for cmd_line in history:
                        log.write(f". {cmd_line}")
                    yield log

            yield Footer()

    app = OpenStatDashboard()
    app.run()
    return "Dashboard closed."
