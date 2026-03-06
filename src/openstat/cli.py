"""CLI entry point for OpenStat (Typer-based)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from openstat import __version__

app = typer.Typer(
    name="openstat",
    help="OpenStat — Open-source statistical analysis tool.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"openstat {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging (INFO level).",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Enable debug logging (DEBUG level).",
    ),
) -> None:
    """OpenStat — Open-source statistical analysis tool."""
    from openstat.logging_config import setup_logging
    setup_logging(verbose=verbose, debug=debug)


@app.command()
def repl() -> None:
    """Start the interactive REPL."""
    from openstat.repl import run_repl
    run_repl()


@app.command()
def run(
    script: Path = typer.Argument(..., help="Path to an .ost script file."),
    strict: bool = typer.Option(
        False, "--strict",
        help="Stop on first error and exit with code 1.",
    ),
) -> None:
    """Execute an .ost script file."""
    if not script.exists():
        console.print(f"[red]File not found: {script}[/red]")
        raise typer.Exit(1)
    from openstat.repl import run_script
    run_script(str(script), strict=strict)


@app.command()
def serve(
    port: int = typer.Option(8080, help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", help="Host to bind to."),
) -> None:
    """Start the web-based GUI."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web GUI requires: pip install openstat[web][/red]")
        raise typer.Exit(1)
    from openstat.web.app import app as web_app
    if web_app is None:
        console.print("[red]Web GUI requires: pip install openstat[web][/red]")
        raise typer.Exit(1)
    console.print(f"[bold cyan]OpenStat Web[/bold cyan] starting on http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
