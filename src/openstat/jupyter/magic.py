"""IPython magic commands for OpenStat in Jupyter notebooks."""

from __future__ import annotations

try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
    from IPython.display import display, HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from openstat.session import Session


def _dispatch_line(session: Session, line: str) -> str | None:
    """Dispatch a single command line using the REPL dispatcher."""
    from openstat.repl import _dispatch
    return _dispatch(session, line)


def _rich_to_html(text: str) -> str:
    """Convert Rich-styled text to HTML."""
    from rich.console import Console
    console = Console(record=True, width=120)
    console.print(text)
    return console.export_html(inline_styles=True)


if HAS_IPYTHON:
    @magics_class
    class OpenStatMagics(Magics):
        """IPython magics for OpenStat: %ost and %%openstat."""

        def __init__(self, shell):
            super().__init__(shell)
            self.session = Session()

        @line_magic
        def ost(self, line):
            """Run a single OpenStat command: %ost load data.csv"""
            result = _dispatch_line(self.session, line)
            if result and result != "__QUIT__":
                display(HTML(f"<pre>{_rich_to_html(result)}</pre>"))

        @cell_magic
        def openstat(self, line, cell):
            """Run multiple OpenStat commands in a cell."""
            for cmd_line in cell.strip().split('\n'):
                cmd_line = cmd_line.strip()
                if not cmd_line or cmd_line.startswith('#'):
                    continue
                result = _dispatch_line(self.session, cmd_line)
                if result == "__QUIT__":
                    break
                if result:
                    display(HTML(f"<pre>{_rich_to_html(result)}</pre>"))
else:
    class OpenStatMagics:
        """Stub when IPython is not available."""
        pass
