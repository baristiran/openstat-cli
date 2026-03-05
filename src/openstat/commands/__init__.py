"""Command package — import all command modules to register them."""

# Importing these modules triggers the @command decorators,
# which populate the global registry in base.py.
from openstat.commands import data_cmds   # noqa: F401
from openstat.commands import stat_cmds   # noqa: F401
from openstat.commands import plot_cmds   # noqa: F401
from openstat.commands import report_cmds  # noqa: F401

from openstat.commands.base import get_registry

# Public API — the COMMANDS dict used by the REPL
COMMANDS = get_registry()
