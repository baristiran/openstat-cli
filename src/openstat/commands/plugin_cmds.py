"""Plugin management commands: plugin list, plugin info."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from openstat.session import Session
from openstat.commands.base import command, rich_to_str


# Global plugin manager — initialized at REPL startup
_manager = None


def get_plugin_manager():
    global _manager
    if _manager is None:
        from openstat.plugins.manager import PluginManager
        _manager = PluginManager()
    return _manager


def init_plugins() -> list[str]:
    """Discover and load all plugins. Called at REPL startup."""
    mgr = get_plugin_manager()
    return mgr.discover()


@command("plugin", usage="plugin list | plugin info <name>")
def cmd_plugin(session: Session, args: str) -> str:
    """Manage plugins: list installed plugins or show plugin details."""
    mgr = get_plugin_manager()
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""

    if subcmd == "list":
        plugins = mgr.list_plugins()
        if not plugins:
            return "No plugins installed. Install plugins with: pip install <plugin-package>"

        def render(console: Console) -> None:
            table = Table(title="Installed Plugins")
            table.add_column("Name", style="cyan")
            table.add_column("Version", justify="right")
            table.add_column("Description")
            table.add_column("Commands", style="green")
            for p in plugins:
                table.add_row(
                    p.name, p.version, p.description,
                    ", ".join(p.commands) if p.commands else "—",
                )
            console.print(table)

        errors = mgr.errors
        result = rich_to_str(render)
        if errors:
            result += "\n\nFailed to load:"
            for name, err in errors.items():
                result += f"\n  {name}: {err}"
        return result

    elif subcmd == "info":
        name = parts[1].strip() if len(parts) > 1 else ""
        if not name:
            return "Usage: plugin info <name>"
        info = mgr.get_info(name)
        if info is None:
            return f"Plugin '{name}' not found. Use 'plugin list' to see installed plugins."
        lines = [
            f"Plugin: {info.name}",
            f"Version: {info.version}",
            f"Description: {info.description or '(none)'}",
            f"Commands: {', '.join(info.commands) if info.commands else '(none)'}",
        ]
        return "\n".join(lines)

    else:
        return "Usage: plugin list | plugin info <name>"
