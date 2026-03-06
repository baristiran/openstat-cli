"""Plugin discovery and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points

from openstat.logging_config import get_logger

log = get_logger("plugins")

ENTRY_POINT_GROUP = "openstat_plugin"


@dataclass
class PluginInfo:
    """Metadata for an installed plugin."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    commands: list[str] = field(default_factory=list)


class PluginManager:
    """Discovers and loads OpenStat plugins via entry_points."""

    def __init__(self) -> None:
        self._loaded: dict[str, PluginInfo] = {}
        self._errors: dict[str, str] = {}

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return dict(self._loaded)

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def discover(self) -> list[str]:
        """Discover and load all installed plugins.

        Returns list of successfully loaded plugin names.
        """
        eps = entry_points(group=ENTRY_POINT_GROUP)
        loaded: list[str] = []
        for ep in eps:
            try:
                module = ep.load()
                if hasattr(module, "setup"):
                    info = module.setup()
                    if not isinstance(info, PluginInfo):
                        info = PluginInfo(name=ep.name)
                    self._loaded[ep.name] = info
                else:
                    # Module loaded — commands registered via @command at import
                    self._loaded[ep.name] = PluginInfo(name=ep.name)
                loaded.append(ep.name)
                log.info("Loaded plugin: %s", ep.name)
            except Exception as exc:
                self._errors[ep.name] = str(exc)
                log.warning("Failed to load plugin %s: %s", ep.name, exc)
        return loaded

    def list_plugins(self) -> list[PluginInfo]:
        return list(self._loaded.values())

    def get_info(self, name: str) -> PluginInfo | None:
        return self._loaded.get(name)
