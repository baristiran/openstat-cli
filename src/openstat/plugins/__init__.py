"""Plugin system for OpenStat — discover and load extensions via entry_points."""

from openstat.plugins.manager import PluginManager, PluginInfo

__all__ = ["PluginManager", "PluginInfo"]
