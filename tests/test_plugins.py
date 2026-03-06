"""Tests for plugin system (F7)."""

import pytest
from unittest.mock import patch, MagicMock

from openstat.session import Session
from openstat.plugins.manager import PluginManager, PluginInfo
from openstat.commands.plugin_cmds import cmd_plugin


class TestPluginManager:
    """Test plugin discovery and management."""

    def test_empty_discover(self):
        mgr = PluginManager()
        with patch("openstat.plugins.manager.entry_points", return_value=[]):
            loaded = mgr.discover()
        assert loaded == []
        assert mgr.list_plugins() == []

    def test_discover_with_setup(self):
        mgr = PluginManager()
        ep = MagicMock()
        ep.name = "test_plugin"
        module = MagicMock()
        module.setup.return_value = PluginInfo(
            name="test_plugin", version="1.0", description="Test", commands=["test_cmd"],
        )
        ep.load.return_value = module

        with patch("openstat.plugins.manager.entry_points", return_value=[ep]):
            loaded = mgr.discover()

        assert loaded == ["test_plugin"]
        info = mgr.get_info("test_plugin")
        assert info.name == "test_plugin"
        assert info.version == "1.0"

    def test_discover_without_setup(self):
        mgr = PluginManager()
        ep = MagicMock()
        ep.name = "simple_plugin"
        module = MagicMock(spec=[])  # No setup method
        ep.load.return_value = module

        with patch("openstat.plugins.manager.entry_points", return_value=[ep]):
            loaded = mgr.discover()

        assert "simple_plugin" in loaded

    def test_discover_broken_plugin(self):
        mgr = PluginManager()
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("no module")

        with patch("openstat.plugins.manager.entry_points", return_value=[ep]):
            loaded = mgr.discover()

        assert loaded == []
        assert "broken" in mgr.errors


class TestPluginCommand:
    """Test plugin list and info commands."""

    @pytest.fixture
    def session(self, tmp_path):
        return Session(output_dir=tmp_path / "out")

    def test_plugin_list_empty(self, session):
        result = cmd_plugin(session, "list")
        assert "No plugins" in result

    def test_plugin_info_not_found(self, session):
        result = cmd_plugin(session, "info nonexistent")
        assert "not found" in result

    def test_plugin_usage(self, session):
        result = cmd_plugin(session, "")
        assert "Usage" in result
