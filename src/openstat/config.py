"""Configuration management for OpenStat.

Loads settings from ~/.openstat/config.toml (if exists) with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CONFIG_DIR = Path.home() / ".openstat"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"


@dataclass
class Config:
    """OpenStat configuration with defaults."""

    # Data
    output_dir: str = "outputs"
    csv_separator: str = ","
    infer_schema_length: int = 10_000

    # Display
    tabulate_limit: int = 50
    head_default: int = 10

    # Undo
    max_undo_stack: int = 20
    max_undo_memory_mb: int = 500  # adaptive: skip snapshots if exceeds

    # Plotting
    plot_dpi: int = 150
    plot_figsize_w: float = 8.0
    plot_figsize_h: float = 5.0
    plot_style: str = "default"

    # Model
    condition_threshold: int = 30
    min_obs_per_predictor: int = 5
    bootstrap_iterations: int = 1000

    @classmethod
    def load(cls) -> "Config":
        """Load config from TOML file, falling back to defaults."""
        cfg = cls()
        if not _CONFIG_FILE.exists():
            return cfg

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                return cfg  # no TOML parser available, use defaults

        try:
            with open(_CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return cfg  # malformed config, use defaults

        # Flatten sections
        flat: dict[str, object] = {}
        for section_key, section_val in data.items():
            if isinstance(section_val, dict):
                for k, v in section_val.items():
                    flat[f"{section_key}_{k}"] = v
            else:
                flat[section_key] = section_val

        # Apply known keys
        for key in cfg.__dataclass_fields__:
            if key in flat:
                try:
                    setattr(cfg, key, flat[key])
                except (TypeError, ValueError):
                    pass  # ignore invalid values

        return cfg


# Singleton — loaded once at import time
_config: Config | None = None


def get_config() -> Config:
    """Return the global configuration (loads on first call)."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config(override: Config | None = None) -> Config:
    """Reset the global config singleton.

    If *override* is given it becomes the new config; otherwise
    a fresh default ``Config()`` is used.  Returns the new config.

    Intended for tests that need isolation from each other.
    """
    global _config
    _config = override if override is not None else Config()
    return _config
