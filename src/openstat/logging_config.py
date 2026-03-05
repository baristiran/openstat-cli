"""Logging configuration for OpenStat."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path.home() / ".openstat" / "logs"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(*, verbose: bool = False, debug: bool = False) -> None:
    """Configure logging for OpenStat.

    - Default: WARNING to file only
    - --verbose: INFO to file + stderr
    - --debug: DEBUG to file + stderr
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("openstat")

    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    root.setLevel(logging.DEBUG)  # capture everything, handlers filter

    # File handler — always active, always DEBUG
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(_LOG_DIR / "openstat.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        root.addHandler(fh)
    except OSError:
        pass  # if we can't write logs, don't crash

    # Console handler — only if verbose/debug
    if verbose or debug:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the openstat namespace."""
    return logging.getLogger(f"openstat.{name}")
