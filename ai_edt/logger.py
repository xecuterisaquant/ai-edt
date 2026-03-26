"""Logging setup for AI-EDT.

Provides a get_logger() factory that attaches two handlers to the root
"ai_edt" logger (once, on first call):

  - StreamHandler  — INFO+ to console in a human-readable format
  - RotatingFileHandler — DEBUG+ to ``paths.log_file`` (5 MB, 3 backups)

The log file path is read from ``config/settings.yaml`` so it stays in
sync with the rest of the config system.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_log_path() -> Path:
    """Read the log file path from settings.yaml (or fall back to default)."""
    settings_path = _PROJECT_ROOT / "config" / "settings.yaml"
    try:
        with settings_path.open("r", encoding="utf-8") as f:
            settings: dict[str, Any] = yaml.safe_load(f) or {}
        rel = settings.get("paths", {}).get("log_file", "logs/ai_edt.log")
    except (OSError, yaml.YAMLError):
        rel = "logs/ai_edt.log"
    return _PROJECT_ROOT / rel


_LOG_FILE = _resolve_log_path()

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("ai_edt")
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Guard against duplicate handlers on re-import in interactive sessions
    if not root.handlers:
        root.addHandler(console)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ai_edt namespace."""
    _configure()
    return logging.getLogger(f"ai_edt.{name}")
