"""Logging setup for AI-EDT.

Provides a get_logger() factory that attaches two handlers to the root
"ai_edt" logger (once, on first call):

  - StreamHandler  — INFO+ to console in a human-readable format
  - RotatingFileHandler — DEBUG+ to logs/ai_edt.log (5 MB, 3 backups)

Path is resolved from __file__, so it works regardless of cwd.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_LOG_FILE = _PROJECT_ROOT / "logs" / "ai_edt.log"

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
