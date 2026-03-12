"""Configuration loader for AI-EDT.

Loads config/keywords.yaml, config/feeds.yaml, and config/settings.yaml.
All file paths are resolved relative to the project root so the pipeline
works regardless of the working directory.

Usage:
    from ai_edt.config import get_config
    cfg = get_config()
    print(cfg.reasoning_model)   # "deepseek-r1:8b"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root is the parent of the ai_edt/ package directory.
PROJECT_ROOT = Path(__file__).parent.parent


def _load_yaml(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class _Config:
    """Singleton holder for all runtime configuration."""

    def __init__(self) -> None:
        settings = _load_yaml("config/settings.yaml")
        keywords = _load_yaml("config/keywords.yaml")
        feeds = _load_yaml("config/feeds.yaml")

        # Ollama
        self.ollama_host: str = settings["ollama"]["host"]

        # Models
        self.sieve_model: str = settings["models"]["sieve"]
        self.reasoning_model: str = settings["models"]["reasoning"]

        # Timeouts (seconds)
        self.sieve_timeout: int = settings["timeouts"]["sieve"]
        self.reasoning_timeout: int = settings["timeouts"]["reasoning"]

        # Watcher
        self.poll_interval: int = settings["watcher"]["poll_interval"]

        # Keywords — evaluated in priority order: no_pass > high_alpha > general
        self.no_pass_keywords: list[str] = keywords.get("no_pass", [])
        self.high_alpha_keywords: list[str] = keywords["high_alpha"]
        self.general_keywords: list[str] = keywords["general"]

        # RSS feeds: list of {"name": str, "url": str}
        self.feeds: list[dict[str, str]] = feeds["feeds"]

        # Resolved file paths
        self.knowledge_base_path: Path = PROJECT_ROOT / settings["paths"]["knowledge_base"]
        self.signal_log_path: Path = PROJECT_ROOT / settings["paths"]["signal_log"]
        self.log_file_path: Path = PROJECT_ROOT / settings["paths"]["log_file"]


_instance: _Config | None = None


def get_config() -> _Config:
    """Return the singleton Config, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = _Config()
    return _instance
