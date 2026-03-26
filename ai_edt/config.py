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
from dotenv import load_dotenv

# Load .env from the project root (silently ignored if the file doesn't exist).
# This runs once at import time, before any os.environ.get() calls.
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require(data: dict[str, Any], dotted_key: str) -> Any:
    """Traverse *data* using a dotted key like ``'ollama.host'``.

    Raises ``KeyError`` with a human-friendly message if any segment
    is missing, so the user knows exactly which YAML key to fix.
    """
    keys = dotted_key.split(".")
    node = data
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            path_so_far = ".".join(keys[: i + 1])
            raise KeyError(f"Missing required setting '{path_so_far}' in config/settings.yaml")
        node = node[key]
    return node


_VALID_PROVIDERS = frozenset({"gemini", "ollama"})


class _Config:
    """Singleton holder for all runtime configuration."""

    def __init__(self) -> None:
        settings = _load_yaml("config/settings.yaml")
        keywords = _load_yaml("config/keywords.yaml")
        feeds = _load_yaml("config/feeds.yaml")

        # Reasoning provider: "gemini" or "ollama"
        self.reasoning_provider: str = settings.get("reasoning_provider", "ollama")
        if self.reasoning_provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Invalid reasoning_provider '{self.reasoning_provider}' in settings.yaml. "
                f"Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}"
            )

        # Ollama
        self.ollama_host: str = _require(settings, "ollama.host")

        # Gemini
        self.gemini_model: str = settings.get("gemini", {}).get("model", "gemini-2.0-flash")

        # Models
        self.sieve_model: str = _require(settings, "models.sieve")
        self.reasoning_model: str = _require(settings, "models.reasoning")

        # Timeouts (seconds)
        self.sieve_timeout: int = _require(settings, "timeouts.sieve")
        self.reasoning_timeout: int = _require(settings, "timeouts.reasoning")

        # Watcher
        self.poll_interval: int = _require(settings, "watcher.poll_interval")
        self.market_hours_only: bool = settings.get("watcher", {}).get("market_hours_only", False)
        self.max_entries_per_feed: int = settings.get("watcher", {}).get("max_entries_per_feed", 50)

        # Signals
        self.min_confidence: int = settings.get("signals", {}).get("min_confidence", 0)
        self.keep_jsonl_backup: bool = settings.get("signals", {}).get("keep_jsonl_backup", False)

        # Keywords — evaluated in priority order: no_pass > high_alpha > general
        self.no_pass_keywords: list[str] = keywords.get("no_pass", [])
        self.high_alpha_keywords: list[str] = keywords["high_alpha"]
        self.general_keywords: list[str] = keywords["general"]

        # RSS feeds: list of {"name": str, "url": str}
        self.feeds: list[dict[str, str]] = feeds["feeds"]

        # Resolved file paths
        self.knowledge_base_path: Path = PROJECT_ROOT / settings["paths"]["knowledge_base"]
        self.macro_context_path: Path = PROJECT_ROOT / settings["paths"].get(
            "macro_context", "data/macro_context.json"
        )
        self.signal_log_path: Path = PROJECT_ROOT / settings["paths"]["signal_log"]
        self.db_path: Path = PROJECT_ROOT / settings["paths"].get("signal_db", "signals/signals.db")
        self.log_file_path: Path = PROJECT_ROOT / settings["paths"]["log_file"]


_instance: _Config | None = None


def get_config() -> _Config:
    """Return the singleton Config, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = _Config()
    return _instance
