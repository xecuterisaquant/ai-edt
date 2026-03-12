"""RSS feed watcher for AI-EDT.

Polls all configured feeds every `poll_interval` seconds. Pre-filters
headlines by keyword before invoking the full 3-stage pipeline, so
Ollama is never called for clearly irrelevant content (e.g. sport scores
that slip through a general news feed).

processed_links is an in-memory set — it resets on restart. This is
intentional for the current scope; file-based persistence is tracked as
a future improvement.
"""

from __future__ import annotations

import time

import feedparser

from ai_edt import pipeline
from ai_edt.config import get_config
from ai_edt.logger import get_logger

logger = get_logger("watcher")


def start() -> None:
    """Start monitoring RSS feeds. Blocks until Ctrl+C."""
    cfg = get_config()
    all_keywords = cfg.high_alpha_keywords + cfg.general_keywords
    processed_links: set[str] = set()

    feed_names = [f["name"] for f in cfg.feeds]
    logger.info(
        "AI-EDT Watcher started | %d feeds: %s",
        len(cfg.feeds),
        ", ".join(feed_names),
    )
    logger.info("Poll interval: %ds | Press Ctrl+C to stop", cfg.poll_interval)

    while True:
        for feed_cfg in cfg.feeds:
            try:
                feed = feedparser.parse(feed_cfg["url"])
                for entry in feed.entries:
                    link = getattr(entry, "link", "") or ""
                    if link in processed_links:
                        continue
                    processed_links.add(link)

                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue

                    # Pre-filter: mirrors Stage 1, but avoids any Ollama
                    # overhead for feeds that mix in completely unrelated content.
                    if not any(k in title.lower() for k in all_keywords):
                        continue

                    logger.info("[%s] New headline: %s", feed_cfg["name"], title)
                    signal = pipeline.analyze(title)

                    if signal:
                        logger.info(
                            ">>> SIGNAL: %s %s @ %d%% — %s",
                            signal.ticker,
                            signal.direction,
                            signal.confidence,
                            signal.rationale,
                        )

            except Exception as exc:
                logger.warning(
                    "Error reading feed '%s': %s", feed_cfg["name"], exc
                )

        time.sleep(cfg.poll_interval)
