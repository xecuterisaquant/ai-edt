"""RSS feed watcher for AI-EDT.

Polls all configured feeds every `poll_interval` seconds. Pre-filters
headlines by keyword before invoking the full 3-stage pipeline, so
the LLM is never called for clearly irrelevant content.

Seen links are persisted to ``signals/seen_links.txt`` so the watcher
survives restarts without reprocessing old headlines.

Phase 1 hardening:
- Exponential backoff per feed on errors (5s → 15min cap)
- Health heartbeat written to ``signals/watcher_health.json`` every cycle
- Optional market-hours gate (US 09:30–16:00 ET, weekdays only)
- Flood warning if a single feed returns too many entries
"""

from __future__ import annotations

import json as _json
import signal as _signal
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser

from ai_edt import db as _db
from ai_edt import pipeline
from ai_edt.config import get_config
from ai_edt.logger import get_logger
from ai_edt.pipeline import stage1_matches

logger = get_logger("watcher")

# Maximum entries kept in seen_links.txt before trimming.
_MAX_SEEN_LINKS = 10_000

# Backoff constants (seconds)
_INITIAL_BACKOFF = 5
_MAX_BACKOFF = 900  # 15 minutes
_BACKOFF_FACTOR = 2

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Seen-links persistence
# ---------------------------------------------------------------------------


def _seen_links_path() -> Path:
    """Return the path to the seen-links file (next to signals.jsonl)."""
    cfg = get_config()
    return cfg.signal_log_path.parent / "seen_links.txt"


def _load_seen_links(path: Path) -> set[str]:
    """Load previously seen URLs from disk."""
    if path.exists():
        return set(path.read_text(encoding="utf-8").splitlines())
    return set()


def _record_link(path: Path, link: str) -> None:
    """Append a single URL to the seen-links file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(link + "\n")


def _trim_seen_links(path: Path, max_entries: int = _MAX_SEEN_LINKS) -> None:
    """Keep only the most recent *max_entries* lines."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_entries:
        return
    trimmed = lines[-max_entries:]
    path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
    logger.info("Trimmed seen_links.txt from %d to %d entries", len(lines), len(trimmed))


# ---------------------------------------------------------------------------
# Health heartbeat
# ---------------------------------------------------------------------------


def _health_path() -> Path:
    """Return the path to the watcher health file."""
    cfg = get_config()
    return cfg.signal_log_path.parent / "watcher_health.json"


def _write_health(
    path: Path,
    feeds_ok: int,
    feeds_err: int,
    headlines_processed: int,
) -> None:
    """Write a JSON heartbeat with poll-cycle stats."""
    payload = {
        "last_poll_utc": datetime.now(UTC).isoformat(),
        "feeds_ok": feeds_ok,
        "feeds_err": feeds_err,
        "headlines_processed": headlines_processed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Market-hours gate
# ---------------------------------------------------------------------------


def _is_market_hours() -> bool:
    """Return True if now is within US equity market hours.

    Market hours: Monday–Friday, 09:30–16:00 Eastern Time.
    Returns True unconditionally on weekends/holidays if you want
    the caller to skip — the caller checks ``cfg.market_hours_only``.
    """
    now_et = datetime.now(_ET)
    # weekday(): Monday=0 … Friday=4
    if now_et.weekday() > 4:
        return False
    t = now_et.time()
    return dt_time(9, 30) <= t < dt_time(16, 0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def start() -> None:
    """Start monitoring RSS feeds. Blocks until Ctrl+C."""
    cfg = get_config()

    sl_path = _seen_links_path()
    processed_links = _load_seen_links(sl_path)
    logger.info("Loaded %d previously seen links from %s", len(processed_links), sl_path)
    _trim_seen_links(sl_path)

    hp = _health_path()

    feed_names = [f["name"] for f in cfg.feeds]
    logger.info(
        "AI-EDT Watcher started | %d feeds: %s",
        len(cfg.feeds),
        ", ".join(feed_names),
    )
    logger.info("Poll interval: %ds | Press Ctrl+C to stop", cfg.poll_interval)
    if cfg.market_hours_only:
        logger.info("Market-hours gate ENABLED — polls only during 09:30-16:00 ET weekdays")

    # Graceful shutdown on Ctrl+C / SIGTERM
    _shutdown = False

    def _handle_shutdown(signum: int, frame: object) -> None:
        nonlocal _shutdown
        _shutdown = True

    _signal.signal(_signal.SIGINT, _handle_shutdown)
    _signal.signal(_signal.SIGTERM, _handle_shutdown)

    # Per-feed exponential backoff state
    feed_backoff: dict[str, float] = {}  # feed_name → current backoff seconds
    feed_backoff_until: dict[str, float] = {}  # feed_name → monotonic deadline

    while not _shutdown:
        # Market-hours gate
        if cfg.market_hours_only and not _is_market_hours():
            logger.debug("Outside market hours — skipping poll cycle")
            if not _shutdown:
                time.sleep(cfg.poll_interval)
            continue

        feeds_ok = 0
        feeds_err = 0
        headlines_processed = 0

        for feed_cfg in cfg.feeds:
            if _shutdown:
                break

            feed_name = feed_cfg["name"]

            # Check if this feed is still in backoff
            if feed_name in feed_backoff_until:
                remaining = feed_backoff_until[feed_name] - time.monotonic()
                if remaining > 0:
                    logger.debug("Feed '%s' in backoff (%.0fs remaining)", feed_name, remaining)
                    feeds_err += 1
                    continue
                # Backoff expired — try again
                del feed_backoff_until[feed_name]

            try:
                feed = feedparser.parse(feed_cfg["url"])

                # Treat bozo (malformed/unreachable) with no entries as an error
                if feed.bozo and not feed.entries:
                    raise RuntimeError(f"Feed returned bozo with 0 entries: {feed.bozo_exception}")

                n_entries = len(feed.entries)

                # Flood warning
                if n_entries > cfg.max_entries_per_feed:
                    logger.warning(
                        "Feed '%s' returned %d entries (threshold %d) — possible flood",
                        feed_name,
                        n_entries,
                        cfg.max_entries_per_feed,
                    )

                for entry in feed.entries:
                    link = getattr(entry, "link", "") or ""
                    if link in processed_links:
                        continue
                    processed_links.add(link)
                    _record_link(sl_path, link)

                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue

                    title_lower = title.lower()

                    # Pre-filter: reuse Stage 1 logic — single source of truth.
                    tier = stage1_matches(title_lower)
                    if tier is None or tier == "no_pass":
                        continue

                    # Semantic dedup: skip if a near-identical headline was
                    # processed in the last 4 hours.  Catches Reuters + CNBC
                    # same-event duplicates that arrive from different URLs.
                    fingerprint = _db.headline_fingerprint(title_lower)
                    if _db.is_semantic_duplicate(fingerprint, window_hours=4):
                        logger.debug("[%s] Semantic duplicate skipped: %s", feed_name, title)
                        continue
                    _db.record_headline(link, fingerprint, feed_name)

                    headlines_processed += 1
                    logger.info("[%s] New headline: %s", feed_name, title)
                    signals = pipeline.analyze(title, feed_source=feed_name)

                    for signal in signals:
                        logger.info(
                            ">>> SIGNAL: %s %s @ %d%% — %s",
                            signal.ticker,
                            signal.direction,
                            signal.confidence,
                            signal.rationale,
                        )

                # Success — reset backoff for this feed
                if feed_name in feed_backoff:
                    logger.info("Feed '%s' recovered — backoff reset", feed_name)
                    del feed_backoff[feed_name]
                feeds_ok += 1

            except Exception as exc:
                # Exponential backoff
                prev = feed_backoff.get(feed_name, _INITIAL_BACKOFF / _BACKOFF_FACTOR)
                new_backoff = min(prev * _BACKOFF_FACTOR, _MAX_BACKOFF)
                feed_backoff[feed_name] = new_backoff
                feed_backoff_until[feed_name] = time.monotonic() + new_backoff
                feeds_err += 1
                logger.warning(
                    "Error reading feed '%s': %s — backing off %.0fs",
                    feed_name,
                    exc,
                    new_backoff,
                )

        # Write health heartbeat after each cycle
        _write_health(hp, feeds_ok, feeds_err, headlines_processed)

        if not _shutdown:
            time.sleep(cfg.poll_interval)

    logger.info("Watcher stopped cleanly.")
