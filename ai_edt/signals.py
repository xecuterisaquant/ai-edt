"""Signal parsing and history logging for AI-EDT.

Parses the 8B model's free-text output into a structured Signal dataclass
and appends it to a JSON Lines file for future backtesting.

DeepSeek-R1 wraps its chain-of-thought in <think>...</think> tags.
These are stripped before parsing so only the final answer is processed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from ai_edt import db
from ai_edt.config import get_config
from ai_edt.logger import get_logger

logger = get_logger("signals")

# Strip DeepSeek-R1 chain-of-thought blocks before parsing the structured answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Word-boundary patterns to avoid false matches ("KNOW" ≠ "NO", "YESTERDAY" ≠ "YES")
_NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)
_YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)


@dataclass
class Signal:
    headline: str
    ticker: str
    direction: str  # "LONG" or "SHORT"
    confidence: int  # 0–100
    rationale: str
    timestamp: str = ""
    feed_source: str = ""  # RSS feed or data source that generated this headline
    event_id: str = ""  # shared across multi-order signals from same headline (Phase 9)
    order_level: int = 2  # 1=first-order named entity, 2=indirect winner, 3=tertiary (Phase 9)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def __str__(self) -> str:
        source = f" [{self.feed_source}]" if self.feed_source else ""
        return f"{self.ticker} {self.direction} @ {self.confidence}%{source} | {self.rationale}"


def strip_think(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
    return _THINK_RE.sub("", text).strip()


def sieve_says_no(response: str) -> bool:
    """Return True only when the sieve model gives a clear, unambiguous NO.

    Conservative by design: ambiguous responses let the headline through.
    A clear NO requires the word 'NO' to appear without 'YES' also appearing.
    """
    return bool(_NO_RE.search(response)) and not bool(_YES_RE.search(response))


def parse_signal(raw: str, headline: str) -> Signal | None:
    """Parse the reasoning engine's raw output into a Signal.

    Returns None and logs a warning if Ticker or Signal direction is missing.
    """
    cleaned = strip_think(raw)

    ticker_m = re.search(r"-?\s*Ticker:\s*(\S+)", cleaned)
    dir_m = re.search(r"-?\s*Signal:\s*(LONG|SHORT)", cleaned, re.IGNORECASE)
    conf_m = re.search(r"-?\s*Confidence:\s*(\d+)", cleaned)
    rat_m = re.search(r"-?\s*Rationale:\s*(.+?)(?:\n-?\s*\w+:|\Z)", cleaned, re.DOTALL)

    if not ticker_m or not dir_m:
        logger.warning(
            "Malformed signal — missing Ticker or Signal field. Raw output (first 500 chars):\n%s",
            raw[:500],
        )
        return None

    return Signal(
        headline=headline,
        ticker=ticker_m.group(1).upper().rstrip(".,;"),
        direction=dir_m.group(1).upper(),
        confidence=int(conf_m.group(1)) if conf_m else 0,
        rationale=rat_m.group(1).strip() if rat_m else "",
    )


def log_signal(signal: Signal) -> None:
    """Persist a Signal to the SQLite database.

    Two gates are applied before writing:
    1. Confidence gate: signals below ``min_confidence`` are dropped.
    2. Deduplication gate: same ticker+direction within a 30-minute window
       is suppressed to prevent multi-feed burst duplicates.

    When ``keep_jsonl_backup`` is enabled in settings.yaml the signal is
    also appended to ``signals.jsonl`` for compatibility.
    """
    cfg = get_config()

    if signal.confidence < cfg.min_confidence:
        logger.debug(
            "Signal below min_confidence (%d < %d), not logged: %s %s",
            signal.confidence,
            cfg.min_confidence,
            signal.ticker,
            signal.direction,
        )
        return

    if db.is_duplicate(signal.ticker, signal.direction, window_minutes=30):
        logger.debug(
            "Duplicate signal suppressed (%s %s within 30-min window)",
            signal.ticker,
            signal.direction,
        )
        return

    db.insert_signal(signal)

    if cfg.keep_jsonl_backup:
        cfg.signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(signal)
        with cfg.signal_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    logger.debug(
        "Signal logged → %s %s @ %d%%",
        signal.ticker,
        signal.direction,
        signal.confidence,
    )


def is_duplicate_signal(new_signal: Signal, window_minutes: int = 30) -> bool:
    """Return True if the same ticker+direction signal exists in the DB within *window_minutes*.

    Delegates to the database for authoritative deduplication that survives
    process restarts (unlike the previous in-memory list approach).
    """
    return db.is_duplicate(new_signal.ticker, new_signal.direction, window_minutes)
