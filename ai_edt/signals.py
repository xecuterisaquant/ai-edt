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
from datetime import datetime, timezone
from typing import Optional

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
    direction: str    # "LONG" or "SHORT"
    confidence: int   # 0–100
    rationale: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def __str__(self) -> str:
        return (
            f"{self.ticker} {self.direction} @ {self.confidence}% | {self.rationale}"
        )


def strip_think(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
    return _THINK_RE.sub("", text).strip()


def sieve_says_no(response: str) -> bool:
    """Return True only when the sieve model gives a clear, unambiguous NO.

    Conservative by design: ambiguous responses let the headline through.
    A clear NO requires the word 'NO' to appear without 'YES' also appearing.
    """
    return bool(_NO_RE.search(response)) and not bool(_YES_RE.search(response))


def parse_signal(raw: str, headline: str) -> Optional[Signal]:
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
    """Append a Signal record to the JSONL history file."""
    cfg = get_config()
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
