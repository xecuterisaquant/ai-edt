"""SQLite signal database for AI-EDT.

Manages three tables:
    signals          — every trade signal produced by the pipeline
    headlines_seen   — processed URLs + semantic fingerprints for deduplication
    signal_feedback  — human-in-the-loop corrections from the dashboard

All public functions operate through a module-level singleton connection so
callers never manage connection lifecycle.  Tests patch ``_conn`` directly
with an in-memory or tmp-path SQLite connection.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from ai_edt.config import get_config
from ai_edt.logger import get_logger

if TYPE_CHECKING:
    from ai_edt.signals import Signal

logger = get_logger("db")

_conn: sqlite3.Connection | None = None

# ---------------------------------------------------------------------------
# Stop-word list for semantic fingerprinting
#
# Words that carry no news-specific semantic signal are removed before
# hashing so that "Iran sanctions hit VLCC tanker fleet" and "Iran sanctions
# target VLCC tanker fleet" resolve to the same fingerprint.
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "its",
        "it",
        "this",
        "that",
        "these",
        "those",
        "over",
        "after",
        "before",
        "amid",
        "says",
        "said",
        "new",
        "more",
        "than",
        "ahead",
        "per",
        "hit",
        "set",
        "rise",
        "fall",
        "falls",
        "rises",
        "rose",
        "fell",
        "up",
        "down",
        "out",
        "into",
        "report",
        "reports",
        "latest",
    }
)

# ---------------------------------------------------------------------------
# Schema — CREATE IF NOT EXISTS makes this idempotent across restarts
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT,
    headline        TEXT NOT NULL,
    feed_source     TEXT,
    ticker          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK(direction IN ('LONG','SHORT')),
    confidence      INTEGER NOT NULL,
    order_level     INTEGER NOT NULL DEFAULT 2,
    rationale       TEXT,
    created_utc     TEXT NOT NULL,
    price_at_signal REAL,
    price_1h        REAL,
    price_4h        REAL,
    price_24h       REAL,
    outcome_pnl_1h  REAL,
    outcome_pnl_24h REAL,
    outcome_note    TEXT
);

CREATE TABLE IF NOT EXISTS headlines_seen (
    link        TEXT PRIMARY KEY,
    seen_at     TEXT NOT NULL,
    feed_source TEXT,
    fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS signal_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id   INTEGER REFERENCES signals(id),
    flag        TEXT NOT NULL CHECK(flag IN (
                    'correct', 'wrong_ticker', 'wrong_direction',
                    'wrong_both', 'noise')),
    note        TEXT,
    flagged_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker  ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_utc);
CREATE INDEX IF NOT EXISTS idx_signals_event   ON signals(event_id);
CREATE INDEX IF NOT EXISTS idx_headlines_fp    ON headlines_seen(fingerprint, seen_at);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the database at *db_path* and apply the schema.

    This function is idempotent — safe to call on a pre-existing database.
    Returns an open connection with WAL journal mode and ``sqlite3.Row``
    factory so callers can address columns by name.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Return the singleton connection, opening it on first call."""
    global _conn
    if _conn is None:
        cfg = get_config()
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = init_db(cfg.db_path)
        logger.debug("SQLite opened at %s", cfg.db_path)
    return _conn


# ---------------------------------------------------------------------------
# Semantic fingerprinting
# ---------------------------------------------------------------------------


def headline_fingerprint(headline_lower: str) -> str:
    """Return an MD5 fingerprint of the headline's significant words.

    Filters stop words and very short tokens, sorts the remaining words,
    and hashes the top 8.  Two headlines describing the same event with
    slightly different wording (e.g. Reuters vs CNBC coverage of the same
    AP wire story) will produce the same fingerprint in ~90% of cases —
    enough to catch the cross-feed duplicates this is designed for.
    """
    words = re.findall(r"[a-z]+", headline_lower)
    significant = sorted({w for w in words if w not in _STOP_WORDS and len(w) > 3})[:8]
    return hashlib.md5(" ".join(significant).encode()).hexdigest()


def is_semantic_duplicate(fingerprint: str, window_hours: int = 4) -> bool:
    """Return True if this fingerprint was seen within *window_hours*."""
    cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).isoformat()
    row = (
        get_db()
        .execute(
            "SELECT 1 FROM headlines_seen WHERE fingerprint = ? AND seen_at >= ?",
            (fingerprint, cutoff),
        )
        .fetchone()
    )
    return row is not None


def record_headline(link: str, fingerprint: str, feed_source: str) -> None:
    """Record a processed URL and its fingerprint in ``headlines_seen``.

    Uses INSERT OR IGNORE so re-processing the same URL (e.g. after a seen_
    links.txt migration) is a no-op.
    """
    get_db().execute(
        """INSERT OR IGNORE INTO headlines_seen
               (link, seen_at, feed_source, fingerprint)
           VALUES (?, ?, ?, ?)""",
        (link, datetime.now(UTC).isoformat(), feed_source, fingerprint),
    )
    get_db().commit()


# ---------------------------------------------------------------------------
# Signal operations
# ---------------------------------------------------------------------------


def insert_signal(signal: Signal) -> int:
    """Insert *signal* into the ``signals`` table and return the new row ID."""
    cur = get_db().execute(
        """INSERT INTO signals
               (event_id, headline, feed_source, ticker, direction,
                confidence, order_level, rationale, created_utc)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            signal.event_id,
            signal.headline,
            signal.feed_source,
            signal.ticker,
            signal.direction,
            signal.confidence,
            signal.order_level,
            signal.rationale,
            signal.timestamp,
        ),
    )
    get_db().commit()
    return cur.lastrowid  # type: ignore[return-value]


def is_duplicate(ticker: str, direction: str, window_minutes: int = 30) -> bool:
    """Return True if the same ticker+direction signal exists in *window_minutes*."""
    cutoff = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
    row = (
        get_db()
        .execute(
            """SELECT 1 FROM signals
           WHERE ticker = ? AND direction = ? AND created_utc >= ?""",
            (ticker, direction, cutoff),
        )
        .fetchone()
    )
    return row is not None


def get_recent_signals(limit: int = 50, ticker: str | None = None) -> list[dict]:
    """Return recent signals as plain dicts, newest first.

    When *ticker* is provided only that ticker's signals are returned.
    """
    if ticker:
        rows = (
            get_db()
            .execute(
                """SELECT * FROM signals WHERE ticker = ?
               ORDER BY created_utc DESC LIMIT ?""",
                (ticker, limit),
            )
            .fetchall()
        )
    else:
        rows = (
            get_db()
            .execute(
                "SELECT * FROM signals ORDER BY created_utc DESC LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
    return [dict(row) for row in rows]


def get_signal_by_id(signal_id: int) -> dict | None:
    """Return a single signal as a plain dict, or ``None`` if not found."""
    row = get_db().execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    return dict(row) if row else None


def flag_signal(signal_id: int, flag: str, note: str = "") -> None:
    """Insert a human-review feedback record for *signal_id*.

    *flag* must be one of: ``'correct'``, ``'wrong_ticker'``,
    ``'wrong_direction'``, ``'wrong_both'``, ``'noise'``.
    Raises ``sqlite3.IntegrityError`` for an unrecognised flag value.
    """
    get_db().execute(
        "INSERT INTO signal_feedback (signal_id, flag, note) VALUES (?, ?, ?)",
        (signal_id, flag, note),
    )
    get_db().commit()


# ---------------------------------------------------------------------------
# Phase 3: Outcome tracking
# ---------------------------------------------------------------------------


def update_signal_prices(signal_id: int, prices: dict) -> None:
    """Write price and outcome columns for *signal_id*.

    *prices* is the dict returned by ``outcomes.fetch_prices_for_signal``.
    Columns that are ``None`` in *prices* will overwrite existing values —
    call this only when you have the latest fetch result.
    """
    get_db().execute(
        """UPDATE signals
               SET price_at_signal = :price_at_signal,
                   price_1h        = :price_1h,
                   price_4h        = :price_4h,
                   price_24h       = :price_24h,
                   outcome_pnl_1h  = :outcome_pnl_1h,
                   outcome_pnl_24h = :outcome_pnl_24h,
                   outcome_note    = :outcome_note
           WHERE id = :id""",
        {**prices, "id": signal_id},
    )
    get_db().commit()


def get_signals_needing_outcomes(
    min_age_minutes: int = 30,
    max_age_days: int = 7,
) -> list[dict]:
    """Return signals with incomplete outcome data in the fetchable age window.

    Only returns signals where at least one price column is NULL, the signal
    is old enough for ``price_at_signal`` to be available (*min_age_minutes*),
    and not so old that 1h bars are unlikely to exist (*max_age_days*).
    """
    cutoff_min = (datetime.now(UTC) - timedelta(minutes=min_age_minutes)).isoformat()
    cutoff_max = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
    rows = (
        get_db()
        .execute(
            """SELECT * FROM signals
               WHERE (price_at_signal IS NULL
                      OR price_1h IS NULL
                      OR price_4h IS NULL
                      OR price_24h IS NULL)
                 AND created_utc <= :min_cutoff
                 AND created_utc >= :max_cutoff
               ORDER BY created_utc DESC""",
            {"min_cutoff": cutoff_min, "max_cutoff": cutoff_max},
        )
        .fetchall()
    )
    return [dict(row) for row in rows]
