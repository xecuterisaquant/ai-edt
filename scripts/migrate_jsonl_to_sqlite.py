"""One-time migration: signals.jsonl + seen_links.txt → SQLite.

Run this script once to seed the new database with historical data.
After a successful run you may archive (but keep) the original JSONL.

Usage:
    python -m scripts.migrate_jsonl_to_sqlite
    # or with explicit paths:
    python -m scripts.migrate_jsonl_to_sqlite \\
        --jsonl signals/signals.jsonl \\
        --seen  signals/seen_links.txt \\
        --db    signals/signals.db

Exit codes:
    0 — success
    1 — signals.jsonl not found (db still created, seen_links migrated if present)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Apply the same schema as ai_edt/db.py (kept in sync manually)."""
    conn.executescript("""
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
        """)
    conn.commit()


def _migrate_signals(jsonl_path: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert all rows from *jsonl_path* into ``signals``.

    Returns ``(inserted, skipped)`` counts.  Rows are skipped when they
    share a ``(ticker, direction, created_utc)`` triple that already exists in
    the DB — this makes the migration idempotent if re-run.
    """
    if not jsonl_path.exists():
        return 0, 0

    inserted = skipped = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING line {lineno}: invalid JSON ({exc}), skipping.")
                skipped += 1
                continue

            # Accept both old-style and new-style field names.
            ticker = row.get("ticker", "")
            direction = row.get("direction", "")
            created_utc = row.get("timestamp") or row.get("created_utc", "")
            headline = row.get("headline", "")

            if not ticker or direction not in ("LONG", "SHORT") or not created_utc:
                print(f"  WARNING line {lineno}: missing required fields, skipping: {row}")
                skipped += 1
                continue

            existing = conn.execute(
                "SELECT id FROM signals WHERE ticker=? AND direction=? AND created_utc=?",
                (ticker, direction, created_utc),
            ).fetchone()
            if existing:
                skipped += 1
                continue

            conn.execute(
                """INSERT INTO signals
                       (event_id, headline, feed_source, ticker, direction,
                        confidence, order_level, rationale, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.get("event_id", ""),
                    headline,
                    row.get("feed_source", ""),
                    ticker,
                    direction,
                    int(row.get("confidence", 0)),
                    int(row.get("order_level", 2)),
                    row.get("rationale", ""),
                    created_utc,
                ),
            )
            inserted += 1

    conn.commit()
    return inserted, skipped


def _migrate_seen_links(seen_path: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert all URLs from *seen_path* into ``headlines_seen``.

    The historical ``seen_links.txt`` file has no timestamps, so all entries
    are tagged with ``1970-01-01`` — old enough that none will trigger the
    4-hour semantic-dedup window in a live run.

    Returns ``(inserted, skipped)`` counts.
    """
    if not seen_path.exists():
        return 0, 0

    HISTORICAL_TS = "1970-01-01T00:00:00+00:00"
    inserted = skipped = 0
    with seen_path.open(encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if not url:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO headlines_seen (link, seen_at, feed_source) VALUES (?, ?, ?)",
                    (url, HISTORICAL_TS, "migrated"),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    skipped += 1
            except sqlite3.Error as exc:
                print(f"  WARNING: could not insert URL {url!r}: {exc}")
                skipped += 1

    conn.commit()
    return inserted, skipped


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate signals.jsonl and seen_links.txt to SQLite."
    )
    parser.add_argument(
        "--jsonl",
        default="signals/signals.jsonl",
        help="Path to the legacy JSONL signal log (default: signals/signals.jsonl).",
    )
    parser.add_argument(
        "--seen",
        default="signals/seen_links.txt",
        help="Path to the legacy seen-links file (default: signals/seen_links.txt).",
    )
    parser.add_argument(
        "--db",
        default="signals/signals.db",
        help="Destination SQLite database path (default: signals/signals.db).",
    )
    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    seen_path = Path(args.seen)
    db_path = Path(args.db)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _bootstrap_schema(conn)

    # --- signals.jsonl --------------------------------------------------
    if jsonl_path.exists():
        print(f"Migrating signals from: {jsonl_path}")
        ins, skip = _migrate_signals(jsonl_path, conn)
        print(f"  signals: {ins} inserted, {skip} skipped")
    else:
        print(f"  signals.jsonl not found at {jsonl_path} — skipping signal migration.")

    # --- seen_links.txt -------------------------------------------------
    if seen_path.exists():
        print(f"Migrating seen links from: {seen_path}")
        ins, skip = _migrate_seen_links(seen_path, conn)
        print(f"  seen_links: {ins} inserted, {skip} skipped")
    else:
        print(f"  seen_links.txt not found at {seen_path} — skipping URL migration.")

    conn.close()
    print("Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
