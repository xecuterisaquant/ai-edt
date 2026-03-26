"""Unit tests for ai_edt/db.py — SQLite signal database.

All tests use a fresh per-test database (tmp_path) so they are
fully isolated from the production database and from each other.

Tests cover:
  - Schema creation and idempotency
  - Insert / retrieve signals and feedback
  - DB-backed duplicate detection (within / outside window)
  - Semantic fingerprinting and deduplication
  - Feed-source and event-id storage
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import ai_edt.config as _config_mod
import ai_edt.db as db_mod
from ai_edt.signals import Signal
from tests.conftest import make_mock_config

# ---------------------------------------------------------------------------
# Fixture: isolated per-test database
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """Provide a fresh, initialized SQLite connection for each test.

    Also patches the config and db singletons so any code under test that
    calls ``get_config()`` or ``get_db()`` receives the test instances.
    """
    cfg = make_mock_config(tmp_path)
    monkeypatch.setattr(_config_mod, "_instance", cfg)
    conn = db_mod.init_db(tmp_path / "signals.db")
    monkeypatch.setattr(db_mod, "_conn", conn)
    return conn


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_signals_table(self, test_db: sqlite3.Connection) -> None:
        row = test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        ).fetchone()
        assert row is not None

    def test_creates_headlines_seen_table(self, test_db: sqlite3.Connection) -> None:
        row = test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='headlines_seen'"
        ).fetchone()
        assert row is not None

    def test_creates_signal_feedback_table(self, test_db: sqlite3.Connection) -> None:
        row = test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_feedback'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling init_db twice on the same file must not raise."""
        path = tmp_path / "idempotent.db"
        db_mod.init_db(path)
        db_mod.init_db(path)  # no-op — CREATE IF NOT EXISTS


# ---------------------------------------------------------------------------
# Insert and retrieve signals
# ---------------------------------------------------------------------------


class TestInsertAndRetrieve:
    def test_insert_returns_rowid(self, test_db: sqlite3.Connection) -> None:
        signal = Signal(
            headline="Test headline",
            ticker="FRO",
            direction="LONG",
            confidence=85,
            rationale="Test rationale",
        )
        row_id = db_mod.insert_signal(signal)
        assert row_id == 1

    def test_get_signal_by_id(self, test_db: sqlite3.Connection) -> None:
        signal = Signal(
            headline="Iran sanctions hit VLCC fleet",
            ticker="FRO",
            direction="LONG",
            confidence=88,
            rationale="Supply reduction boosts rates.",
        )
        row_id = db_mod.insert_signal(signal)
        row = db_mod.get_signal_by_id(row_id)

        assert row is not None
        assert row["ticker"] == "FRO"
        assert row["direction"] == "LONG"
        assert row["confidence"] == 88
        assert row["headline"] == "Iran sanctions hit VLCC fleet"
        assert row["rationale"] == "Supply reduction boosts rates."

    def test_get_signal_by_id_missing(self, test_db: sqlite3.Connection) -> None:
        assert db_mod.get_signal_by_id(999) is None

    def test_get_recent_signals_newest_first(self, test_db: sqlite3.Connection) -> None:
        s1 = Signal(headline="first", ticker="FRO", direction="LONG", confidence=70, rationale="r")
        s2 = Signal(headline="second", ticker="TNK", direction="LONG", confidence=80, rationale="r")
        db_mod.insert_signal(s1)
        db_mod.insert_signal(s2)

        rows = db_mod.get_recent_signals(limit=10)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "TNK"  # newest first

    def test_get_recent_signals_limit_respected(self, test_db: sqlite3.Connection) -> None:
        for i in range(5):
            db_mod.insert_signal(
                Signal(
                    headline=f"h{i}", ticker="FRO", direction="LONG", confidence=70, rationale="r"
                )
            )
        rows = db_mod.get_recent_signals(limit=3)
        assert len(rows) == 3

    def test_get_recent_signals_filter_by_ticker(self, test_db: sqlite3.Connection) -> None:
        db_mod.insert_signal(
            Signal(headline="fro", ticker="FRO", direction="LONG", confidence=70, rationale="r")
        )
        db_mod.insert_signal(
            Signal(headline="tnk", ticker="TNK", direction="LONG", confidence=80, rationale="r")
        )

        rows = db_mod.get_recent_signals(ticker="FRO")
        assert len(rows) == 1
        assert rows[0]["ticker"] == "FRO"

    def test_feed_source_and_event_id_stored(self, test_db: sqlite3.Connection) -> None:
        signal = Signal(
            headline="Test",
            ticker="FRO",
            direction="LONG",
            confidence=80,
            rationale="r",
            feed_source="Reuters Energy",
            event_id="abc-123",
            order_level=2,
        )
        row_id = db_mod.insert_signal(signal)
        row = db_mod.get_signal_by_id(row_id)

        assert row["feed_source"] == "Reuters Energy"
        assert row["event_id"] == "abc-123"
        assert row["order_level"] == 2

    def test_direction_constraint_rejects_invalid(self, test_db: sqlite3.Connection) -> None:
        """DB CHECK constraint must reject directions other than LONG or SHORT."""
        with pytest.raises(sqlite3.IntegrityError):
            test_db.execute(
                """INSERT INTO signals (headline, ticker, direction, confidence,
                   order_level, rationale, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("h", "FRO", "HOLD", 50, 2, "r", datetime.now(UTC).isoformat()),
            )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    def test_duplicate_within_window(self, test_db: sqlite3.Connection) -> None:
        db_mod.insert_signal(
            Signal(headline="test", ticker="FRO", direction="LONG", confidence=85, rationale="r")
        )
        assert db_mod.is_duplicate("FRO", "LONG", window_minutes=30) is True

    def test_outside_window_not_duplicate(self, test_db: sqlite3.Connection) -> None:
        old_ts = (datetime.now(UTC) - timedelta(minutes=60)).isoformat()
        db_mod.insert_signal(
            Signal(
                headline="old",
                ticker="FRO",
                direction="LONG",
                confidence=85,
                rationale="r",
                timestamp=old_ts,
            )
        )
        assert db_mod.is_duplicate("FRO", "LONG", window_minutes=30) is False

    def test_different_direction_not_duplicate(self, test_db: sqlite3.Connection) -> None:
        db_mod.insert_signal(
            Signal(headline="test", ticker="FRO", direction="LONG", confidence=85, rationale="r")
        )
        assert db_mod.is_duplicate("FRO", "SHORT", window_minutes=30) is False

    def test_different_ticker_not_duplicate(self, test_db: sqlite3.Connection) -> None:
        db_mod.insert_signal(
            Signal(headline="test", ticker="FRO", direction="LONG", confidence=85, rationale="r")
        )
        assert db_mod.is_duplicate("TNK", "LONG", window_minutes=30) is False

    def test_empty_db_not_duplicate(self, test_db: sqlite3.Connection) -> None:
        assert db_mod.is_duplicate("FRO", "LONG", window_minutes=30) is False


# ---------------------------------------------------------------------------
# Semantic fingerprinting and deduplication
# ---------------------------------------------------------------------------


class TestSemanticDedup:
    def test_record_then_detect_duplicate(self, test_db: sqlite3.Connection) -> None:
        fp = db_mod.headline_fingerprint("iran sanctions vlcc tanker fleet")
        assert db_mod.is_semantic_duplicate(fp, window_hours=4) is False

        db_mod.record_headline("http://reuters.com/1", fp, "Reuters")
        assert db_mod.is_semantic_duplicate(fp, window_hours=4) is True

    def test_insert_or_ignore_same_link(self, test_db: sqlite3.Connection) -> None:
        """Re-recording the same URL must not raise (INSERT OR IGNORE)."""
        fp = db_mod.headline_fingerprint("iran sanctions vlcc tanker fleet")
        db_mod.record_headline("http://reuters.com/1", fp, "Reuters")
        db_mod.record_headline("http://reuters.com/1", fp, "Reuters")  # no-op

    def test_fingerprint_stable_across_stop_words(self) -> None:
        """Adding/removing common stop words must not change the fingerprint."""
        fp1 = db_mod.headline_fingerprint("iran sanctions vlcc tanker fleet")
        fp2 = db_mod.headline_fingerprint("iran sanctions hit the vlcc tanker fleet")
        assert fp1 == fp2

    def test_fingerprint_differs_for_unrelated_headlines(self) -> None:
        fp1 = db_mod.headline_fingerprint("iran sanctions vlcc tanker fleet")
        fp2 = db_mod.headline_fingerprint("venezuela crude production opec decision")
        assert fp1 != fp2

    def test_outside_window_not_semantic_duplicate(self, test_db: sqlite3.Connection) -> None:
        fp = db_mod.headline_fingerprint("some energy market headline event")
        old_ts = (datetime.now(UTC) - timedelta(hours=8)).isoformat()
        test_db.execute(
            "INSERT INTO headlines_seen (link, seen_at, feed_source, fingerprint) VALUES (?, ?, ?, ?)",
            ("http://old.com/1", old_ts, "Reuters", fp),
        )
        test_db.commit()
        assert db_mod.is_semantic_duplicate(fp, window_hours=4) is False


# ---------------------------------------------------------------------------
# Signal feedback
# ---------------------------------------------------------------------------


class TestFlagSignal:
    def test_flag_inserts_feedback_record(self, test_db: sqlite3.Connection) -> None:
        signal = Signal(
            headline="test", ticker="FRO", direction="LONG", confidence=85, rationale="r"
        )
        row_id = db_mod.insert_signal(signal)
        db_mod.flag_signal(row_id, "correct", note="verified by price data")

        feedback = test_db.execute(
            "SELECT * FROM signal_feedback WHERE signal_id = ?", (row_id,)
        ).fetchone()
        assert feedback is not None
        assert feedback["flag"] == "correct"
        assert feedback["note"] == "verified by price data"

    def test_invalid_flag_raises_integrity_error(self, test_db: sqlite3.Connection) -> None:
        signal = Signal(
            headline="test", ticker="FRO", direction="LONG", confidence=85, rationale="r"
        )
        row_id = db_mod.insert_signal(signal)

        with pytest.raises(sqlite3.IntegrityError):
            db_mod.flag_signal(row_id, "made_up_flag")
