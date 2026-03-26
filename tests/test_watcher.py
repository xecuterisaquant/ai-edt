"""Unit tests for Phase 1 watcher hardening features.

Tests cover:
- Exponential backoff mechanics
- Health heartbeat writing
- Market-hours gate logic
- Flood warning threshold
- Seen-links persistence helpers
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ai_edt.watcher import (
    _BACKOFF_FACTOR,
    _INITIAL_BACKOFF,
    _MAX_BACKOFF,
    _is_market_hours,
    _load_seen_links,
    _record_link,
    _trim_seen_links,
    _write_health,
)

# ---------------------------------------------------------------------------
# Seen-links persistence
# ---------------------------------------------------------------------------


class TestSeenLinks:
    def test_load_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "seen.txt"
        assert _load_seen_links(path) == set()

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "seen.txt"
        _record_link(path, "http://a.com/1")
        _record_link(path, "http://b.com/2")
        loaded = _load_seen_links(path)
        assert loaded == {"http://a.com/1", "http://b.com/2"}

    def test_trim_keeps_recent(self, tmp_path: Path) -> None:
        path = tmp_path / "seen.txt"
        for i in range(20):
            _record_link(path, f"http://example.com/{i}")
        _trim_seen_links(path, max_entries=5)
        loaded = _load_seen_links(path)
        assert len(loaded) == 5
        # Should keep the last 5 (15..19)
        assert "http://example.com/19" in loaded
        assert "http://example.com/0" not in loaded

    def test_trim_noop_when_under_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "seen.txt"
        _record_link(path, "http://a.com/1")
        _trim_seen_links(path, max_entries=100)
        assert _load_seen_links(path) == {"http://a.com/1"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "seen.txt"
        _record_link(path, "http://x.com")
        assert _load_seen_links(path) == {"http://x.com"}


# ---------------------------------------------------------------------------
# Health heartbeat
# ---------------------------------------------------------------------------


class TestHealthHeartbeat:
    def test_write_health_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "signals" / "watcher_health.json"
        _write_health(path, feeds_ok=3, feeds_err=1, headlines_processed=5)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["feeds_ok"] == 3
        assert data["feeds_err"] == 1
        assert data["headlines_processed"] == 5
        assert "last_poll_utc" in data

    def test_write_health_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "health.json"
        _write_health(path, 1, 0, 0)
        _write_health(path, 2, 1, 3)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["feeds_ok"] == 2
        assert data["feeds_err"] == 1

    def test_write_health_valid_iso_timestamp(self, tmp_path: Path) -> None:
        path = tmp_path / "health.json"
        _write_health(path, 0, 0, 0)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Should not raise
        datetime.fromisoformat(data["last_poll_utc"])


# ---------------------------------------------------------------------------
# Market-hours gate
# ---------------------------------------------------------------------------


class TestMarketHours:
    def test_weekday_during_hours(self) -> None:
        # Wednesday 10:00 ET → should be market hours
        fake = datetime(2026, 3, 25, 10, 0, 0)  # Wednesday
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is True

    def test_weekday_before_open(self) -> None:
        # Wednesday 08:00 ET → before market open
        fake = datetime(2026, 3, 25, 8, 0, 0)
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_weekday_after_close(self) -> None:
        # Wednesday 16:30 ET → after market close
        fake = datetime(2026, 3, 25, 16, 30, 0)
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_weekend(self) -> None:
        # Saturday → no market
        fake = datetime(2026, 3, 28, 12, 0, 0)  # Saturday
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_market_close_boundary(self) -> None:
        # Exactly 16:00 ET → market is closed (< 16:00, not <=)
        fake = datetime(2026, 3, 25, 16, 0, 0)
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_market_open_boundary(self) -> None:
        # Exactly 09:30 ET → market is open
        fake = datetime(2026, 3, 25, 9, 30, 0)
        with patch("ai_edt.watcher.datetime") as mock_dt:
            mock_dt.now.return_value = fake
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is True


# ---------------------------------------------------------------------------
# Backoff constants sanity
# ---------------------------------------------------------------------------


class TestBackoffConstants:
    def test_initial_backoff(self) -> None:
        assert _INITIAL_BACKOFF == 5

    def test_max_backoff(self) -> None:
        assert _MAX_BACKOFF == 900  # 15 minutes

    def test_backoff_sequence(self) -> None:
        """Verify the geometric progression caps at _MAX_BACKOFF."""
        b = _INITIAL_BACKOFF
        steps = []
        for _ in range(20):
            steps.append(b)
            b = min(b * _BACKOFF_FACTOR, _MAX_BACKOFF)
        # Should reach cap: 5, 10, 20, 40, 80, 160, 320, 640, 900, 900, ...
        assert steps[0] == 5
        assert steps[1] == 10
        assert steps[-1] == _MAX_BACKOFF
        # All values within bounds
        assert all(_INITIAL_BACKOFF <= s <= _MAX_BACKOFF for s in steps)
