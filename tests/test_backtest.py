"""Tests for scripts/backtest.py — Phase 6 backtesting engine."""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow direct import of the backtest script as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_edt.db as _db_mod
from ai_edt.signals import Signal
from scripts.backtest import _HeadlineRow, load_csv, main, performance_report, replay

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    ticker: str = "FRO",
    direction: str = "LONG",
    confidence: int = 85,
    est_cost: float = 0.001,
    headline: str = "Test headline",
    ts: str | None = None,
) -> Signal:
    if ts is None:
        ts = datetime.now(UTC).isoformat()
    return Signal(
        headline=headline,
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        rationale="Test rationale.",
        timestamp=ts,
        feed_source="backtest",
        event_id="test-event-id",
        order_level=1,
        est_cost_usd=est_cost,
    )


def _write_csv(tmp_path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    p = tmp_path / "headlines.csv"
    if fieldnames is None:
        fieldnames = ["datetime", "headline", "feed_source"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_parses_valid_rows(self, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "Test headline one", "feed_source": "reuters"},
            {"datetime": "2026-01-02T08:00:00Z", "headline": "Test headline two", "feed_source": "bloomberg"},
        ])
        rows = load_csv(p)
        assert len(rows) == 2
        assert rows[0].headline == "Test headline one"
        assert rows[0].feed_source == "reuters"

    def test_skips_blank_headlines(self, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "", "feed_source": "reuters"},
            {"datetime": "2026-01-01T11:00:00Z", "headline": "Real headline", "feed_source": "reuters"},
        ])
        rows = load_csv(p)
        assert len(rows) == 1
        assert rows[0].headline == "Real headline"

    def test_skips_unparseable_datetime(self, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "not-a-date", "headline": "Should be skipped", "feed_source": "reuters"},
            {"datetime": "2026-01-02T10:00:00Z", "headline": "Good headline", "feed_source": "reuters"},
        ])
        rows = load_csv(p)
        assert len(rows) == 1
        assert rows[0].headline == "Good headline"

    def test_defaults_feed_source_when_blank(self, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "No source", "feed_source": ""},
        ])
        rows = load_csv(p)
        assert rows[0].feed_source == "backtest"

    def test_raises_if_missing_required_columns(self, tmp_path):
        p = _write_csv(tmp_path, [{"text": "oops", "ts": "2026-01-01"}], fieldnames=["text", "ts"])
        with pytest.raises(ValueError, match="datetime"):
            load_csv(p)

    def test_raises_on_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_csv(p)

    def test_naive_datetime_assumed_utc(self, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-03-10T14:32:00", "headline": "Naive ts headline", "feed_source": "x"},
        ])
        rows = load_csv(p)
        assert rows[0].dt.endswith("+00:00")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_dry_run_skips_insert(self, mock_config, tmp_path):
        """In dry-run mode signals must NOT be inserted into the DB."""
        rows = [_HeadlineRow(dt="2026-01-01T10:00:00+00:00", headline="Oil prices surge on OPEC cut", feed_source="test")]
        sig = _make_signal()

        with patch("scripts.backtest.pipeline.stage1_matches", return_value="general"), \
             patch("scripts.backtest.pipeline.analyze", return_value=[sig]):
            summary = replay(rows, dry_run=True, max_cost_usd=10.0, skip_outcomes=True)

        assert summary["signals_generated"] == 1
        # DB should be empty — no insert happened
        recent = _db_mod.get_recent_signals(limit=10)
        assert len(recent) == 0

    def test_non_dry_run_inserts_signals(self, mock_config, tmp_path):
        """Without --dry-run, each signal from analyze() is inserted into DB."""
        rows = [_HeadlineRow(dt="2026-01-01T10:00:00+00:00", headline="Tanker sanctioned by US", feed_source="test")]
        sig = _make_signal()

        with patch("scripts.backtest.pipeline.stage1_matches", return_value="vip"), \
             patch("scripts.backtest.pipeline.analyze", return_value=[sig]), \
             patch("scripts.backtest.fetch_prices_for_signal", return_value={}):
            summary = replay(rows, dry_run=False, max_cost_usd=10.0, skip_outcomes=False)

        assert summary["signals_generated"] == 1
        recent = _db_mod.get_recent_signals(limit=10)
        assert len(recent) == 1
        assert recent[0]["ticker"] == "FRO"

    def test_s1_block_skips_headline(self, mock_config):
        """Headlines that fail Stage 1 are counted but produce no signals."""
        rows = [_HeadlineRow(dt="2026-01-01T10:00:00+00:00", headline="Apple iPhone record sales", feed_source="test")]

        with patch("scripts.backtest.pipeline.stage1_matches", return_value=None):
            summary = replay(rows, dry_run=True, max_cost_usd=10.0, skip_outcomes=True)

        assert summary["passed_s1"] == 0
        assert summary["signals_generated"] == 0

    def test_cost_abort_stops_replay(self, mock_config):
        """Replay aborts early when cumulative cost exceeds max_cost_usd."""
        rows = [
            _HeadlineRow(dt=f"2026-01-0{i}T10:00:00+00:00", headline=f"Tanker headline {i}", feed_source="test")
            for i in range(1, 6)
        ]
        # Each signal costs 0.50 USD; limit is 1.00 → should abort after 2nd signal
        pricey_sig = _make_signal(est_cost=0.50)

        with patch("scripts.backtest.pipeline.stage1_matches", return_value="general"), \
             patch("scripts.backtest.pipeline.analyze", return_value=[pricey_sig]):
            summary = replay(rows, dry_run=True, max_cost_usd=1.00, skip_outcomes=True)

        assert summary["cost_aborted"] is True
        assert summary["total_cost_usd"] <= 1.00  # did not exceed limit
        assert summary["signals_generated"] < 5

    def test_skip_outcomes_does_not_call_fetch(self, mock_config):
        """With --skip-outcomes, fetch_prices_for_signal must not be called."""
        rows = [_HeadlineRow(dt="2026-01-01T10:00:00+00:00", headline="Crude sanctions update", feed_source="test")]
        sig = _make_signal()

        with patch("scripts.backtest.pipeline.stage1_matches", return_value="vip"), \
             patch("scripts.backtest.pipeline.analyze", return_value=[sig]), \
             patch("scripts.backtest.fetch_prices_for_signal") as mock_fetch:
            replay(rows, dry_run=False, max_cost_usd=10.0, skip_outcomes=True)

        mock_fetch.assert_not_called()

    def test_analyze_not_called_when_no_s1_match(self, mock_config):
        """pipeline.analyze must not be called if stage1_matches returns None."""
        rows = [_HeadlineRow(dt="2026-01-01T10:00:00+00:00", headline="No keyword match", feed_source="test")]

        with patch("scripts.backtest.pipeline.stage1_matches", return_value=None), \
             patch("scripts.backtest.pipeline.analyze") as mock_analyze:
            replay(rows, dry_run=True, max_cost_usd=10.0, skip_outcomes=True)

        mock_analyze.assert_not_called()


# ---------------------------------------------------------------------------
# performance_report
# ---------------------------------------------------------------------------


def _seed_signal_with_outcome(mock_config, ticker, direction, confidence, pnl_1h, pnl_24h):
    """Insert a signal + outcome into the test DB."""
    sig = _make_signal(ticker=ticker, direction=direction, confidence=confidence)
    sig_id = _db_mod.insert_signal(sig)
    _db_mod.update_signal_prices(
        sig_id,
        {
            "price_at_signal": 10.00,
            "price_1h": 10.00 * (1 + pnl_1h),
            "price_4h": 10.00 * (1 + pnl_1h),
            "price_24h": 10.00 * (1 + pnl_24h),
            "outcome_pnl_1h": pnl_1h,
            "outcome_pnl_24h": pnl_24h,
            "outcome_note": "test",
        },
    )
    return sig_id


class TestPerformanceReport:
    def test_empty_db_prints_no_data_message(self, mock_config, capsys):
        performance_report()
        out = capsys.readouterr().out
        assert "No signals" in out or "no signals" in out.lower() or "match" in out.lower()

    def test_win_rate_calculation(self, mock_config, capsys):
        """3 wins and 1 loss → 75% win rate."""
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 85, 0.01, 0.03)
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 80, 0.02, 0.04)
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 75, 0.01, 0.02)
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 70, -0.01, -0.02)  # loss

        performance_report()
        out = capsys.readouterr().out
        assert "75.0%" in out

    def test_confidence_bands_appear_in_output(self, mock_config, capsys):
        _seed_signal_with_outcome(mock_config, "CVX", "LONG", 92, 0.01, 0.02)
        _seed_signal_with_outcome(mock_config, "CVX", "LONG", 75, 0.01, 0.02)

        performance_report()
        out = capsys.readouterr().out
        assert "90-100%" in out
        assert "70-79%" in out

    def test_ticker_filter(self, mock_config, capsys):
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 85, 0.01, 0.03)
        _seed_signal_with_outcome(mock_config, "CVX", "SHORT", 80, -0.01, -0.02)

        performance_report(ticker_filter="CVX")
        out = capsys.readouterr().out
        assert "CVX" in out
        # FRO should not appear in the per-ticker table when filtering to CVX
        assert "FRO" not in out

    def test_min_confidence_filter(self, mock_config, capsys):
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 55, 0.01, 0.02)
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 82, 0.01, 0.02)

        # With min_confidence=70, only the 82% signal remains
        performance_report(min_confidence=70)
        out = capsys.readouterr().out
        # Should show n=1 (not n=2)
        assert "n=  1" in out or "n=1" in out

    def test_direction_breakdown(self, mock_config, capsys):
        _seed_signal_with_outcome(mock_config, "FRO", "LONG", 85, 0.01, 0.02)
        _seed_signal_with_outcome(mock_config, "DVN", "SHORT", 80, -0.01, -0.01)

        performance_report()
        out = capsys.readouterr().out
        assert "LONG" in out
        assert "SHORT" in out


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_returns_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_missing_csv_file_returns_error(self, tmp_path):
        rc = main(["--csv", str(tmp_path / "missing.csv")])
        assert rc == 1

    def test_report_flag_with_empty_db(self, mock_config, capsys):
        rc = main(["--report"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "signals" in out.lower() or "match" in out.lower()

    def test_csv_replay_summary_printed(self, mock_config, tmp_path, capsys):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "Tanker rates explode on OPEC surprise", "feed_source": "reuters"},
        ])
        sig = _make_signal()
        with patch("scripts.backtest.pipeline.stage1_matches", return_value="vip"), \
             patch("scripts.backtest.pipeline.analyze", return_value=[sig]), \
             patch("scripts.backtest.fetch_prices_for_signal", return_value={}):
            rc = main(["--csv", str(p), "--skip-outcomes"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Replay summary" in out
        assert "signals" in out.lower()

    def test_dry_run_flag_passed_through(self, mock_config, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "Crude flows sanctions relief", "feed_source": "reuters"},
        ])
        with patch("scripts.backtest.replay") as mock_replay:
            mock_replay.return_value = {
                "total_headlines": 1,
                "passed_s1": 1,
                "passed_s2_or_vip": 1,
                "signals_generated": 1,
                "total_cost_usd": 0.001,
                "cost_aborted": False,
                "outcomes_fetched": 0,
            }
            main(["--csv", str(p), "--dry-run"])

        _, kwargs = mock_replay.call_args
        assert kwargs["dry_run"] is True

    def test_max_cost_argument_forwarded(self, mock_config, tmp_path):
        p = _write_csv(tmp_path, [
            {"datetime": "2026-01-01T10:00:00Z", "headline": "OPEC meeting outcome bullish", "feed_source": "x"},
        ])
        with patch("scripts.backtest.replay") as mock_replay:
            mock_replay.return_value = {
                "total_headlines": 1,
                "passed_s1": 0,
                "passed_s2_or_vip": 0,
                "signals_generated": 0,
                "total_cost_usd": 0.0,
                "cost_aborted": False,
                "outcomes_fetched": 0,
            }
            main(["--csv", str(p), "--max-cost", "0.25"])

        _, kwargs = mock_replay.call_args
        assert kwargs["max_cost_usd"] == 0.25
