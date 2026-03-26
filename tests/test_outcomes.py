"""Unit tests for ai_edt/outcomes.py — Phase 3 Outcome Tracking.

All tests mock the yfinance dependency so no network calls are made.
A synthetic 1h price history DataFrame is built around a controlled
signal timestamp so exact price/PnL values can be asserted precisely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai_edt.outcomes import _pnl, fetch_prices_for_signal

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Anchor: the mock "signal was generated at this time"
_BASE_DT = datetime(2026, 3, 25, 10, 0, 0, tzinfo=UTC)

# 27 hourly bars starting 2h before the signal.
# Bar indices relative to _BASE_DT:
#   0  → -2h   (100.0)
#   1  → -1h   (102.0)
#   2  →  0h   (103.0)  ← price_at_signal
#   3  → +1h   (105.0)  ← price_1h
#   4  → +2h   (101.0)
#   5  → +3h   (112.0)
#   6  → +4h   (110.0)  ← price_4h
#  ...
#  26  → +24h  (104.0)  ← price_24h
_PRICES = [
    100.0,
    102.0,
    103.0,
    105.0,
    101.0,
    112.0,
    110.0,
    108.0,
    109.0,
    111.0,
    107.0,
    106.0,
    104.0,
    103.0,
    100.0,
    99.0,
    98.0,
    97.0,
    96.0,
    97.0,
    98.0,
    99.0,
    100.0,
    101.0,
    102.0,
    103.0,
    104.0,
]
assert len(_PRICES) == 27  # 2h before + 0h + 24 * 1h


def _make_yf_mock(
    prices: list[float] | None = None,
    *,
    base_dt: datetime = _BASE_DT,
    tz: str | None = "UTC",
) -> MagicMock:
    """Return a yfinance module mock whose Ticker.history() yields a tidy DataFrame."""
    if prices is None:
        prices = _PRICES

    idx = pd.date_range(
        start=base_dt - timedelta(hours=2),
        periods=len(prices),
        freq="1h",
        tz=tz,
    )
    hist_df = pd.DataFrame({"Close": prices}, index=idx)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist_df

    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf


# ---------------------------------------------------------------------------
# _pnl helper
# ---------------------------------------------------------------------------


class TestPnlCalc:
    def test_long_positive(self) -> None:
        assert _pnl(100.0, 110.0, "LONG") == pytest.approx(0.1, rel=1e-6)

    def test_long_negative(self) -> None:
        assert _pnl(100.0, 90.0, "LONG") == pytest.approx(-0.1, rel=1e-6)

    def test_short_positive(self) -> None:
        """SHORT profits when price falls."""
        assert _pnl(100.0, 90.0, "SHORT") == pytest.approx(0.1, rel=1e-6)

    def test_short_negative(self) -> None:
        """SHORT loses when price rises."""
        assert _pnl(100.0, 110.0, "SHORT") == pytest.approx(-0.1, rel=1e-6)

    def test_zero_entry_returns_zero(self) -> None:
        assert _pnl(0.0, 100.0, "LONG") == 0.0


# ---------------------------------------------------------------------------
# fetch_prices_for_signal
# ---------------------------------------------------------------------------


class TestFetchPricesForSignal:
    def test_all_prices_filled_when_24h_elapsed(self) -> None:
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(hours=26)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_at_signal"] == pytest.approx(103.0)
        assert result["price_1h"] == pytest.approx(105.0)
        assert result["price_4h"] == pytest.approx(110.0)
        assert result["price_24h"] == pytest.approx(104.0)
        assert result["outcome_note"] is None

    def test_long_pnl_24h_calculated_correctly(self) -> None:
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(hours=26)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        p0, p24 = result["price_at_signal"], result["price_24h"]
        assert p0 == pytest.approx(103.0)
        assert p24 == pytest.approx(104.0)
        expected = round((104.0 - 103.0) / 103.0, 6)
        assert result["outcome_pnl_24h"] == pytest.approx(expected, rel=1e-4)

    def test_short_pnl_is_inverted(self) -> None:
        """SHORT PnL must be the mirror of LONG PnL for the same prices."""
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(hours=26)

        long_r = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)
        short_r = fetch_prices_for_signal(
            "FRO", "SHORT", _BASE_DT.isoformat(), now=now, _yf=mock_yf
        )

        assert long_r["outcome_pnl_24h"] == pytest.approx(-short_r["outcome_pnl_24h"], rel=1e-4)

    def test_signal_too_recent_skips_yfinance(self) -> None:
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(minutes=10)  # only 10 min since signal

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["outcome_note"] == "signal_too_recent"
        assert result["price_at_signal"] is None
        mock_yf.Ticker.assert_not_called()

    def test_price_1h_none_when_not_elapsed(self) -> None:
        """With now = signal+45min, price_at_signal is available but price_1h is not."""
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(minutes=45)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_at_signal"] == pytest.approx(103.0)
        assert result["price_1h"] is None
        assert result["outcome_pnl_1h"] is None

    def test_price_4h_none_when_not_elapsed(self) -> None:
        """With now = signal+2h, price_1h is available but price_4h and price_24h are not."""
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(hours=2)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_at_signal"] is not None
        assert result["price_1h"] == pytest.approx(105.0)
        assert result["price_4h"] is None
        assert result["price_24h"] is None

    def test_price_24h_none_when_not_elapsed(self) -> None:
        """With now = signal+6h, price_4h is available but price_24h is not."""
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(hours=6)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_4h"] is not None
        assert result["price_24h"] is None
        assert result["outcome_pnl_24h"] is None

    def test_empty_dataframe_returns_no_price_data(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        now = _BASE_DT + timedelta(hours=26)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["outcome_note"] == "no_price_data"
        assert result["price_at_signal"] is None

    def test_yfinance_exception_captured_in_note(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("network timeout")
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        now = _BASE_DT + timedelta(hours=26)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["outcome_note"] is not None
        assert result["outcome_note"].startswith("yfinance_error")
        assert result["price_at_signal"] is None

    def test_outcome_pnl_1h_none_when_price_1h_none(self) -> None:
        """outcome_pnl_1h must be None whenever price_1h is None."""
        mock_yf = _make_yf_mock()
        now = _BASE_DT + timedelta(minutes=45)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_1h"] is None
        assert result["outcome_pnl_1h"] is None

    def test_tz_naive_index_is_localized_to_utc(self) -> None:
        """A tz-naive DataFrame index from yfinance must be handled without error."""
        mock_yf = _make_yf_mock(tz=None)  # no timezone on index
        now = _BASE_DT + timedelta(hours=26)

        result = fetch_prices_for_signal("FRO", "LONG", _BASE_DT.isoformat(), now=now, _yf=mock_yf)

        assert result["price_at_signal"] is not None
        assert result["outcome_note"] is None
